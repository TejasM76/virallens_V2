from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from selenium.common.exceptions import (
    ElementClickInterceptedException, StaleElementReferenceException,
    TimeoutException, WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .browser import make_driver, move_into_place, wait_for_xlsx, xlsx_snapshot
from .loader import build_jobs
from .models import Config, FilterXPaths, Job, Selectors, load_config, load_selectors, slug

log = logging.getLogger("ti_export")
AUTH_WALL_MARKERS = ("/login", "/checkpoint", "/authwall", "/uas/")


class JobError(Exception):
    """A single job failed; retry or move on."""


class AbortRun(Exception):
    """Stop everything (login wall / verification challenge). Never retried or bypassed."""


def xpath_literal(s: str) -> str:
    if "'" not in s:
        return f"'{s}'"
    if '"' not in s:
        return f'"{s}"'
    return "concat(" + ', "\'", '.join(f"'{p}'" for p in s.split("'")) + ")"


def fill_option(template: str, value: str) -> str:
    return template.replace("{value}", xpath_literal(value))


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = json.loads(path.read_text("utf-8")) if path.exists() else {}

    def is_done(self, key: str) -> bool:
        return self.data.get(key, {}).get("status") == "done"

    def record(self, key: str, **fields) -> None:
        entry = self.data.setdefault(key, {})
        entry.update(fields, updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), "utf-8")
        os.replace(tmp, self.path)


class Exporter:
    """LinkedIn -> Business -> Talent Insights once per session, then one clean reload per job."""

    def __init__(self, cfg: Config, sel: Selectors, driver, manifest: Manifest):
        self.cfg, self.sel, self.driver, self.manifest = cfg, sel, driver, manifest
        self.wait = WebDriverWait(driver, cfg.pacing.wait_timeout_s)
        self.incoming = cfg.output_dir / "_incoming"
        self.failures = cfg.output_dir / "_failures"
        self.ti_handle: str | None = None
        self.ti_url: str | None = None  # Talent Insights landing URL with no filters; reloaded per job

    # --- primitives -----------------------------------------------------
    def _click(self, xpath: str):
        el = self.wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
        try:
            el.click()
        except (ElementClickInterceptedException, StaleElementReferenceException):
            self.driver.execute_script("arguments[0].click();", el)
        return el

    def _guard_session(self) -> None:
        if any(m in self.driver.current_url for m in AUTH_WALL_MARKERS):
            raise AbortRun(f"Redirected to {self.driver.current_url}. Resolve login/verification manually, then rerun.")

    def _visible_input(self, fx: FilterXPaths):
        for el in self.driver.find_elements(By.XPATH, fx.input):
            if el.is_displayed():
                return el
        return None

    def _select(self, fx: FilterXPaths, value: str, seen: set[str]) -> bool:
        """Type `value`, click its option. False if the option was already chosen (same label)."""
        box = self._visible_input(fx)
        if box is None and fx.open:
            self._click(fx.open)
        box = box or self.wait.until(EC.visibility_of_element_located((By.XPATH, fx.input)))
        box.send_keys(Keys.CONTROL, "a", Keys.DELETE)
        box.send_keys(value)
        option = self.wait.until(EC.element_to_be_clickable((By.XPATH, fill_option(fx.option, value))))
        label = " ".join(option.text.split()) or value
        if label in seen:
            return False
        try:
            option.click()
        except (ElementClickInterceptedException, StaleElementReferenceException):
            self.driver.execute_script("arguments[0].click();", option)
        seen.add(label)
        if fx.confirm:
            self._click(fx.confirm)
        return True

    def _select_many(self, fx: FilterXPaths, values: tuple[str, ...], label: str, job: Job) -> list[str]:
        """Select every value that LinkedIn knows; returns the values that matched."""
        matched, seen = [], set()
        for v in values:
            try:
                if self._select(fx, v, seen):
                    matched.append(v)
            except TimeoutException:
                log.warning("[%s] %s not found in LinkedIn: %s", job.key, label, v)
        return matched

    # --- session start --------------------------------------------------
    def open_talent_insights(self) -> None:
        self.driver.get(self.cfg.start_url)
        self._guard_session()
        before = set(self.driver.window_handles)
        self._click(self.sel.nav_business)
        self._click(self.sel.nav_talent_insights)
        try:  # Talent Insights may open in a new tab
            WebDriverWait(self.driver, 5).until(lambda d: len(d.window_handles) > len(before))
            self.driver.switch_to.window(next(h for h in self.driver.window_handles if h not in before))
        except TimeoutException:
            pass
        self._guard_session()
        self.wait.until(EC.presence_of_element_located((By.XPATH, self.sel.page_ready)))
        self.ti_handle, self.ti_url = self.driver.current_window_handle, self.driver.current_url
        log.info("Talent Insights ready at %s", self.ti_url)

    def _fresh_page(self) -> None:
        if self.ti_url is None:
            self.open_talent_insights()
            return
        self.driver.switch_to.window(self.ti_handle)
        self.driver.get(self.ti_url)
        self._guard_session()
        self.wait.until(EC.presence_of_element_located((By.XPATH, self.sel.page_ready)))

    # --- one job --------------------------------------------------------
    def run_job(self, job: Job) -> tuple[Path, dict]:
        try:
            self._fresh_page()

            locations = self._select_many(self.sel.location, job.location.queries, "location", job)
            if not locations:
                raise JobError(f"no location alias matched: {list(job.location.queries)}")

            if self.sel.skills_match_any:
                self._click(self.sel.skills_match_any)
            skills = self._select_many(self.sel.skills, job.skills, "skill", job)
            if len(skills) < self.cfg.pacing.min_skills_matched:
                raise JobError(f"only {len(skills)} skill(s) matched; need {self.cfg.pacing.min_skills_matched}")

            if not self._select_many(self.sel.industry, (job.industry,), "industry", job):
                raise JobError(f"industry not found: {job.industry}")
            if not self._select_many(self.sel.company, job.company.queries, "company", job):
                raise JobError(f"company not found: {list(job.company.queries)}")

            if self.sel.apply_filters:
                self._click(self.sel.apply_filters)
            self.wait.until(EC.presence_of_element_located((By.XPATH, self.sel.results_ready)))

            self.incoming.mkdir(parents=True, exist_ok=True)
            before = xlsx_snapshot(self.incoming)
            self._click(self.sel.export_button)
            name_box = self.wait.until(EC.visibility_of_element_located((By.XPATH, self.sel.export_name_input)))
            name_box.send_keys(Keys.CONTROL, "a", Keys.DELETE)
            name_box.send_keys(job.export_name)
            self._click(self.sel.export_select_all)
            self._click(self.sel.export_format_xlsx)
            self._click(self.sel.export_save)
            downloaded = wait_for_xlsx(self.incoming, before, self.cfg.pacing.download_timeout_s)
        except TimeoutException as exc:
            raise JobError(f"timed out waiting for UI element: {exc.msg or exc}") from exc
        except TimeoutError as exc:
            raise JobError(str(exc)) from exc
        except WebDriverException as exc:
            raise JobError(f"browser error: {exc.msg or exc}") from exc

        dest = move_into_place(downloaded, self.cfg.output_dir / slug(job.family) / job.filename)
        unmatched_skills = [s for s in job.skills if s not in skills]
        unmatched_locs = [a for a in job.location.queries if a not in locations]
        return dest, {"unmatched_skills": unmatched_skills, "unmatched_locations": unmatched_locs}

    def screenshot(self, job: Job, attempt: int) -> None:
        try:
            self.failures.mkdir(parents=True, exist_ok=True)
            self.driver.save_screenshot(str(self.failures / f"{job.export_name}__a{attempt}.png"))
        except Exception:
            log.debug("screenshot failed", exc_info=True)

    def run_with_retries(self, job: Job) -> None:
        p = self.cfg.pacing
        last_err = ""
        for attempt in range(1, p.retries + 2):
            try:
                dest, extra = self.run_job(job)
            except JobError as exc:
                last_err = str(exc)
                log.error("[%s] attempt %d failed: %s", job.key, attempt, last_err)
                self.screenshot(job, attempt)
                if attempt <= p.retries:
                    time.sleep(p.backoff_s * attempt)
                continue
            self.manifest.record(
                job.key, status="done", attempts=attempt, file=str(dest.relative_to(self.cfg.output_dir)),
                skills_requested=list(job.skills), error=None, function=job.function,
                family=job.family, company=job.company.name, location=job.location.name, **extra,
            )
            log.info("[%s] done -> %s (unmatched skills=%d, locations=%d)", job.key, dest.name,
                     len(extra["unmatched_skills"]), len(extra["unmatched_locations"]))
            return
        self.manifest.record(
            job.key, status="failed", attempts=p.retries + 1, error=last_err,
            function=job.function, family=job.family, company=job.company.name, location=job.location.name,
        )


def select_jobs(jobs: list[Job], args: argparse.Namespace) -> list[Job]:
    def keep(value: str, wanted: list[str] | None) -> bool:
        return not wanted or value.casefold() in {w.casefold() for w in wanted}
    return [j for j in jobs if keep(j.family, args.family) and keep(j.company.name, args.company)
            and keep(j.location.name, args.location)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LinkedIn Talent Insights export runner (Selenium)")
    ap.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="print the job matrix; no browser, placeholders allowed")
    ap.add_argument("--limit", type=int, help="max exports this run (also capped by pacing.max_exports_per_session)")
    ap.add_argument("--family", action="append")
    ap.add_argument("--company", action="append")
    ap.add_argument("--location", action="append")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(cfg.output_dir / "run.log", encoding="utf-8")],
    )

    jobs = select_jobs(build_jobs(cfg), args)
    manifest = Manifest(cfg.output_dir / "manifest.json")
    pending = [j for j in jobs if not manifest.is_done(j.key)]
    log.info("%d jobs selected, %d already done, %d pending", len(jobs), len(jobs) - len(pending), len(pending))

    if args.dry_run:
        for j in pending:
            print(f"{j.export_name}\n    locations={list(j.location.queries)}\n    skills={list(j.skills)}")
        return 0

    sel = load_selectors(cfg.selectors_file)
    problems = [f"selectors.yaml: {p}" for p in sel.unresolved()]
    if problems:
        log.error("Fill these placeholders first:\n  %s", "\n  ".join(problems))
        return 2

    cap = min(args.limit or cfg.pacing.max_exports_per_session, cfg.pacing.max_exports_per_session)
    driver = make_driver(cfg.browser, cfg.output_dir / "_incoming")
    exporter = Exporter(cfg, sel, driver, manifest)
    done = 0
    try:
        for i, job in enumerate(pending[:cap]):
            exporter.run_with_retries(job)
            done += 1
            if i < min(len(pending), cap) - 1:
                time.sleep(random.uniform(cfg.pacing.min_delay_s, cfg.pacing.max_delay_s))
    except AbortRun as exc:
        log.error("ABORTED: %s", exc)
        return 3
    except KeyboardInterrupt:
        log.warning("Interrupted; progress is saved in manifest.json")
        return 130
    finally:
        if cfg.browser.mode == "launch":
            driver.quit()
    remaining = len(pending) - done
    log.info("Session finished: %d attempted, %d still pending%s", done, remaining,
             " (session cap reached; rerun later)" if remaining else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
