"""Run on the laptop against the logged-in debug-port browser. Read-only: never clicks or types.

  python -m ti_export.probe dump  [--out out/probe.txt]   # list visible interactive elements + suggested XPaths
  python -m ti_export.probe check                         # test every filled-in selectors.yaml XPath on the current page
"""
from __future__ import annotations

import argparse
from pathlib import Path

from selenium.webdriver.common.by import By

from .browser import make_driver
from .models import XPATH_PLACEHOLDER, load_config, load_selectors
from .runner import fill_option

JS_DUMP = r"""
const q = 'button,a,input,textarea,select,label,[role=button],[role=option],[role=menuitem],[role=checkbox],[role=tab],[role=combobox],li';
const lit = s => s.includes("'") ? (s.includes('"') ? null : '"'+s+'"') : "'"+s+"'";
const out = [];
for (const el of document.querySelectorAll(q)) {
  const r = el.getBoundingClientRect();
  if (!r.width || !r.height || getComputedStyle(el).visibility === 'hidden') continue;
  const tag = el.tagName.toLowerCase();
  const text = (el.innerText || el.value || '').replace(/\s+/g, ' ').trim().slice(0, 60);
  const a = n => el.getAttribute(n) || '';
  let xp = '';
  if (a('id') && !/\d{3,}/.test(a('id'))) xp = `//*[@id=${lit(a('id'))}]`;
  else if (a('data-test-id')) xp = `//${tag}[@data-test-id=${lit(a('data-test-id'))}]`;
  else if (a('aria-label') && lit(a('aria-label'))) xp = `//${tag}[@aria-label=${lit(a('aria-label'))}]`;
  else if (a('placeholder') && lit(a('placeholder'))) xp = `//${tag}[@placeholder=${lit(a('placeholder'))}]`;
  else if (text && lit(text) && ['button','a','label','li'].includes(tag)) xp = `//${tag}[normalize-space(.)=${lit(text)}]`;
  out.push([tag, a('role'), text, a('aria-label'), a('placeholder'), a('type'), xp]);
}
return out;
"""


def dump(driver, path: Path) -> None:
    rows = driver.execute_script(JS_DUMP)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {driver.title} | {driver.current_url.split('?')[0]}", "tag\trole\ttext\taria-label\tplaceholder\ttype\tsuggested_xpath"]
    lines += ["\t".join(r) for r in rows]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(rows)} elements -> {path}\nReview it (page text may include your name/profile) before sharing.")


def check(driver, cfg_path: Path) -> int:
    cfg = load_config(cfg_path)
    sel = load_selectors(cfg.selectors_file).model_dump()
    sample = {"location": cfg.locations[0].queries[0], "skills": next(iter(cfg.job_families.values()))[0],
              "industry": cfg.industry, "company": cfg.companies[0].queries[0]}
    bad = 0
    def test(path: str, xp: str | None) -> None:
        nonlocal bad
        if not xp or xp == XPATH_PLACEHOLDER:
            print(f"  --   {path}: not set")
            return
        try:
            els = driver.find_elements(By.XPATH, xp)
        except Exception as exc:
            bad += 1; print(f"  ERR  {path}: invalid XPath ({type(exc).__name__})"); return
        vis = sum(e.is_displayed() for e in els)
        flag = "OK " if vis else "NONE"
        bad += not vis
        print(f"  {flag} {path}: {len(els)} match(es), {vis} visible")
    for key, val in sel.items():
        if isinstance(val, dict):
            for f, xp in val.items():
                test(f"{key}.{f}", fill_option(xp, sample[key]) if f == "option" and xp else xp)
        else:
            test(key, val)
    print("Note: option/results/export-dialog XPaths only match once that part of the UI is open; open it by hand first.")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["dump", "check"])
    ap.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.browser.mode != "attach":
        raise SystemExit("probe needs browser.mode: attach (use your logged-in debug-port browser)")
    driver = make_driver(cfg.browser, cfg.output_dir / "_incoming")
    if args.mode == "dump":
        dump(driver, args.out or cfg.output_dir / "probe.txt")
        return 0
    return check(driver, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
