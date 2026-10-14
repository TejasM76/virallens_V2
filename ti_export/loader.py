from __future__ import annotations

import logging
import re
from itertools import product

import pandas as pd

from .models import Config, Job

log = logging.getLogger("ti_export")
SKILL_COL = re.compile(r"^skill\s*(\d+)$", re.IGNORECASE)


def _clean(v) -> str:
    return "" if pd.isna(v) else " ".join(str(v).split())


def load_family_skills(cfg: Config) -> dict[str, list[str]]:
    """Job Family -> ordered, de-duplicated Skill 1..N for rows of `cfg.function`."""
    df = pd.read_excel(cfg.skills_workbook, sheet_name=cfg.sheet, dtype=str)
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    missing = {"function", "job family"} - set(lower)
    if missing:
        raise ValueError(f"{cfg.skills_workbook.name}: missing column(s) {sorted(missing)}; found {list(df.columns)}")
    func_col, fam_col = lower["function"], lower["job family"]

    skill_cols = sorted((int(m.group(1)), c) for c in df.columns if (m := SKILL_COL.match(c)))
    skill_cols = [c for n, c in skill_cols if n <= cfg.max_skill_columns]
    if not skill_cols:
        raise ValueError("no 'Skill N' columns found")

    df[func_col] = df[func_col].ffill()  # tolerate merged Function cells
    rows = df[df[func_col].map(_clean).str.casefold() == cfg.function.casefold()]

    out: dict[str, list[str]] = {}
    for _, row in rows.iterrows():
        family = _clean(row[fam_col])
        if not family:
            continue
        if family in out:
            raise ValueError(f"duplicate Job Family '{family}' under function '{cfg.function}'")
        skills = list(dict.fromkeys(s for c in skill_cols if (s := _clean(row[c]))))
        if not skills:
            log.warning("Job Family '%s' has no skills in columns 1..%d; skipped", family, cfg.max_skill_columns)
            continue
        out[family] = skills
    if not out:
        raise ValueError(f"no rows with Function == '{cfg.function}'")
    return out


def build_jobs(cfg: Config, family_skills: dict[str, list[str]]) -> list[Job]:
    return [
        Job(family=fam, company=co, location=loc, skills=tuple(skills))
        for (fam, skills), co, loc in product(family_skills.items(), cfg.companies, cfg.locations)
    ]
