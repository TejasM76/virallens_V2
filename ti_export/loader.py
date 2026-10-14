from __future__ import annotations

from itertools import product

from .models import Config, Job


def build_jobs(cfg: Config) -> list[Job]:
    """family x company x location, ordered so one company/family is finished before the next."""
    return [
        Job(function=cfg.function, family=fam, company=co, location=loc, skills=tuple(skills), industry=cfg.industry)
        for (fam, skills), co, loc in product(cfg.job_families.items(), cfg.companies, cfg.locations)
    ]
