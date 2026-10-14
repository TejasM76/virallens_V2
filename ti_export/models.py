from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

XPATH_PLACEHOLDER = "PASTE_XPATH_HERE"


class Target(BaseModel, frozen=True):
    """A company or location. `name` labels files; `aliases` are every spelling typed into the filter."""

    name: str
    aliases: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _from_str(cls, v):
        return {"name": v} if isinstance(v, str) else v

    @property
    def queries(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.aliases or (self.name,)))


class Job(BaseModel, frozen=True):
    function: str
    family: str
    company: Target
    location: Target
    skills: tuple[str, ...]
    industry: str

    @property
    def key(self) -> str:
        return f"{self.family}|{self.company.name}|{self.location.name}"

    @property
    def export_name(self) -> str:
        """companyname_function_jobfamily_locationname"""
        return "_".join(slug(p) for p in (self.company.name, self.function, self.family, self.location.name))

    @property
    def filename(self) -> str:
        return f"{self.export_name}.xlsx"


class BrowserCfg(BaseModel):
    name: Literal["chrome", "edge"] = "chrome"
    mode: Literal["attach", "launch"] = "attach"
    debugger_address: str = "127.0.0.1:9222"
    user_data_dir: str | None = None


class PacingCfg(BaseModel):
    min_delay_s: float = Field(20, ge=0)
    max_delay_s: float = Field(60, ge=0)
    max_exports_per_session: int = Field(25, ge=1)
    retries: int = Field(2, ge=0)
    backoff_s: float = Field(15, ge=0)
    wait_timeout_s: float = Field(30, gt=0)
    download_timeout_s: float = Field(120, gt=0)
    min_skills_matched: int = Field(1, ge=1)

    @model_validator(mode="after")
    def _order(self):
        if self.max_delay_s < self.min_delay_s:
            raise ValueError("max_delay_s must be >= min_delay_s")
        return self


class Config(BaseModel):
    start_url: str = "https://www.linkedin.com/feed/"
    function: str = "Implement"
    job_families: dict[str, list[str]]
    industry: str
    companies: list[Target]
    locations: list[Target]
    selectors_file: Path = Path("selectors.yaml")
    output_dir: Path = Path("out")
    merge_header_row: int = 0
    browser: BrowserCfg = BrowserCfg()
    pacing: PacingCfg = PacingCfg()

    @field_validator("job_families")
    @classmethod
    def _skills(cls, v: dict[str, list[str]]):
        for fam, skills in v.items():
            if not skills or len(set(skills)) != len(skills):
                raise ValueError(f"'{fam}': skills must be non-empty and unique")
        return v

    @field_validator("companies", "locations")
    @classmethod
    def _unique(cls, v: list[Target]):
        names = [t.name for t in v]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate names: {names}")
        return v


class FilterXPaths(BaseModel):
    open: str | None = None  # clicked only if `input` is not visible
    input: str
    option: str  # must contain {value}; replaced by an XPath string literal
    confirm: str | None = None  # e.g. a per-filter "Apply" button

    @field_validator("option")
    @classmethod
    def _has_value(cls, v: str):
        if v != XPATH_PLACEHOLDER and "{value}" not in v:
            raise ValueError("option XPath must contain {value}")
        return v


class Selectors(BaseModel):
    nav_business: str  # "For Business" / "Business" menu on the LinkedIn top bar
    nav_talent_insights: str  # "Talent Insights" entry in that menu
    page_ready: str  # element proving the Talent Insights report page has loaded
    location: FilterXPaths
    skills: FilterXPaths
    industry: FilterXPaths
    company: FilterXPaths
    skills_match_any: str | None = None  # toggle to switch the skill filter to ANY, if the UI has one
    apply_filters: str | None = None
    results_ready: str
    export_button: str
    export_name_input: str
    export_select_all: str
    export_format_xlsx: str
    export_save: str

    def unresolved(self) -> list[str]:
        """Dotted paths of every field still empty or left at the placeholder."""
        required = {
            "nav_business", "nav_talent_insights", "page_ready", "results_ready", "export_button",
            "export_name_input", "export_select_all", "export_format_xlsx", "export_save",
            *(f"{sec}.{f}" for sec in ("location", "skills", "industry", "company") for f in ("input", "option")),
        }
        bad: list[str] = []
        flat = {
            k: getattr(self, k) for k in (
                "nav_business", "nav_talent_insights", "page_ready", "results_ready", "export_button",
                "export_name_input", "export_select_all", "export_format_xlsx", "export_save",
                "skills_match_any", "apply_filters",
            )
        }
        for sec in ("location", "skills", "industry", "company"):
            for field, val in getattr(self, sec).model_dump().items():
                flat[f"{sec}.{field}"] = val
        for path, val in flat.items():
            if val == XPATH_PLACEHOLDER or (path in required and not val):
                bad.append(path)
        return bad


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is empty or not a mapping")
    return data


def load_config(path: Path) -> Config:
    path = path.resolve()
    cfg = Config.model_validate(_read_yaml(path))
    base = path.parent
    cfg.selectors_file = cfg.selectors_file if cfg.selectors_file.is_absolute() else base / cfg.selectors_file
    cfg.output_dir = cfg.output_dir if cfg.output_dir.is_absolute() else base / cfg.output_dir
    return cfg


def load_selectors(path: Path) -> Selectors:
    return Selectors.model_validate(_read_yaml(path))
