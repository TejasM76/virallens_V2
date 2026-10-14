from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

XPATH_PLACEHOLDER = "PASTE_XPATH_HERE"
URL_PLACEHOLDER = "PASTE_REPORT_URL_HERE"


class Target(BaseModel, frozen=True):
    """A company or location. `name` labels files; `search` is what gets typed into the filter."""

    name: str
    search: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_str(cls, v):
        return {"name": v} if isinstance(v, str) else v

    @property
    def query(self) -> str:
        return self.search or self.name


class Job(BaseModel, frozen=True):
    family: str
    company: Target
    location: Target
    skills: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.family}|{self.company.name}|{self.location.name}"

    @property
    def filename(self) -> str:
        return f"{slug(self.company.name)}__{slug(self.location.name)}.xlsx"


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
    skills_workbook: Path
    sheet: str | int = 0
    function: str = "Implement"
    max_skill_columns: int = Field(8, ge=1)
    companies: list[Target]
    locations: list[Target]
    report_url: str
    selectors_file: Path = Path("selectors.yaml")
    output_dir: Path = Path("out")
    merge_header_row: int = 0
    browser: BrowserCfg = BrowserCfg()
    pacing: PacingCfg = PacingCfg()

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
    page_ready: str
    company: FilterXPaths
    location: FilterXPaths
    skills: FilterXPaths
    skills_match_any: str | None = None  # toggle to switch the skill filter to ANY, if the UI has one
    apply_filters: str | None = None
    results_ready: str
    export_button: str
    export_confirm: str | None = None

    def unresolved(self) -> list[str]:
        """Dotted paths of every field still empty or left at the placeholder."""
        required = {
            "page_ready", "results_ready", "export_button",
            "company.input", "company.option", "location.input", "location.option",
            "skills.input", "skills.option",
        }
        bad: list[str] = []
        flat = {
            "page_ready": self.page_ready, "results_ready": self.results_ready,
            "export_button": self.export_button, "skills_match_any": self.skills_match_any,
            "apply_filters": self.apply_filters, "export_confirm": self.export_confirm,
        }
        for sec in ("company", "location", "skills"):
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
    cfg.skills_workbook = cfg.skills_workbook if cfg.skills_workbook.is_absolute() else base / cfg.skills_workbook
    cfg.selectors_file = cfg.selectors_file if cfg.selectors_file.is_absolute() else base / cfg.selectors_file
    cfg.output_dir = cfg.output_dir if cfg.output_dir.is_absolute() else base / cfg.output_dir
    return cfg


def load_selectors(path: Path) -> Selectors:
    return Selectors.model_validate(_read_yaml(path))
