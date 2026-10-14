import json
from pathlib import Path

import pytest
import yaml

from ti_export.loader import build_jobs
from ti_export.models import Selectors, load_config, load_selectors
from ti_export.runner import Manifest, fill_option, main, xpath_literal

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config(ROOT / "config.yaml")


def test_config_has_four_implement_families_with_eight_skills(cfg):
    assert list(cfg.job_families) == [
        "ERP Implementation", "Cloud Implementation", "System Integration", "Financial Planning & Analysis",
    ]
    assert all(len(s) == 8 for s in cfg.job_families.values())


def test_job_matrix_and_export_name(cfg):
    jobs = build_jobs(cfg)
    assert len(jobs) == 4 * 4 * 13
    assert len({j.key for j in jobs}) == len(jobs)
    j = next(j for j in jobs if j.family == "Financial Planning & Analysis" and j.location.name == "Bengaluru")
    assert j.export_name == "Accenture_Implement_Financial_Planning_Analysis_Bengaluru"
    assert "Greater Bengaluru" in j.location.queries


def test_location_without_aliases_uses_name(cfg):
    assert cfg.companies[1].queries == ("TCS",)


def test_dry_run_needs_no_placeholders(capsys):
    assert main(["--config", str(ROOT / "config.yaml"), "--dry-run", "--company", "TCS", "--location", "Pune",
                 "--family", "Cloud Implementation"]) == 0
    out = capsys.readouterr().out
    assert "TCS_Implement_Cloud_Implementation_Pune" in out and "Terraform" in out


def test_real_run_refuses_placeholders(tmp_path):
    (tmp_path / "selectors.yaml").write_text((ROOT / "selectors.yaml").read_text())
    c = yaml.safe_load((ROOT / "config.yaml").read_text())
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(c))
    assert main(["--config", str(tmp_path / "config.yaml")]) == 2


def test_shipped_selectors_are_unresolved():
    bad = load_selectors(ROOT / "selectors.yaml").unresolved()
    assert {"nav_business", "export_save", "industry.input", "location.option"} <= set(bad)


def test_resolved_selectors_pass():
    x = {"input": "//i", "option": "//o[{value}]"}
    keys = dict(nav_business="//n", nav_talent_insights="//t", page_ready="//p", results_ready="//r",
                export_button="//e", export_name_input="//n", export_select_all="//a",
                export_format_xlsx="//x", export_save="//s")
    sel = Selectors(location=x, skills=x, industry=x, company=x, **keys)
    assert sel.unresolved() == []


def test_xpath_literal():
    assert xpath_literal("EY") == "'EY'"
    assert xpath_literal("it's") == '"it\'s"'
    assert xpath_literal("a'b\"c") == "concat('a', \"'\", 'b\"c')"
    assert fill_option("//li[.={value}]", "TCS") == "//li[.='TCS']"


def test_manifest_resume(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    m.record("k", status="done")
    m.record("f", status="failed")
    m2 = Manifest(tmp_path / "manifest.json")
    assert m2.is_done("k") and not m2.is_done("f")
    assert json.loads((tmp_path / "manifest.json").read_text())["k"]["status"] == "done"
