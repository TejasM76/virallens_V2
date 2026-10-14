import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ti_export.loader import build_jobs, load_family_skills
from ti_export.models import Selectors, load_config, load_selectors
from ti_export.runner import Manifest, fill_option, main, xpath_literal

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path):
    rows = [
        ["Advise", "Data Scientist", "Data Science"] + ["x"] * 7,
        ["Implement", "ERP Implementation", "SAP S/4HANA", "SAP FICO", "SAP ABAP", "", "Oracle\nFusion  Applications",
         "Workday", "Microsoft Dynamics 365", "ERP Implementations", "IGNORED_9", "SAP FICO"],
        [None, "Cloud Implementation", "AWS", "Azure", "GCP", "Cloud Migration", "Kubernetes", "Terraform", "Docker", "CI/CD", "IGNORED_9", None],
    ]
    cols = ["Function", "Job Family"] + [f"Skill {i}" for i in range(1, 11)]
    rows = [r + [None] * (len(cols) - len(r)) for r in rows]
    pd.DataFrame(rows, columns=cols).to_excel(tmp_path / "skills.xlsx", index=False)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["skills_workbook"] = "skills.xlsx"
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    (tmp_path / "selectors.yaml").write_text((ROOT / "selectors.yaml").read_text())
    return tmp_path


def test_loader_uses_only_skill_1_to_8_and_implement(workspace):
    cfg = load_config(workspace / "config.yaml")
    fs = load_family_skills(cfg)
    assert list(fs) == ["ERP Implementation", "Cloud Implementation"]  # merged-cell Function ffilled, Advise dropped
    assert fs["ERP Implementation"] == [
        "SAP S/4HANA", "SAP FICO", "SAP ABAP", "Oracle Fusion Applications", "Workday", "Microsoft Dynamics 365", "ERP Implementations",
    ]
    assert "IGNORED_9" not in sum(fs.values(), [])
    assert len(fs["Cloud Implementation"]) == 8


def test_job_matrix_size(workspace):
    cfg = load_config(workspace / "config.yaml")
    jobs = build_jobs(cfg, load_family_skills(cfg))
    assert len(jobs) == 2 * 4 * 13
    assert len({j.key for j in jobs}) == len(jobs)


def test_dry_run_needs_no_placeholders(workspace, capsys):
    assert main(["--config", str(workspace / "config.yaml"), "--dry-run", "--limit", "1"]) == 0
    assert "ERP Implementation|Accenture|Bengaluru" in capsys.readouterr().out


def test_real_run_refuses_placeholders(workspace):
    assert main(["--config", str(workspace / "config.yaml")]) == 2


def test_shipped_selectors_are_all_unresolved():
    sel = load_selectors(ROOT / "selectors.yaml")
    assert "export_button" in sel.unresolved() and "skills.option" in sel.unresolved()


def test_resolved_selectors_pass():
    x = {"input": "//i", "option": "//o[{value}]"}
    sel = Selectors(page_ready="//a", company=x, location=x, skills=x, results_ready="//r", export_button="//e")
    assert sel.unresolved() == []


def test_xpath_literal():
    assert xpath_literal("EY GDS") == "'EY GDS'"
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
