from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .models import load_config

log = logging.getLogger("ti_export")


def merge(config_path: Path, out_name: str = "combined.xlsx") -> Path:
    cfg = load_config(config_path)
    manifest = json.loads((cfg.output_dir / "manifest.json").read_text("utf-8"))
    frames: dict[str, list[pd.DataFrame]] = {}
    audit = []
    for key, e in manifest.items():
        audit.append({"job": key, **{k: e.get(k) for k in (
            "status", "attempts", "file", "unmatched_skills", "error", "updated_at")}})
        if e.get("status") != "done":
            continue
        path = cfg.output_dir / e["file"]
        for sheet, df in pd.read_excel(path, sheet_name=None, header=cfg.merge_header_row).items():
            df.insert(0, "location", e["location"])
            df.insert(0, "company", e["company"])
            df.insert(0, "job_family", e["family"])
            frames.setdefault(sheet, []).append(df)

    out = cfg.output_dir / out_name
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for sheet, dfs in frames.items():
            pd.concat(dfs, ignore_index=True).to_excel(xw, sheet_name=sheet[:31], index=False)
        audit_df = pd.DataFrame(audit)
        audit_df["unmatched_skills"] = audit_df["unmatched_skills"].map(
            lambda v: "; ".join(v) if isinstance(v, list) else v)
        audit_df.to_excel(xw, sheet_name="Run Audit", index=False)
    log.info("Wrote %s (%d exports merged)", out, sum(len(v) for v in frames.values()))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    merge(ap.parse_args().config)
