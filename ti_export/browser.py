from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions

from .models import BrowserCfg

log = logging.getLogger("ti_export")


def make_driver(cfg: BrowserCfg, download_dir: Path) -> webdriver.Remote:
    """Attach to a browser you already logged in to (MFA done by you), or launch with a profile."""
    download_dir.mkdir(parents=True, exist_ok=True)
    opts = ChromeOptions() if cfg.name == "chrome" else EdgeOptions()
    if cfg.mode == "attach":
        opts.debugger_address = cfg.debugger_address
    else:
        if cfg.user_data_dir:
            opts.add_argument(f"--user-data-dir={Path(cfg.user_data_dir).resolve()}")
        opts.add_experimental_option("prefs", {
            "download.default_directory": str(download_dir.resolve()),
            "download.prompt_for_download": False,
        })
    driver = webdriver.Chrome(options=opts) if cfg.name == "chrome" else webdriver.Edge(options=opts)
    try:  # needed for attach mode, where prefs cannot be passed
        driver.execute_cdp_cmd("Browser.setDownloadBehavior", {
            "behavior": "allow", "downloadPath": str(download_dir.resolve()),
        })
    except Exception as exc:
        log.warning("Could not set download directory via CDP (%s); downloads go to the browser default", exc)
    return driver


def xlsx_snapshot(folder: Path) -> set[str]:
    return {p.name for p in folder.glob("*.xlsx")}


def wait_for_xlsx(folder: Path, before: set[str], timeout: float) -> Path:
    """Wait for a new, fully written .xlsx (no .crdownload present, size stable)."""
    deadline = time.monotonic() + timeout
    last_size = -1
    while time.monotonic() < deadline:
        new = [p for p in folder.glob("*.xlsx") if p.name not in before]
        if new and not any(folder.glob("*.crdownload")):
            p = max(new, key=lambda x: x.stat().st_mtime)
            size = p.stat().st_size
            if size > 0 and size == last_size:
                return p
            last_size = size
        time.sleep(0.5)
    raise TimeoutError(f"no completed .xlsx appeared in {folder} within {timeout}s")


def move_into_place(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.move(str(src), str(dest))
    return dest
