# ti_export: LinkedIn Talent Insights export automation (Python + Selenium)

Exports one xlsx per **Job Family × Company × Location** for the `Implement` function, using Skill 1–8 of each family as the skill filter, then merges everything into one workbook. No Java needed: Selenium 4 manages the driver itself.

> LinkedIn's User Agreement restricts automated access. Run only on your own licensed account, keep the pacing defaults, and note the runner stops (never bypasses) on any login/verification page.

## Setup (company laptop)
```bash
pip install -r ti_export/requirements.txt
# put the skills workbook at ti_export/input/Job Family Skills for Talent Insights 7419 1.xlsx
```
1. `ti_export/config.yaml`: paste the report link into `report_url`; add the 14th location if there is one.
2. `ti_export/selectors.yaml`: replace every `PASTE_XPATH_HERE`. Option XPaths use `{value}`, e.g. `//li[@role='option'][contains(normalize-space(.), {value})]`. Optional fields set to `""` are skipped.
3. Start Chrome (or Edge, set `browser.name: edge`) with a debug port, log in to Talent Insights yourself, and leave it open:
   ```bash
   chrome.exe --remote-debugging-port=9222 --user-data-dir=C:\ti-profile
   ```

## Run (from the repo root)
```bash
python -m ti_export.runner --dry-run                 # job matrix + parsed skills, no browser
python -m ti_export.runner --limit 2 --company TCS --location Pune   # smoke test
python -m ti_export.runner                           # up to pacing.max_exports_per_session (25) jobs
python -m ti_export.merge                            # out/combined.xlsx + "Run Audit" sheet
pytest ti_export/tests
```
Rerun the same command until nothing is pending: `out/manifest.json` records finished jobs (resume), failures and skills LinkedIn didn't recognise. Failure screenshots go to `out/_failures/`, logs to `out/run.log`.

## Layout
| File | Role |
|---|---|
| `models.py` | Pydantic v2 config, `Job`, selector schema + placeholder check |
| `loader.py` | xlsx → skills per Implement family → job matrix |
| `browser.py` | driver attach/launch, download handling |
| `runner.py` | per-job filter → export → rename, retries, pacing, manifest |
| `merge.py` | consolidate exports, add job_family/company/location columns |

Exit codes: `0` ok, `2` placeholders unfilled, `3` login/verification wall, `130` interrupted.
