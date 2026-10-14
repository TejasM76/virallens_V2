# ti_export: LinkedIn Talent Insights export automation (Python + Selenium)

One xlsx per **Job Family × Company × Location** for the `Implement` function (4 families × 4 companies × 13 locations = 208). Skills (Skill 1–8 of rows 10–13) are hard-coded in `config.yaml`.

Flow per session: LinkedIn → Business → Talent Insights (once), then per job: reload the clean Talent Insights page → Location (all aliases) → Skills → Industry → Company → Apply → Export (name `company_function_jobfamily_location`, Select all, xlsx, Save) → rename into `out/<family>/`.

> LinkedIn's User Agreement restricts automated access. Use only your own licensed account, keep the pacing defaults, and note the runner stops (never bypasses) on any login/verification page.

## Setup (company laptop)
```bash
pip install -r ti_export/requirements.txt
```
1. Start Chrome with a **separate profile** (Chrome 136+ ignores the debug port on the default profile), log in to LinkedIn yourself, leave it open:
   ```
   "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir=C:\ti-profile
   ```
2. `selectors.yaml`: replace every `PASTE_XPATH_HERE` (nav, filters, results, export dialog). Option XPaths use `{value}`, e.g. `//li[@role='option'][normalize-space(.)={value}]`. Fields set to `""` are skipped.
3. `config.yaml`: check `industry`, the company names (EY) and each location's `aliases` against what the filters actually offer.

## Finding / verifying XPaths (read-only, attaches to your logged-in browser)
```bash
python -m ti_export.probe dump    # open the page/dialog you want, run, then see out/probe.txt (elements + suggested XPaths)
python -m ti_export.probe check   # after filling selectors.yaml: does each XPath match a visible element on the current page?
```
Open each part of the UI by hand first (filter panel, a suggestion list, the export dialog) and re-run `dump`; the probe never clicks or types.

## Run (from the repo root)
```bash
python -m ti_export.runner --dry-run                                   # job matrix, names, aliases, skills
python -m ti_export.runner --limit 2 --company TCS --location Pune --family "ERP Implementation"   # smoke test
python -m ti_export.runner                                             # up to 25 jobs per session
python -m ti_export.merge                                              # out/combined.xlsx + "Run Audit" sheet
pytest ti_export/tests
```
Rerun until nothing is pending: `out/manifest.json` records finished jobs (resume), failures, and any skills / location aliases LinkedIn didn't recognise. Failure screenshots: `out/_failures/`; log: `out/run.log`.

Exit codes: `0` ok, `2` placeholders unfilled, `3` login/verification wall, `130` interrupted.
