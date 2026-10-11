# IPL Fantasy Match Lab

A working Streamlit UI and reusable Python backend for the supplied `IPL_Fantasy_Dataset.xlsx` and your friend's IPL match JSON files. The predictor retains **AdaBoost with depth-4 decision trees, 100 estimators, learning rate 0.05, random state 42**. An AdaBoost classifier with chronological sigmoid calibration supplies experimental match-win estimates.

## Run on Windows

Extract this updated project into its own folder, then open PowerShell there:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app_ipl.py
```

The app opens on **Model validation**. Use the sidebar to open **Match explorer**, enter a date as `DD-MM-YYYY`, select the match if that day has two, and click **Generate match report**. The nearest date is used when requested; an equal-distance tie chooses the earlier date. The page shows which date was used.

The first run builds features from the Excel and can take a few minutes. Later runs use caches. Training-match queries may require a separate model fit for that historical date. Raw JSON metadata is read with four workers; AdaBoost's successive boosting stages are sequential.

## Expanded model dashboard

The sidebar has four pages:

1. **Model validation:** chronological validation and held-out test scorecards, season and quality filters, adjustable point-error tolerance, and four analysis tabs: Fantasy points, Match winners, XI quality, Feature importance.
2. **Match explorer:** predicted and actual XIs, the result of the winner prediction, per-match prediction errors, individual-player inspection, interactive graphs, the original printable graphs, and report downloads.
3. **Requested matches:** persistent performance of the matches you request through the UI, with separate filters for after-cutoff matches and training-period replays. Repeated requests do not inflate the sample size.
4. **Data & model:** dataset coverage, missing-role/quality counts, model settings and plain-language metric definitions.

Interactive graphs include predicted-versus-actual points, error distribution, AdaBoost versus recent-five baseline, error by role and season, win calibration, correct/incorrect winners, rolling winner accuracy, XI overlap distribution, actual points missed per match, XI points versus a recent-five selection, feature importance, and request-by-request error. Hover for values; use the Plotly toolbar to zoom or download a graph.

**The evaluation periods are distinct.** With the default cutoff, validation fits on matches completed before 1 January 2024 and evaluates 2024 matches through 30 June. The final fixed model then fits through 30 June 2024 and is evaluated on later matches. Its win classifier has its own earlier chronological calibration slice. No parameter search was performed using held-out results.

**Accuracy has a specific meaning.** Winner accuracy is the fraction of correct match-winner predictions. Player fantasy points use MAE, RMSE and R². The displayed within-±20-points rate is a tolerance hit rate, not classification accuracy. Actual-best-XI overlap can vary among equally scoring optimal lineups; the point gap is the more direct selection-quality measure.

**Requested performance is retrospective.** Outcomes already exist in the supplied workbook. This page is not a prospective live-forecast benchmark, and the requested subset is not an independent test set. A future live workflow would need predictions saved before a match and results ingested afterward. Each unique input/model/cutoff/match combination counts once; subsequent clicks update its request counter. The local log is `outputs/request_history.sqlite3`. A fresh download starts with an empty log; demo/test requests are not shipped.

Cached diagnostic tables and a metrics JSON are saved under `outputs/dashboard/<version>/`. For programmatic access:

```python
from src.dashboard_analytics import DashboardAnalytics, RequestLog

diagnostics = DashboardAnalytics(service).build()
report = service.get_match_report(match_id="1473504", record_request=True)
requested = RequestLog(service.root).read(service)
```

`record_request` defaults to `False` for backend callers so evaluation scripts do not pollute user-request metrics. The UI explicitly enables it.

Python 3.11–3.13 is the intended environment. Dependencies are installed through the same Python interpreter used to launch the app.

`requirements-tested.txt` records the exact dependency versions used for the included integration checks. PuLP is kept below version 4 because the original project's solver API changed in version 4.

## Your four outputs

| Output | Contents |
|---|---|
| Predicted XI | Eleven players maximizing the sum of AdaBoost-predicted fantasy points under the original ZIP's roster constraints |
| Actual best XI | Eleven players maximizing the sum of the Excel's actual fantasy points, under the same constraints and candidate pool |
| Player evidence | Last appearance points, previous-five and previous-ten averages, prior opponent performance, historical batting/bowling/fielding specialty, five-match history, and model feature sensitivity for each selected player |
| Win estimates | Percentages summing to 100 for the two real teams, plus earlier head-to-head records, recent results, team fantasy form, and overall records |

The page also shows actual points earned by the predicted XI, actual best-XI points, selection regret (the gap), and player overlap. Matplotlib graphs show the previous-five trend of all 11 selected players and their historical domain contributions.

## Designated output folder

Each generated match is saved under `outputs/matches/<match_id>/`:

- `report.json`: complete UI payload; absent values are JSON `null`.
- `predicted_team.csv`: chosen XI and predicted/actual points.
- `actual_team.csv`: optimal XI based on recorded Excel points.
- `all_player_predictions.csv`: all workbook players for this match, including any players excluded from selection. Read `candidate_pool` in the report for exclusions.
- `player_form.png`: previous-five trend panels for the selected 11.
- `player_domains.png`: previous-five batting, bowling and fielding averages.

`outputs/evaluation.json` holds held-out evaluation results. Example reports are included in the ZIP.

## Call it from a different UI

```python
from src.ui_backend import FantasyService

service = FantasyService()  # Keep one instance for reuse.
choices = service.list_matches(date="25-05-2025", nearest=True)
# Display all choices; don't silently choose one on a doubleheader day.
report = service.get_match_report(
    match_id=choices[0]["match_id"], date="25-05-2025", charts=True
)
```

The primary UI fields are:

```python
report["predicted_team"]
report["actual_team"]
report["player_explanations"]
report["team_win_probabilities"]["probabilities"]
report["comparison"]
report["charts"]
report["warnings"]
```

`get_match_report(date="25-05-2025")` returns `{"status": "choose_match", "matches": [...]}` when multiple matches exist; use their IDs to request a report. `status="partial"` means a pool limitation or unavailable XI. The earliest matches can have an actual XI but no prediction because there is no earlier training history.

No HTTP server is required for the included Streamlit app. A separate frontend can expose this same function through its own API and serve the returned chart paths as static files. Files are regenerated when that match is requested, not precomputed for every possible date.

## Command-line options

```powershell
python run_outputs.py --prepare
python run_outputs.py --date 25-05-2025
python run_outputs.py --match-id 1473504
python run_outputs.py --match-id 1426312
python run_outputs.py --date 01-01-2025 --exact-date
python run_outputs.py --evaluate
python -m unittest discover -s tests -v
```

The date-only example prints the two matching IDs. Add `--no-charts` to skip PNG generation. Old preprocessing/training entry points now forward to the new backend:

```powershell
python src/preprocess_ipl.py
python src/train_model_ipl.py
```

## How the prediction is built

1. **Read the workbook:** one row per player and match. Preserve its total and domain points; validate IDs, dates, numeric values and component sums.
2. **Build histories:** use only appearances with `end_date < current match date`. Exclude current, later and same-day matches. Player histories follow stable player IDs across team transfers. Recognized franchise renames share opponent/team history; different franchises remain distinct.
3. **Fit the score model:** use historical form, run/wicket averages, opponent/venue history, experience, inactivity and missing-history counts. Actual current-match statistics and the target are never features. Rows with workbook quality flags are excluded from model fitting. Their earlier recorded points can still enter histories, with a flagged-history count supplied to the model.
4. **Select both XIs:** PuLP optimizes once over predicted scores and once over actual Excel scores. No guessed roles or heuristic invalid-team fallback.
5. **Estimate wins:** AdaBoostClassifier uses differences in earlier team results, head-to-head record, fantasy form, candidate form and experience. The first approximately 80% of earlier match dates fits the classifier; the later approximately 20% calibrates its probability output. No queried match result enters that fit or calibration.

The default fixed fitting cutoff is **30 June 2024**, matching the original training setup. For a match after the cutoff, the fixed model uses form available before that match, including earlier post-cutoff results. This is a rolling historical replay, not a forecast of an entire future season from the cutoff date. For a match on/before the cutoff, a model is refitted using only earlier completed matches. For fewer than 100 eligible prior player rows, a historical-mean fallback is labeled; with no prior results it returns no predicted XI. Win probabilities require at least 100 earlier matches with recorded winners and a valid calibration split.

For an alternate chronological split:

```python
service = FantasyService(training_cutoff="2023-06-30")
```

## Interpretation and data limits

- **Actual best XI** means the maximum Excel-point total, not an official published Dream11 lineup. It uses your supplied custom scoring and reproduces existing Excel totals exactly. There are no added economy/strike-rate/participation points or captain/vice-captain multipliers.
- Selection retains the ZIP's rules: 11 players, 1–8 each of BAT/BOWL/AR/WK, and at least one from each real team. Credits, a credit budget and other contest-specific rules were not supplied. This does not certify eligibility for an actual Dream11 contest.
- The player pool is the recorded raw-match roster intersected with Excel rows. Nonroster workbook fielders are omitted. This is a retrospective participant-known comparison; the data does not provide a full squad snapshot known before each match. Do not interpret it as a verified pre-toss forecast.
- Unknown roles are excluded from both optimizations and are listed explicitly. If players are missing/unknown, the actual optimum is labeled as a known-role subset optimum. If a required role is unavailable, selection returns an explanation rather than an invalid XI.
- Workbook roles are largely 2025 profile candidates, not verified historical fantasy roles. Historical role availability remains a limitation. You may copy `data/input/role_overrides.example.csv` to `role_overrides.csv` and fill sourced, date-bounded corrections. This changes roster constraints only, not existing Excel scores.
- Workbook quality flags are retained in the report and team CSVs. Actual-score comparisons involving flagged rows inherit those uncertainties.
- Specialty is the player's strongest average domain in earlier appearances, not a newly invented player role. Feature sensitivity replaces one feature with its training median; it is neither a causal explanation nor an additive breakdown of points.
- Win probabilities are conditional on a recorded winner, including tiebreak winners. Unresolved ties and no-results are not modeled as a third class. Calibration does not guarantee accuracy.
- The current held-out results are shown in the UI and `outputs/evaluation.json`. This is an experimental baseline; compare the score model with the recent-five average and win Brier score with 0.25 (the 50/50 baseline). No predictive advantage is assumed.
- Matches must exist in the workbook. If the chosen date is outside it, nearest-date mode reports an existing historical match, not a new fixture prediction.

## What changed in the original project

- The old containers lacked `target_points`, causing the old trainer to fill targets with zero. The new trainer reads real Excel outcomes.
- Match IDs replace date-only grouping, so doubleheaders stay separate.
- Workbook roles replace the app's repeating invented BAT/BOWL/AR/WK cycle.
- A missing required role now prevents selection instead of silently relaxing the constraint.
- Training, calibration and rolling history use chronological boundaries.
- Original source is preserved in `original_code/*.py.txt`; the previous serialized model and containers are unused.

Model/feature caches are keyed by input content and code version; model caches also include the scikit-learn version. Restart Streamlit after changing raw JSON files. The backend is in `src/ui_backend.py`; the dashboard and diagnostics are in `src/dashboard_ui.py` and `src/dashboard_analytics.py`; `app_ipl.py` is the launcher and `run_outputs.py` is the CLI.

Reference documentation: [AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html), [probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
