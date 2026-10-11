# Model dashboard update

The previous version focused on one-match reports and exposed a small JSON metrics section. This version adds a dedicated evaluation dashboard and persistent request-performance page, while retaining the same AdaBoost algorithms and settings.

## Current results on all evaluation rows

| Measure | Validation | Held-out test |
|---|---:|---:|
| Evaluated player rows | 1,715 | 3,568 |
| Player MAE | 25.96 points | 26.38 points |
| Player RMSE | 31.64 points | 31.92 points |
| Within ±20 points | 42.74% | 40.72% |
| Winner accuracy | 52.11% (37/71) | 40.28% (58/144) |
| Win probability Brier score | 0.2495 | 0.2589 |
| Mean overlap with actual best XI | 5.94/11 | 6.51/11 |
| Mean actual points missed by selected XI | 207.35 | 178.24 |

The constant 50/50 probability baseline has Brier score 0.25. These results do not establish a useful held-out predictive advantage. All figures are calculated from the data; changing the display filters changes the evaluated subset, not the fitted models.

XI statistics cover 71 validation and 148 held-out matches. Respectively 47 and 140 have an incomplete known-role pool, so their actual best XIs are optima only over that smaller pool. Winner metrics exclude unresolved/no-result matches, explaining the different denominators. Historical participant and role limitations are documented in README.md.

## Files added or changed

- `src/dashboard_ui.py`: four-page Streamlit UI, scorecards and interactive Plotly diagnostics.
- `src/dashboard_analytics.py`: separate validation/test evaluation, optimal-XI comparisons, metric calculations and persistent deduplicated request log.
- `app_ipl.py`: dashboard entry point.
- `src/ui_backend.py`: complete candidate-prediction payload, full-precision win probabilities and optional request recording.
- `src/optimizerr.py`: serial CBC solving avoids the worker-mode stalls observed during batch evaluation; constraints and objective are unchanged.
- `.streamlit/config.toml`: dashboard theme.
- `tests/test_dashboard.py`: metric, chronology and persistent-request checks.
- `requirements.txt`: adds Plotly. `requirements-tested.txt` records tested versions.

The input Excel and raw match data remain unchanged. The prior single-page report UI is preserved as `original_code/app_previous_report_ui.py.txt`.
