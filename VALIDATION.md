# Validation record

Validated on 10 October 2026 against the supplied workbook and raw JSON data.

- Workbook: 27,884 player-match rows; 1,233 matches. The input Excel is unchanged (SHA-256 matched).
- Seven automated checks passed: exact optimizer versus exhaustive search; strict earlier-history features; same-day/current/future exclusion; date and doubleheader handling; real report/Excel agreement and 11 player explanations; training-date prefix refit; earliest-match no-history behavior; current winner/score exclusion from win features (some checks share a test).
- Streamlit interaction passed: enter 25-05-2025, receive two match choices, click Generate, render both XIs and all player evidence without exceptions.
- Both Matplotlib charts inspected for labels and layout.
- Runtime: Python 3.12; dependency versions are in requirements-tested.txt. Windows itself was not available for testing; commands and paths use portable Python APIs.

## Held-out metrics

Fixed fitting cutoff: 2024-06-30. Clean training player rows: 20,494. Held-out player rows: 3,568. Match-win test cases: 144.

| Metric | Result |
|---|---:|
| AdaBoost player MAE | 26.3761 |
| Recent-five average player MAE | 26.2089 |
| Win probability Brier score | 0.2589 |
| Constant 50/50 Brier score | 0.2500 |
| Match-win classification accuracy | 40.28% |

The current model does not beat these simple baselines. The implementation supplies the requested outputs but does not establish a predictive advantage. No parameter tuning was performed against the held-out matches.

## Included example

Match 1473504 (25 May 2025): predicted XI expected points 524.31; predicted XI actual points 418; actual optimal subset points 648; overlap 6/11. Two unknown-role players are explicitly excluded from both optimizations. The recorded-player roster is retrospective. This example is not a certified contest lineup.

Match 1426312 demonstrates a historical training-date prefix refit with a complete known-role roster. Match 335982 demonstrates the earliest-match response with no fabricated prediction or probability.

## Expanded dashboard verification

- Eleven automated tests passed, including exact regression metrics with zero/negative targets, winner probability metrics, deduplicated persistent requests, chronological split separation and consistency with the original held-out evaluation.
- Streamlit interaction checks passed for all four pages, both evaluation periods, empty season filters, doubleheaders, report generation, all 11 player explanations, repeated requests and empty request subgroups.
- The Model validation page rendered 12 Plotly graphs and 10 metric cards; the match example rendered four interactive graphs in addition to the printable charts.
- Two Generate clicks for match 1473504 produced one unique evaluated request and a request count of two. Test request logs are excluded from the distributed ZIP.
- See DASHBOARD_CHANGES.md for separate validation and held-out results.
