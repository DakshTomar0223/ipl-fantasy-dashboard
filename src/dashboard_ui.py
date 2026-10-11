"""Four-page Streamlit dashboard for evaluation, replay and player evidence."""
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .ui_backend import FantasyService, ROOT, parse_date
from .dashboard_analytics import DashboardAnalytics, RequestLog, point_metrics, winner_metrics, selection_summary

COLORS = ["#0f766e", "#f59e0b", "#4969a8", "#c86460"]


@st.cache_resource
def get_service(workbook_stamp, overrides_stamp):
    return FantasyService()


@st.cache_resource
def get_diagnostics(fingerprint, cutoff, _service):
    return DashboardAnalytics(_service).build()


def fmt(value, decimals=2, suffix=""):
    return "—" if value is None or pd.isna(value) else f"{value:,.{decimals}f}{suffix}"


def chart(fig, key):
    fig.update_layout(template="plotly_white", font=dict(family="Arial", color="#18354a"),
                      colorway=COLORS, margin=dict(l=20, r=20, t=50, b=35),
                      legend=dict(orientation="h", y=-.22, x=0), height=390)
    st.plotly_chart(fig, width="stretch", key=key)


def score_cards(players, tolerance, prefix=""):
    model = point_metrics(players, tolerance=tolerance)
    baseline = point_metrics(players, "recent_average", tolerance) if "recent_average" in players else {}
    a, b, c, d = st.columns(4)
    change = model.get("mae", np.nan)-baseline.get("mae", np.nan)
    a.metric(prefix+"MAE · points", fmt(model.get("mae")),
             delta=f"{change:+.2f} vs recent-5" if np.isfinite(change) else None, delta_color="inverse")
    b.metric(prefix+"RMSE · points", fmt(model.get("rmse")))
    c.metric(prefix+"R²", fmt(model.get("r2"), 3))
    d.metric(f"Within ±{tolerance} points", fmt(model.get("within_tolerance_percent"), 1, "%"))
    st.caption(f"{model.get('rows', 0):,} scored player appearances. Lower MAE/RMSE is better. R² can be negative. The tolerance rate is not winner accuracy.")
    return model, baseline


def points_plots(players, prefix, tolerance=20):
    valid = players.dropna(subset=["predicted_points", "total_fantasy_points"]).copy()
    if valid.empty:
        st.info("No scored player predictions for this selection.")
        return
    valid["error"] = valid.predicted_points-valid.total_fantasy_points
    valid["absolute_error"] = valid.error.abs()
    a, b = st.columns(2)
    with a:
        hover = [c for c in ["player", "match_id", "date", "role"] if c in valid]
        fig = px.scatter(valid, x="total_fantasy_points", y="predicted_points", color="role" if "role" in valid else None,
                         hover_data=hover, opacity=.5, title="Predicted vs actual player points",
                         labels={"total_fantasy_points": "Actual Excel points", "predicted_points": "Predicted points"})
        low, high = valid[["total_fantasy_points", "predicted_points"]].min().min(), valid[["total_fantasy_points", "predicted_points"]].max().max()
        fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Perfect prediction", line=dict(color="#45556a", dash="dash")))
        chart(fig, prefix+"scatter")
    with b:
        fig = px.histogram(valid, x="error", nbins=45, title="How far predictions missed", labels={"error": "Predicted − actual points"}, color_discrete_sequence=COLORS)
        fig.add_vline(x=0, line_dash="dash", line_color="#45556a")
        fig.add_vrect(x0=-tolerance, x1=tolerance, fillcolor="#0f766e", opacity=.08, line_width=0)
        chart(fig, prefix+"errors")
    model = point_metrics(valid); baseline = point_metrics(valid, "recent_average") if "recent_average" in valid else {}
    a, b = st.columns(2)
    with a:
        if baseline:
            comparison = pd.DataFrame([{ "Model": label, "Metric": metric.upper(), "Error": results[metric]}
                for label, results in [("AdaBoost", model), ("Recent-five average", baseline)] for metric in ["mae", "rmse"]])
            chart(px.bar(comparison, x="Metric", y="Error", color="Model", barmode="group", title="Does AdaBoost beat recent form?", color_discrete_sequence=COLORS), prefix+"baseline")
    with b:
        if "role" in valid:
            role = valid.groupby("role", as_index=False).agg(MAE=("absolute_error", "mean"), Rows=("absolute_error", "size"))
            chart(px.bar(role, x="role", y="MAE", text="Rows", title="Prediction error by role · labels = row counts", color_discrete_sequence=COLORS), prefix+"roles")
    if "date" in valid:
        valid["year"] = pd.to_datetime(valid.date).dt.year
        annual = []
        for year, group in valid.groupby("year"):
            for label, column in [("AdaBoost", "predicted_points"), ("Recent-five average", "recent_average")]:
                if column in group:
                    metrics = point_metrics(group, column)
                    if metrics:
                        annual.append({"Year": str(year), "Model": label, "MAE": metrics["mae"], "Rows": metrics["rows"]})
        if annual:
            chart(px.line(pd.DataFrame(annual), x="Year", y="MAE", color="Model", markers=True, hover_data=["Rows"], title="Error across seasons", color_discrete_sequence=COLORS), prefix+"annual")
    with st.expander("Inspect the largest misses"):
        cols = [c for c in ["match_id", "date", "player", "role", "predicted_points", "total_fantasy_points", "error", "data_quality_notes"] if c in valid]
        st.dataframe(valid.nlargest(25, "absolute_error")[cols], hide_index=True, width="stretch")


def wins_panel(wins, prefix):
    metrics = winner_metrics(wins)
    if not metrics:
        st.info("No evaluable winner predictions for this selection.")
        return
    a, b, c = st.columns(3)
    a.metric("Winner accuracy", fmt(metrics["accuracy_percent"], 1, "%"))
    b.metric("Brier score · lower is better", fmt(metrics["brier"], 3), delta=f"{metrics['brier']-.25:+.3f} vs 50/50", delta_color="inverse")
    c.metric("Correct / evaluated matches", f"{metrics['correct']} / {metrics['matches']}")
    st.caption(f"Log loss: {metrics['log_loss']:.3f}. Only matches with a recorded winner count; no-results are excluded. Calibration is fitted on an earlier, separate chronological slice.")
    if metrics["brier"] >= .25:
        st.warning("These probability estimates do not beat the 50/50 Brier-score baseline on this selection.")
    rows = wins.dropna(subset=["target", "probability_a"]).copy()
    rows["correct"] = rows.probability_a.ge(.5).eq(rows.target.astype(bool))
    rows["favoured_probability"] = np.maximum(rows.probability_a, 1-rows.probability_a)
    a, b = st.columns(2)
    with a:
        if rows.favoured_probability.nunique() > 1:
            rows["bin"] = pd.qcut(rows.favoured_probability, q=min(5, rows.favoured_probability.nunique()), duplicates="drop")
        else:
            rows["bin"] = "All predictions"
        calibration = rows.groupby("bin", observed=True, as_index=False).agg(predicted=("favoured_probability", "mean"), observed=("correct", "mean"), Matches=("correct", "size"))
        calibration["predicted"] *= 100; calibration["observed"] *= 100
        fig = px.scatter(calibration, x="predicted", y="observed", size="Matches", hover_data=["Matches"],
                         title="Are the win percentages reliable?", labels={"predicted": "Average predicted chance for favoured team (%)", "observed": "How often the favoured team won (%)"})
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect calibration", line=dict(dash="dash", color="#718096")))
        fig.update_xaxes(range=[0, 100]); fig.update_yaxes(range=[0, 100])
        chart(fig, prefix+"calibration")
    with b:
        counts = pd.DataFrame({"Result": ["Correct winner", "Incorrect winner"], "Matches": [metrics["correct"], metrics["matches"]-metrics["correct"]]})
        chart(px.bar(counts, x="Result", y="Matches", text="Matches", color="Result", color_discrete_sequence=COLORS, title="Winner prediction outcomes"), prefix+"correctness")
    if "date" in rows:
        rows["date"] = pd.to_datetime(rows.date)
        rows = rows.sort_values(["date", "match_id"])
        rows["Rolling accuracy (%)"] = rows.correct.rolling(20, min_periods=1).mean()*100
        rows["Window matches"] = np.minimum(np.arange(1, len(rows)+1), 20)
        chart(px.line(rows, x="date", y="Rolling accuracy (%)", hover_data=["match_id", "Window matches"], title="Winner accuracy over the latest 20 evaluated matches", color_discrete_sequence=COLORS), prefix+"rolling")
    st.dataframe(rows.drop(columns="bin", errors="ignore"), hide_index=True, width="stretch")


def xi_panel(table, prefix):
    if table.empty:
        st.info("No XI comparisons available.")
        return
    summary = selection_summary(table)
    scored = table[table.selection_status.eq("ok")].copy()
    a, b, c = st.columns(3)
    a.metric("Mean overlap with actual best XI", fmt(summary.get("mean_overlap"), 1, " / 11"))
    b.metric("Mean actual-point gap", fmt(summary.get("mean_regret"), 1))
    c.metric("Mean available points captured", fmt(summary.get("mean_capture_percent"), 1, "%"))
    st.caption(f"{summary['scored_matches']} scored matches; {summary['unavailable_matches']} unavailable; {summary.get('subset_matches', 0)} use a limited known-role pool. Each match has equal weight. Tied optimal XIs can change overlap without changing points.")
    if scored.empty:
        st.dataframe(table, hide_index=True)
        return
    a, b = st.columns(2)
    with a:
        counts = scored.groupby("overlap", as_index=False).size()
        chart(px.bar(counts, x="overlap", y="size", title="Players shared with the actual best XI", labels={"overlap": "Shared players out of 11", "size": "Matches"}, color_discrete_sequence=COLORS), prefix+"overlap")
    with b:
        chart(px.scatter(scored, x="date", y="regret", color="pool_scope", hover_data=["match_id", "overlap"], title="Actual points missed by match", labels={"regret": "Best actual XI − selected XI points"}, color_discrete_sequence=COLORS), prefix+"regret")
    comparison = scored[["predicted_xi_actual", "baseline_xi_actual", "oracle_points"]].mean().rename({"predicted_xi_actual": "AdaBoost XI", "baseline_xi_actual": "Recent-five XI", "oracle_points": "Actual best XI"}).rename_axis("Selection").reset_index(name="Mean actual points")
    chart(px.bar(comparison, x="Selection", y="Mean actual points", color="Selection", color_discrete_sequence=COLORS, title="Which selection earned more actual points?"), prefix+"actualscores")
    st.dataframe(table, hide_index=True, width="stretch")


def validation_page(service):
    st.header("Model validation")
    st.write("Measure prediction error, winner accuracy and the quality of the selected XI against recorded outcomes.")
    with st.spinner("Preparing chronological validation, held-out predictions and XI comparisons..."):
        diagnostics = get_diagnostics(service.fingerprint, str(service.cutoff), service)
    left, middle, right = st.columns([2, 2, 1])
    split = left.selectbox("Evaluation period", ["Held-out test", "Validation"])
    source = diagnostics["players"][diagnostics["players"].split.eq(split)].copy()
    years = sorted(pd.to_datetime(source.date).dt.year.unique().tolist())
    selected_years = middle.multiselect("Seasons", years, default=years)
    tolerance = right.slider("Point tolerance (±)", 5, 50, 20, 5)
    unflagged = st.checkbox("Use only unflagged player rows and XI comparisons", value=False)
    meta = next(m for m in diagnostics["metadata"] if m["split"] == split)
    st.info(f"Fitting uses results before {meta['training_end_exclusive']}. Evaluation: {meta['evaluation_start']} to {meta['evaluation_end']}. No evaluated match is used to fit this model.")
    players = source[pd.to_datetime(source.date).dt.year.isin(selected_years)].copy()
    wins = diagnostics["wins"][diagnostics["wins"].split.eq(split)].copy()
    wins = wins[pd.to_datetime(wins.date).dt.year.isin(selected_years)]
    selections = diagnostics["selections"][diagnostics["selections"].split.eq(split)].copy()
    selections = selections[pd.to_datetime(selections.date).dt.year.isin(selected_years)]
    if unflagged:
        players = players[~players.flagged]
        selections = selections[~selections.quality_flagged]
    st.caption("The quality filter affects player scores and XI comparisons; winner metrics use recorded match results. Historical rosters and candidate roles remain data limitations.")
    model, baseline = score_cards(players, tolerance)
    if model and baseline and model["mae"] >= baseline["mae"]:
        st.warning("AdaBoost's point error is higher than the recent-five average on this selection. The dashboard shows this rather than implying improved accuracy.")
    ptab, wtab, xtab, ftab = st.tabs(["Fantasy points", "Match winners", "XI quality", "Feature importance"])
    with ptab:
        points_plots(players, "validation_", tolerance)
    with wtab:
        wins_panel(wins, "validation_")
    with xtab:
        xi_panel(selections, "validation_")
    with ftab:
        importance = diagnostics["importances"][diagnostics["importances"].split.eq(split)].nlargest(15, "importance").sort_values("importance")
        if len(importance):
            chart(px.bar(importance, x="importance", y="feature", orientation="h", title="Top training feature importances", color_discrete_sequence=COLORS), "validation_importance")
        st.caption("Tree-based training importance describes model usage, not cause or guaranteed predictive value. Missing-history indicators can also matter. Season/quality display filters do not refit this model.")
        st.dataframe(pd.DataFrame(diagnostics["metadata"]), hide_index=True, width="stretch")
    st.download_button("Download displayed player predictions", players.to_csv(index=False), "validation_player_predictions.csv", "text/csv")


def show_team(players):
    if not players:
        st.info("A valid XI is unavailable. See data and selection notes.")
        return
    frame = pd.DataFrame(players)
    st.dataframe(frame[["player", "team", "role", "predicted_points", "total_fantasy_points"]], hide_index=True, width="stretch")


def match_page(service):
    st.header("Match explorer")
    st.write("Generate a historical prediction, compare it with the actual result, and inspect every selected player.")
    date_text = st.text_input("Match date (DD-MM-YYYY)", service.data.date.max().strftime("%d-%m-%Y"))
    nearest = st.checkbox("Use nearest available date", value=True)
    try:
        requested = parse_date(date_text).strftime("%Y-%m-%d")
        choices = service.list_matches(date_text, nearest)
    except ValueError:
        st.warning("Enter a valid date such as 25-05-2025.")
        return
    if not choices:
        st.info("No match on that exact date. Enable the nearest-date option.")
        return
    labels = {m["match_id"]: f"{m['date']} · {' vs '.join(m['teams'])} · {m['match_id']}" for m in choices}
    selected = st.selectbox("Match (doubleheaders are separate)", list(labels), format_func=labels.get)
    if st.button("Generate match report", type="primary"):
        with st.spinner("Predicting, optimizing and recording this requested replay..."):
            st.session_state["report"] = service.get_match_report(match_id=selected, date=date_text, nearest=nearest, record_request=True)
    report = st.session_state.get("report")
    if not report or report["match"]["match_id"] != selected or report["match"]["requested_date"] != requested:
        st.info("Generate the report to see both XIs, graphs and this match's prediction errors.")
        return
    match = report["match"]
    st.subheader(" vs ".join(match["teams"]))
    st.caption(f"{match['date']} · {match['venue']} · Recorded winner: {match['actual_winner'] or match['actual_result']} · {report['model']['query_mode']}")
    if match["nearest_date_distance_days"]:
        st.info(f"Nearest match date: {match['date']}, {match['nearest_date_distance_days']} days from your request.")
    if report["status"] != "ok":
        st.warning("This report has a restricted player pool or insufficient history. See the notes for excluded players.")
    comparison = report["comparison"] or {}
    a, b, c, d = st.columns(4)
    a.metric("Predicted XI · actual points", fmt(comparison.get("predicted_team_actual_points"), 0))
    b.metric("Actual best XI · points", fmt(comparison.get("actual_best_team_points"), 0))
    c.metric("Shared players", fmt(comparison.get("overlap"), 0, " / 11"))
    d.metric("Actual-point gap", fmt(comparison.get("selection_regret"), 0))
    teams, evidence, diagnostics, files = st.tabs(["Teams & result", "Player evidence", "Match diagnostics", "Report files & notes"])
    with teams:
        win = report["team_win_probabilities"]
        if win["probabilities"]:
            chances = pd.DataFrame(win["probabilities"])
            chart(px.bar(chances, x="win_percent", y="team", orientation="h", text="win_percent", title="Estimated win chances (%)", color="team", color_discrete_sequence=COLORS), "match_wins")
            favourite = max(win["probabilities"], key=lambda x: x["probability"])["team"]
            outcome = "Correct winner prediction" if favourite == match["actual_winner"] else "Incorrect winner prediction"
            st.write(f"Predicted winner: **{favourite}** · {outcome if match['actual_winner'] else 'No recorded winner to evaluate'}")
            st.caption(win["interpretation"])
        else:
            st.info(win.get("reason", "Insufficient earlier results for win estimates."))
        left, right = st.columns(2)
        with left:
            st.subheader("Predicted XI")
            show_team(report["predicted_team"])
        with right:
            st.subheader("Actual best XI")
            show_team(report["actual_team"])
        st.caption(report["actual_team_definition"])
        with st.expander("Earlier team form and head-to-head evidence"):
            st.json(win.get("evidence", {}))
    with evidence:
        players = report["player_explanations"]
        if players:
            columns = ["player", "role", "predicted_points", "last_fantasy_points", "last_5_average", "last_10_average", "previous_opponent_points", "specialty_from_previous_matches"]
            st.dataframe(pd.DataFrame(players)[columns], hide_index=True, width="stretch")
            player_names = {p["player_id"]: p["player"] for p in players}
            chosen = st.selectbox("Inspect a selected player", list(player_names), format_func=player_names.get)
            player = next(p for p in players if p["player_id"] == chosen)
            a, b, c = st.columns(3)
            a.metric("Previous appearance", fmt(player["last_fantasy_points"], 1))
            b.metric("Previous five average", fmt(player["last_5_average"], 1))
            c.metric("Previous ten average", fmt(player["last_10_average"], 1))
            history = pd.DataFrame(player["history"])
            if len(history):
                fig = px.line(history, x="date", y="total_fantasy_points", markers=True, title=f"{player['player']} · earlier appearances", color_discrete_sequence=COLORS)
                fig.add_hline(y=player["predicted_points"], line_dash="dash", annotation_text="Current prediction")
                chart(fig, "match_player_history")
            if player["feature_sensitivity"]:
                sensitivity = pd.DataFrame(player["feature_sensitivity"])
                chart(px.bar(sensitivity, x="prediction_difference", y="feature", orientation="h", title="Change from replacing a feature with its training median", color_discrete_sequence=COLORS), "match_player_sensitivity")
            st.caption(player["explanation"])
            with st.expander("Printable graphs for all 11 selected players"):
                for path in report["charts"].values():
                    st.image(str(ROOT / path), width="stretch")
        else:
            st.info("No predicted XI is available, so there are no selected-player explanations.")
    with diagnostics:
        candidates = pd.DataFrame(report["player_predictions"])
        score_cards(candidates, 20, "Match ")
        st.caption("Player errors cover all recorded roster players available in Excel, including unknown-role players excluded from XI optimization.")
        if len(candidates):
            comparison_rows = candidates.melt(id_vars=["player", "role"], value_vars=["predicted_points", "total_fantasy_points", "recent_average"], var_name="Series", value_name="Points")
            chart(px.bar(comparison_rows, x="player", y="Points", color="Series", barmode="group", title="Every candidate: predicted, actual and recent-five points", color_discrete_sequence=COLORS), "match_candidate_points")
            st.dataframe(candidates, hide_index=True, width="stretch")
    with files:
        for note in report["warnings"] + report["selection_errors"]:
            st.write(note)
        st.json({"model": report["model"], "candidate_pool": report["candidate_pool"]})
        st.download_button("Download report JSON", json.dumps(report, indent=2), f"match_{selected}_report.json", "application/json")
        st.download_button("Download all candidate predictions", pd.DataFrame(report["player_predictions"]).to_csv(index=False), f"match_{selected}_players.csv", "text/csv")
        st.caption(f"Saved to outputs/matches/{selected}/. This request is also recorded under Requested matches.")


def requests_page(service):
    st.header("Requested match performance")
    st.write("Track the matches you actually request through this UI and compare their predictions with the recorded outcomes.")
    st.info("These are requested historical replays, not new live forecasts or an independent test set. Repeated clicks increase the request count but count once in performance metrics.")
    data = RequestLog(service.root).read(service)
    requests = data["requests"]
    if requests.empty:
        st.info("No requests recorded for this input/model version. Open Match explorer and generate a report first.")
        return
    scope = st.selectbox("Requested match group", ["All requested matches", "After cutoff only", "Training-period replays only"])
    if scope == "After cutoff only":
        requests = requests[requests.query_mode.eq("held_out_after_cutoff")]
    elif scope == "Training-period replays only":
        requests = requests[requests.query_mode.eq("historical_prefix_refit")]
    keys = set(requests.request_key)
    players = data["players"]
    players = players[players.request_key.isin(keys)] if len(players) else players
    wins = data["wins"]
    wins = wins[wins.request_key.isin(keys)] if len(wins) else wins
    a, b, c, d = st.columns(4)
    a.metric("Unique requested matches", len(requests))
    b.metric("Total generate requests", int(requests.request_count.sum()))
    c.metric("Mean XI overlap", fmt(requests.overlap.mean(), 1, " / 11"))
    d.metric("Mean actual-point gap", fmt(requests.regret.mean(), 1))
    score_cards(players, 20)
    a, b = st.tabs(["Point predictions & request history", "Winner predictions"])
    with a:
        if len(requests):
            chart(px.line(requests.sort_values("first_requested_at"), x="first_requested_at", y="player_mae", markers=True, hover_data=["match_id", "date", "pool_scope"], title="Error on each unique requested match", color_discrete_sequence=COLORS), "request_trend")
        points_plots(players, "requests_")
        st.dataframe(requests, hide_index=True, width="stretch")
    with b:
        wins_panel(wins, "requests_")
    st.download_button("Download request performance CSV", requests.to_csv(index=False), "requested_match_performance.csv", "text/csv")
    st.caption("The local SQLite log persists across app restarts and filters to the current input fingerprint, dependency version and training cutoff. It stores match reports, not personal information.")


def data_page(service):
    st.header("Data & model")
    a, b, c, d = st.columns(4)
    a.metric("Matches", f"{len(service.match_rows):,}")
    b.metric("Player appearances", f"{len(service.data):,}")
    c.metric("Unknown-role rows", f"{(~service.data.role.isin(['BAT','BOWL','AR','WK'])).sum():,}")
    d.metric("Rows with quality flags", f"{service.data.data_quality_notes.ne('').sum():,}")
    st.write("**Algorithm:** AdaBoost with depth-4 decision trees, 100 estimators and learning rate 0.05. A separately calibrated AdaBoost classifier estimates match winners.")
    st.write(f"**Default fitting cutoff:** {service.cutoff:%d-%m-%Y}. Validation uses the cutoff year's earlier season; held-out testing starts after the cutoff. All rolling features exclude current, same-day and later results.")
    st.write("**Selection:** 11 players; 1–8 per role; at least one from each team. Actual XIs maximize your Excel's custom points under those same constraints. Credits and captain multipliers are not available.")
    st.warning("This is a retrospective participant-known evaluation. Workbook roles are candidate profile roles, not historically verified contest roles; quality flags and unknown roles can limit comparisons.")
    counts = service.data.assign(year=service.data.date.dt.year).groupby("year", as_index=False).agg(Matches=("match_id", "nunique"), Player_rows=("player_id", "size"))
    chart(px.bar(counts, x="year", y="Matches", hover_data=["Player_rows"], title="Dataset coverage by season", color_discrete_sequence=COLORS), "data_coverage")
    st.markdown("**Metric guide**\n\n- **MAE:** average absolute miss in fantasy points.\n- **RMSE:** gives larger misses more weight.\n- **R²:** fit relative to a constant observed-mean reference; negative is possible.\n- **Within ±20:** proportion of predictions within 20 fantasy points.\n- **Winner accuracy:** correct predicted winner / matches with a recorded winner.\n- **Brier score:** squared probability error; 50/50 produces 0.25.\n- **XI gap:** actual best-XI points minus selected-XI actual points.\n- **Points captured:** selected-XI actual points / actual-best-XI points, when the latter is positive.")


def main():
    st.set_page_config(page_title="IPL Model Observatory", page_icon="🏏", layout="wide")
    st.markdown("""<style>
    [data-testid="stMetric"]{background:#fff;border:1px solid #dce5ed;border-radius:12px;padding:16px;}
    [data-testid="stMetricLabel"]{color:#526575;}
    .model-banner{background:linear-gradient(105deg,#142e46,#126d68);border-radius:16px;padding:26px 32px;color:#fff;margin-bottom:24px;}
    .model-banner h1{font-size:2rem;margin:0;color:#fff;}.model-banner p{margin:8px 0 0;color:#dbeeee;}
    </style><div class="model-banner"><h1>IPL Model Observatory</h1><p>Prediction · Evidence · Validation · Requested match performance</p></div>""", unsafe_allow_html=True)
    workbook = ROOT / "data/input/IPL_Fantasy_Dataset.xlsx"
    overrides = ROOT / "data/input/role_overrides.csv"
    with st.spinner("Loading player history..."):
        service = get_service(workbook.stat().st_mtime_ns, overrides.stat().st_mtime_ns if overrides.exists() else 0)
    st.sidebar.title("IPL Fantasy Lab")
    page = st.sidebar.radio("Workspace", ["Model validation", "Match explorer", "Requested matches", "Data & model"])
    st.sidebar.caption(f"AdaBoost · {len(service.match_rows):,} matches\n\nFitting cutoff: {service.cutoff:%d-%m-%Y}")
    st.sidebar.info("Model metrics and request metrics have separate pages. A requested historical match is not a live accuracy measurement.")
    {"Model validation": validation_page, "Match explorer": match_page,
     "Requested matches": requests_page, "Data & model": data_page}[page](service)
