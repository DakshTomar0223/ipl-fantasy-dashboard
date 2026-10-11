"""UI entry point using Excel outcomes and the original AdaBoost algorithm."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import hashlib
import json
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from .optimizerr import select_dream11_team, ROLES

ROOT = Path(__file__).resolve().parents[1]
VERSION = "ipl-ui-v2"
TEAM_ALIASES = {"Royal Challengers Bangalore": "Royal Challengers Bengaluru",
                "Delhi Daredevils": "Delhi Capitals", "Kings XI Punjab": "Punjab Kings",
                "Rising Pune Supergiants": "Rising Pune Supergiant"}
COMPONENTS = ["batting_fantasy_points", "bowling_fantasy_points", "fielding_fantasy_points"]
FEATURES = ["avg_pts_last_5", "avg_pts_last_10", "max_pts_last_5", "std_pts_last_5", "last_pts",
            "avg_runs_last_5", "avg_wickets_last_5", "avg_batting_last_5", "avg_bowling_last_5",
            "avg_fielding_last_5", "avg_opponent_last_5", "last_opponent_pts", "avg_venue_last_5",
            "previous_matches", "history_count_5", "history_count_10", "opponent_history_count",
            "venue_history_count", "days_since_last_match", "flagged_history_count_5"]
WIN_FEATURES = ["recent_win_rate_diff", "overall_win_rate_diff", "head_to_head_rate_diff",
                "team_fantasy_last_5_diff", "candidate_form_diff", "candidate_batting_diff",
                "candidate_bowling_diff", "candidate_fielding_diff", "experience_diff",
                "head_to_head_count", "recent_decisive_count_min"]


def canonical(team):
    return TEAM_ALIASES.get(str(team), str(team))


def save_cache(value, path):
    """Concurrent readers should only see complete cache files."""
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    try:
        joblib.dump(value, name)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def parse_date(value):
    if isinstance(value, str):
        try:
            return pd.Timestamp(datetime.strptime(value, "%d-%m-%Y"))
        except ValueError:
            return pd.Timestamp(datetime.strptime(value, "%Y-%m-%d"))
    return pd.Timestamp(value).normalize()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()[:10]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def read_metadata(path):
    info = json.loads(path.read_text(encoding="utf-8"))["info"]
    registry = info.get("registry", {}).get("people", {})
    roster = {registry[name]: canonical(team) for team, names in info.get("players", {}).items()
              for name in names if name in registry}
    outcome = info.get("outcome", {})
    winner = outcome.get("winner") or outcome.get("eliminator")
    return path.stem, {"start": pd.Timestamp(min(info["dates"])), "end": pd.Timestamp(max(info["dates"])),
        "teams": sorted(canonical(t) for t in info["teams"]),
        "winner": canonical(winner) if winner else None,
        "result": outcome.get("result", "decisive"), "roster": roster}


class FantasyService:
    def __init__(self, workbook=None, raw_dir=None, root=None, training_cutoff="2024-06-30"):
        self.root = Path(root or ROOT)
        self.workbook = Path(workbook or self.root / "data/input/IPL_Fantasy_Dataset.xlsx")
        self.raw_dir = Path(raw_dir or self.root / "data/raw/ipl")
        self.cutoff = parse_date(training_cutoff)
        self.models_dir = self.root / "src/model_artifacts/ui_v1"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        overrides = self.root / "data/input/role_overrides.csv"
        raw_files = sorted(self.raw_dir.glob("*.json"))
        content_hash = hashlib.sha256(self.workbook.read_bytes()).hexdigest()
        # Content digest avoids stale caches when a match winner/roster is edited.
        raw_hash = hashlib.sha256()
        for path in raw_files:
            raw_hash.update(path.name.encode()); raw_hash.update(path.read_bytes())
        self.fingerprint = hashlib.sha256((VERSION + pd.__version__ + content_hash + raw_hash.hexdigest() +
            (overrides.read_text() if overrides.exists() else "")).encode()).hexdigest()[:16]
        cache = self.root / "data/processed" / f"ui_features_{self.fingerprint}.joblib"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            stored = joblib.load(cache)
            self.data, self.metadata, self.win_data = stored["data"], stored["metadata"], stored["win_data"]
        else:
            self.data = self._load_workbook(overrides)
            with ThreadPoolExecutor(max_workers=4) as pool:
                self.metadata = dict(pool.map(read_metadata, raw_files))
            self._build_features()
            self.win_data = self._build_win_data()
            save_cache({"data": self.data, "metadata": self.metadata, "win_data": self.win_data}, cache)
        self.match_rows = {mid: g.copy() for mid, g in self.data.groupby("match_id", sort=False)}
        self.player_rows = {pid: g.sort_values(["end_date", "match_id"]) for pid, g in self.data.groupby("player_id")}
        self._models = {}

    def _load_workbook(self, overrides):
        data = pd.read_excel(self.workbook, sheet_name="Player Match Data", dtype={"match_id": str, "player_id": str})
        required = {"match_id", "player_id", "player", "date", "end_date", "team", "opponent", "venue", "role",
                    "total_fantasy_points", "runs", "wickets", *COMPONENTS}
        if required - set(data.columns):
            raise ValueError(f"Workbook missing columns: {sorted(required - set(data.columns))}")
        for col in ["date", "end_date"]:
            data[col] = pd.to_datetime(data[col], errors="raise").dt.normalize()
        if data[list(required)].drop(columns=["role"]).isna().any().any():
            raise ValueError("Missing required workbook values")
        if data.duplicated(["match_id", "player_id"]).any() or (data.end_date < data.date).any():
            raise ValueError("Duplicate player-match IDs or invalid dates")
        if not np.isfinite(data[["total_fantasy_points", *COMPONENTS]].to_numpy(float)).all():
            raise ValueError("Nonfinite point totals in workbook")
        if not np.allclose(data[COMPONENTS].sum(axis=1), data.total_fantasy_points):
            raise ValueError("Workbook domain components do not sum to total points")
        data["team_key"] = data.team.map(canonical)
        data["opponent_key"] = data.opponent.map(canonical)
        data["role"] = data.role.fillna("UNKNOWN").str.upper().str.strip()
        data["role_source"] = data.get("role_source", pd.Series("unspecified", index=data.index)).fillna("unspecified")
        data["data_quality_notes"] = data.get("data_quality_notes", pd.Series("", index=data.index)).fillna("")
        if overrides.exists():
            for row in pd.read_csv(overrides, dtype=str).fillna("").to_dict("records"):
                if row.get("role") not in ROLES or not row.get("source"):
                    raise ValueError("Role overrides require BAT/BOWL/AR/WK and a source")
                mask = data.player_id.eq(row["player_id"])
                if row.get("valid_from"):
                    mask &= data.date.ge(parse_date(row["valid_from"]))
                if row.get("valid_to"):
                    mask &= data.date.le(parse_date(row["valid_to"]))
                data.loc[mask, "role"] = row["role"]
                data.loc[mask, "role_source"] = row["source"]
        return data.sort_values(["date", "match_id", "player_id"]).reset_index(drop=True)

    def _build_features(self):
        values = []
        for _, group in self.data.groupby("player_id", sort=False):
            completed = group.sort_values(["end_date", "match_id"])
            ends = completed.end_date.to_numpy()
            for idx, row in group.iterrows():
                count = np.searchsorted(ends, row.date.to_datetime64(), side="left")
                past = completed.iloc[:count]
                last5, last10 = past.tail(5), past.tail(10)
                opposition = past[past.opponent_key.eq(row.opponent_key)]
                venue = past[past.venue.eq(row.venue)]
                values.append({"_index": idx,
                    "avg_pts_last_5": last5.total_fantasy_points.mean(), "avg_pts_last_10": last10.total_fantasy_points.mean(),
                    "max_pts_last_5": last5.total_fantasy_points.max(), "std_pts_last_5": last5.total_fantasy_points.std(ddof=0),
                    "last_pts": past.total_fantasy_points.iloc[-1] if count else np.nan,
                    "avg_runs_last_5": last5.runs.mean(), "avg_wickets_last_5": last5.wickets.mean(),
                    "avg_batting_last_5": last5.batting_fantasy_points.mean(), "avg_bowling_last_5": last5.bowling_fantasy_points.mean(),
                    "avg_fielding_last_5": last5.fielding_fantasy_points.mean(),
                    "avg_opponent_last_5": opposition.tail(5).total_fantasy_points.mean(),
                    "last_opponent_pts": opposition.total_fantasy_points.iloc[-1] if len(opposition) else np.nan,
                    "avg_venue_last_5": venue.tail(5).total_fantasy_points.mean(), "previous_matches": count,
                    "history_count_5": len(last5), "history_count_10": len(last10), "opponent_history_count": len(opposition),
                    "venue_history_count": len(venue), "days_since_last_match": (row.date-past.end_date.iloc[-1]).days if count else np.nan,
                    "flagged_history_count_5": int(last5.data_quality_notes.ne("").sum())})
        features = pd.DataFrame(values).set_index("_index")
        for column in FEATURES:
            self.data[column] = features[column].reindex(self.data.index)

    def _build_win_data(self):
        result_history, pair_history = {}, {}
        completed = sorted(self.metadata.items(), key=lambda item: (item[1]["end"], item[0]))
        pointer, records = 0, []
        team_points = {mid: g.groupby("team_key").total_fantasy_points.mean().to_dict() for mid, g in self.data.groupby("match_id")}
        for mid, group in self.data.groupby("match_id", sort=False):
            start = group.date.iloc[0]
            while pointer < len(completed) and completed[pointer][1]["end"] < start:
                old_id, meta = completed[pointer]
                for team in meta["teams"]:
                    result_history.setdefault(team, []).append({"winner": meta["winner"],
                        "points": team_points.get(old_id, {}).get(team, np.nan)})
                if meta["winner"] in meta["teams"]:
                    pair_history.setdefault(tuple(meta["teams"]), []).append(meta["winner"])
                pointer += 1
            teams = sorted(group.team_key.unique())
            if len(teams) != 2:
                continue
            summaries = {}
            for team in teams:
                history = result_history.get(team, [])
                decisive = [r for r in history if r["winner"] is not None]
                recent = decisive[-5:]
                pg = group[group.team_key.eq(team)]
                roster = self.metadata.get(mid, {}).get("roster", {})
                if roster:
                    pg = pg[pg.player_id.isin(roster)]
                summaries[team] = {"recent_decisive_matches": len(recent), "recent_wins": sum(r["winner"] == team for r in recent),
                    "overall_decisive_matches": len(decisive), "overall_wins": sum(r["winner"] == team for r in decisive),
                    "team_fantasy_last_5": pd.Series([r["points"] for r in history[-5:]], dtype=float).mean(),
                    "candidate_form": pg.avg_pts_last_5.mean(), "candidate_batting": pg.avg_batting_last_5.mean(),
                    "candidate_bowling": pg.avg_bowling_last_5.mean(), "candidate_fielding": pg.avg_fielding_last_5.mean(),
                    "experience": pg.previous_matches.mean()}
            a, b = teams; sa, sb = summaries[a], summaries[b]
            pair = pair_history.get(tuple(teams), [])
            vector = {
                "recent_win_rate_diff": (sa["recent_wins"]+1)/(sa["recent_decisive_matches"]+2)-(sb["recent_wins"]+1)/(sb["recent_decisive_matches"]+2),
                "overall_win_rate_diff": (sa["overall_wins"]+1)/(sa["overall_decisive_matches"]+2)-(sb["overall_wins"]+1)/(sb["overall_decisive_matches"]+2),
                "head_to_head_rate_diff": (2*pair.count(a)-len(pair))/(len(pair)+2), "head_to_head_count": len(pair),
                "recent_decisive_count_min": min(sa["recent_decisive_matches"], sb["recent_decisive_matches"])}
            for key in ["team_fantasy_last_5", "candidate_form", "candidate_batting", "candidate_bowling", "candidate_fielding", "experience"]:
                vector[key+"_diff"] = sa[key]-sb[key]
            winner = self.metadata.get(mid, {}).get("winner")
            records.append({"match_id": mid, "date": start, "end_date": group.end_date.max(), "team_a": a, "team_b": b,
                "target": int(winner == a) if winner in teams else np.nan,
                "evidence": {"teams": summaries, "head_to_head_matches": len(pair), "head_to_head_wins": {a: pair.count(a), b: pair.count(b)}}, **vector})
        return pd.DataFrame(records)

    @staticmethod
    def _pipeline(classifier=False):
        tree = DecisionTreeClassifier(max_depth=4, random_state=42) if classifier else DecisionTreeRegressor(max_depth=4, random_state=42)
        algo = AdaBoostClassifier if classifier else AdaBoostRegressor
        return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                         ("model", algo(estimator=tree, n_estimators=100, learning_rate=.05, random_state=42))])

    def _fit_bundle(self, as_of):
        limit = min(parse_date(as_of), self.cutoff+pd.Timedelta(days=1))
        key = limit.strftime("%Y-%m-%d")
        path = self.models_dir / f"models_{self.fingerprint}_{key}_{sklearn.__version__}.joblib"
        if key in self._models:
            return self._models[key]
        if path.exists():
            bundle = joblib.load(path); self._models[key] = bundle
            return bundle
        train = self.data[self.data.end_date.lt(limit)].copy()
        clean = train.data_quality_notes.eq("")
        fitting = train[clean]
        bundle = {"limit_exclusive": key, "training_rows": len(fitting), "excluded_training_rows": int((~clean).sum()),
            "training_match_ids": sorted(fitting.match_id.unique()), "regressor": None,
            "fallback_mean": float(fitting.total_fantasy_points.mean()) if len(fitting) else None,
            "win_model": None, "win_training_matches": 0, "calibration_matches": 0,
            "win_status": "insufficient_history", "sklearn_version": sklearn.__version__}
        if len(fitting) >= 100:
            model = self._pipeline(); model.fit(fitting[FEATURES], fitting.total_fantasy_points)
            bundle["regressor"] = model
        win = self.win_data[self.win_data.end_date.lt(limit) & self.win_data.target.notna()].sort_values(["date", "match_id"])
        if len(win) >= 100 and win.target.nunique() == 2:
            dates = sorted(win.date.unique()); boundary = dates[int(.8*len(dates))]
            base = win[win.end_date.lt(boundary)]; calibration = win[win.date.ge(boundary)]
            if len(base) >= 60 and len(calibration) >= 20 and base.target.nunique() == calibration.target.nunique() == 2:
                clf = self._pipeline(classifier=True); clf.fit(base[WIN_FEATURES], base.target.astype(int))
                calibrated = CalibratedClassifierCV(FrozenEstimator(clf), method="sigmoid")
                calibrated.fit(calibration[WIN_FEATURES], calibration.target.astype(int))
                bundle.update(win_model=calibrated, win_status="chronologically_calibrated_adaboost",
                              win_training_matches=len(base), calibration_matches=len(calibration))
        save_cache(bundle, path); self._models[key] = bundle
        return bundle

    def prepare(self):
        bundle = self._fit_bundle(self.cutoff+pd.Timedelta(days=1))
        return {k: v for k, v in bundle.items() if k not in {"regressor", "win_model", "training_match_ids"}}

    def list_matches(self, date=None, nearest=True):
        catalog = [{"match_id": mid, "date": g.date.iloc[0], "teams": sorted(g.team_key.unique()), "venue": g.venue.iloc[0]}
                   for mid, g in self.data.groupby("match_id", sort=False)]
        if date is not None:
            requested = parse_date(date); selected = requested; dates = {x["date"] for x in catalog}
            if requested not in dates and nearest:
                selected = min(dates, key=lambda d: (abs((d-requested).days), d))
            catalog = [x for x in catalog if x["date"] == selected]
        return json_safe(catalog)

    def get_match_report(self, match_id=None, date=None, nearest=True, charts=True, record_request=False):
        if match_id is None:
            choices = self.list_matches(date, nearest)
            if len(choices) != 1:
                return {"status": "choose_match", "matches": choices}
            match_id = choices[0]["match_id"]
        match_id = str(match_id)
        if match_id not in self.match_rows:
            raise ValueError(f"Unknown match ID {match_id}")
        group = self.match_rows[match_id].copy(); start = group.date.iloc[0]
        bundle = self._fit_bundle(start)
        notes = ["Targets follow the uploaded Excel and custom points formula, not an official Dream11 export.",
            "Recorded participants are used for this retrospective comparison; a full pre-toss squad is not available.",
            "Roles in the workbook are largely 2025 profile candidates, not verified historical fantasy roles.",
            "History uses earlier completed matches, including earlier post-cutoff results as input; fitting stays within the cutoff."]
        if bundle["regressor"] is not None:
            group["predicted_points"] = bundle["regressor"].predict(group[FEATURES]); method = "AdaBoostRegressor"
        elif bundle["fallback_mean"] is not None:
            group["predicted_points"] = bundle["fallback_mean"]; method = "earlier_results_mean_insufficient_training"
            notes.append("Too little history to fit AdaBoost: predictions use the earlier training mean.")
        else:
            group["predicted_points"] = np.nan; method = "unavailable_no_prior_training_results"
            notes.append("No earlier training results exist; no predicted XI is fabricated.")
        meta = self.metadata.get(match_id, {}); roster = meta.get("roster", {})
        if roster:
            pool = group[group.player_id.isin(roster)].copy(); missing_excel = sorted(set(roster)-set(group.player_id))
        else:
            pool, missing_excel = group.copy(), []
            notes.append("Raw roster unavailable: candidate pool is the workbook's player list.")
        pool["team"] = pool.team_key
        unknown = pool[~pool.role.isin(ROLES)]; eligible = pool[pool.role.isin(ROLES)].copy()
        scope = "all_recorded_participants" if unknown.empty and not missing_excel else "known_role_subset"
        if len(unknown):
            notes.append(f"{len(unknown)} unknown-role players excluded. Both optimizations use the same known-role subset.")
        if missing_excel:
            notes.append(f"{len(missing_excel)} roster players missing from Excel: full-pool optimum cannot be claimed.")
        flagged = int(eligible.data_quality_notes.ne("").sum())
        if flagged:
            notes.append(f"{flagged} eligible players have workbook quality flags. Actual totals are reproduced from Excel and remain subject to those flags.")
        notes.append("Selection retains the ZIP's constraints: 11 players, 1–8 per role, at least one from each team. Credits and captain/vice-captain multipliers are not supplied.")
        predicted, actual, errors = None, None, []
        if eligible.predicted_points.notna().all():
            try:
                predicted = select_dream11_team(eligible, "predicted_points")
            except ValueError as e:
                errors.append(str(e))
        try:
            actual = select_dream11_team(eligible, "total_fantasy_points")
        except ValueError as e:
            errors.append(str(e))
        explanations = [self._player_evidence(row, start, bundle) for _, row in (predicted.iterrows() if predicted is not None else [])]
        comparison = None
        if predicted is not None and actual is not None:
            comparison = {"predicted_team_expected_points": float(predicted.predicted_points.sum()),
                "predicted_team_actual_points": float(predicted.total_fantasy_points.sum()),
                "actual_best_team_points": float(actual.total_fantasy_points.sum()),
                "selection_regret": float(actual.total_fantasy_points.sum()-predicted.total_fantasy_points.sum()),
                "overlap": len(set(predicted.player_id)&set(actual.player_id)), "pool_scope": scope}
        cols = ["player_id", "player", "team", "role", "role_source", "predicted_points", "total_fantasy_points", *COMPONENTS, "data_quality_notes"]
        win_row = self.win_data[self.win_data.match_id.eq(match_id)]
        win_result = {"status": "unavailable", "probabilities": [], "reason": "No match features"}
        if len(win_row):
            wr = win_row.iloc[0]
            if bundle["win_model"] is not None:
                p = float(bundle["win_model"].predict_proba(win_row[WIN_FEATURES])[0, 1])
                win_result = {"status": bundle["win_status"], "probabilities": [
                    {"team": wr.team_a, "win_percent": round(p*100, 2), "probability": p},
                    {"team": wr.team_b, "win_percent": round(100-p*100, 2), "probability": 1-p}],
                    "evidence": wr.evidence, "training_matches": bundle["win_training_matches"],
                    "calibration_matches": bundle["calibration_matches"],
                    "interpretation": "Estimate conditional on a recorded winner, including tiebreak winners. Unresolved ties/no-results are not a third class; calibration does not guarantee accuracy."}
            else:
                win_result = {"status": "insufficient_history", "probabilities": [], "evidence": wr.evidence,
                              "reason": "At least 100 earlier decisive matches and a valid calibration split are required."}
        out_dir = self.root / "outputs/matches" / match_id; out_dir.mkdir(parents=True, exist_ok=True)
        report = {"schema_version": VERSION, "status": "ok" if predicted is not None and actual is not None and scope == "all_recorded_participants" else "partial",
            "match": {"match_id": match_id, "date": start, "venue": group.venue.iloc[0], "teams": sorted(group.team_key.unique()),
                      "requested_date": parse_date(date) if date is not None else None,
                      "nearest_date_distance_days": abs((start-parse_date(date)).days) if date is not None else 0,
                      "actual_winner": meta.get("winner"), "actual_result": meta.get("result")},
            "model": {"algorithm": method, "base_tree_max_depth": 4, "estimators": 100, "learning_rate": .05,
                      "training_cutoff": self.cutoff, "training_end_exclusive": bundle["limit_exclusive"], "training_rows": bundle["training_rows"],
                      "excluded_training_rows": bundle["excluded_training_rows"],
                      "query_mode": "historical_prefix_refit" if start <= self.cutoff else "held_out_after_cutoff",
                      "selected_match_in_training": match_id in bundle["training_match_ids"]},
            "candidate_pool": {"scope": scope, "participants": len(pool), "known_roles": len(eligible),
                "quality_flagged_players": flagged,
                "unknown_roles": unknown[["player_id", "player"]].to_dict("records"), "missing_excel_ids": missing_excel,
                "omitted_non_roster_rows": len(group)-len(pool)},
            "predicted_team": predicted[cols].to_dict("records") if predicted is not None else [],
            "actual_team": actual[cols].to_dict("records") if actual is not None else [],
            "actual_team_definition": "Highest Excel-point sum under the same constraints and candidate subset; no captain multipliers or credits.",
            "selection_constraints": {"players": 11, "min_per_role": 1, "max_per_role": 8, "min_per_team": 1,
                                      "credit_budget": None, "captain_multipliers": False},
            "player_explanations": explanations, "team_win_probabilities": win_result, "comparison": comparison,
            "warnings": notes, "selection_errors": sorted(set(errors)), "charts": {}}
        details = pool[cols + ["avg_pts_last_5", "avg_pts_last_10", "last_pts", "previous_matches"]].copy()
        details["recent_average"] = details.avg_pts_last_5.fillna(bundle["fallback_mean"] if bundle["fallback_mean"] is not None else np.nan)
        details["flagged"] = details.data_quality_notes.ne("")
        details["eligible_for_selection"] = details.role.isin(ROLES)
        details["selected_predicted_xi"] = details.player_id.isin(predicted.player_id if predicted is not None else [])
        details["selected_actual_xi"] = details.player_id.isin(actual.player_id if actual is not None else [])
        report["player_predictions"] = details.to_dict("records")
        if charts and explanations:
            report["charts"] = render_charts(report, out_dir, self.root)
        else:
            for filename in ["player_form.png", "player_domains.png"]:
                (out_dir / filename).unlink(missing_ok=True)
        group[cols].to_csv(out_dir / "all_player_predictions.csv", index=False)
        for name, table in [("predicted_team", predicted), ("actual_team", actual)]:
            if table is not None:
                table[cols].to_csv(out_dir / f"{name}.csv", index=False)
            else:
                (out_dir / f"{name}.csv").unlink(missing_ok=True)
        report = json_safe(report)
        if record_request:
            from .dashboard_analytics import RequestLog
            report["request_key"] = RequestLog(self.root).record(self, report)
        (out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        return report

    def _player_evidence(self, row, start, bundle):
        past = self.player_rows[row.player_id]; past = past[past.end_date.lt(start)].tail(5)
        domain = {label: row[column] for label, column in [("Batting", "avg_batting_last_5"), ("Bowling", "avg_bowling_last_5"), ("Fielding", "avg_fielding_last_5")]}
        known = {k: v for k, v in domain.items() if pd.notna(v)}
        specialty = max(known, key=known.get) if known else "No earlier IPL history"
        sensitivities = []; model = bundle["regressor"]
        if model is not None:
            medians = model.named_steps["imputer"].statistics_
            vector = pd.DataFrame([row[FEATURES].to_dict()])
            variants = pd.concat([vector]*len(FEATURES), ignore_index=True)
            for i, feature in enumerate(FEATURES):
                variants.loc[i, feature] = medians[i]
            changed_predictions = model.predict(variants)
            for i, feature in enumerate(FEATURES):
                sensitivities.append({"feature": feature, "value": row[feature], "training_median": medians[i],
                                      "prediction_difference": float(row.predicted_points-changed_predictions[i])})
            sensitivities = sorted(sensitivities, key=lambda x: abs(x["prediction_difference"]), reverse=True)[:4]
        return {"player_id": row.player_id, "player": row.player, "team": row.team_key, "role": row.role,
            "predicted_points": row.predicted_points, "last_fantasy_points": row.last_pts,
            "last_5_average": row.avg_pts_last_5, "last_10_average": row.avg_pts_last_10,
            "last_5_matches_available": row.history_count_5, "previous_matches": row.previous_matches,
            "previous_opponent_points": row.last_opponent_pts, "opponent_average": row.avg_opponent_last_5,
            "specialty_from_previous_matches": specialty, "past_domain_averages": domain,
            "history": past[["match_id", "date", "end_date", "opponent", "total_fantasy_points", *COMPONENTS, "data_quality_notes"]].to_dict("records"),
            "feature_sensitivity": sensitivities,
            "explanation": "Selected in the maximum-predicted-total valid XI. Form uses earlier matches. Replacing one feature with its training median measures model sensitivity, not an additive or causal contribution."}

    def evaluate(self):
        bundle = self._fit_bundle(self.cutoff+pd.Timedelta(days=1))
        test = self.data[self.data.date.gt(self.cutoff)]
        result = {"cutoff": self.cutoff, "test_rows": len(test), "training_rows": bundle["training_rows"]}
        if len(test) and bundle["regressor"] is not None:
            p = bundle["regressor"].predict(test[FEATURES]); baseline = test.avg_pts_last_5.fillna(bundle["fallback_mean"])
            result.update(player_mae=mean_absolute_error(test.total_fantasy_points, p), recent_average_mae=mean_absolute_error(test.total_fantasy_points, baseline))
        win = self.win_data[self.win_data.date.gt(self.cutoff) & self.win_data.target.notna()]
        if len(win) and bundle["win_model"] is not None:
            p = bundle["win_model"].predict_proba(win[WIN_FEATURES])[:, 1]
            result.update(win_test_matches=len(win), win_brier=brier_score_loss(win.target, p), win_log_loss=log_loss(win.target, p),
                          win_accuracy=float(np.mean((p >= .5) == win.target)), neutral_50_percent_brier=.25)
        result["interpretation"] = "Chronological held-out evaluation, with actual recorded participants. Lower MAE/Brier is better. Experimental baseline: no predictive advantage is assumed. Earlier held-out results may enter later rolling features, never model fitting."
        return json_safe(result)


def render_charts(report, folder, root):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    explanations = report["player_explanations"]; output = {}
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), constrained_layout=True)
    for ax, player in zip(axes.flat, explanations):
        history = player["history"]
        if history:
            x = list(range(len(history)))
            ax.plot(x, [h["total_fantasy_points"] for h in history], marker="o", color="#176b87")
            ax.set_xticks(x, [pd.Timestamp(h["date"]).strftime("%d-%m-%y") for h in history], rotation=35, fontsize=8)
            ax.axhline(player["predicted_points"], color="#c75b12", linestyle="--")
        else:
            ax.text(.5, .5, "No earlier IPL history", ha="center", transform=ax.transAxes)
        ax.set_title(player["player"], fontsize=11); ax.set_ylabel("Fantasy points"); ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(explanations):]:
        ax.set_visible(False)
    fig.suptitle("Selected XI: previous five appearances (dashed line = prediction)", fontsize=15)
    path = folder / "player_form.png"; fig.savefig(path, dpi=130); plt.close(fig)
    output["player_form"] = path.relative_to(root).as_posix()
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    x = np.arange(len(explanations)); bottom = np.zeros(len(explanations))
    for label, color in [("Batting", "#176b87"), ("Bowling", "#45a187"), ("Fielding", "#eda451")]:
        values = np.nan_to_num(np.array([p["past_domain_averages"].get(label, np.nan) for p in explanations], float))
        ax.bar(x, values, bottom=bottom, label=label, color=color); bottom += values
    ax.set_xticks(x, [p["player"] for p in explanations], rotation=40, ha="right")
    ax.set_ylabel("Average points in previous five appearances")
    ax.set_title("Historical contribution by domain (missing history shown as zero)"); ax.legend()
    path = folder / "player_domains.png"; fig.savefig(path, dpi=130); plt.close(fig)
    output["player_domains"] = path.relative_to(root).as_posix()
    return output
