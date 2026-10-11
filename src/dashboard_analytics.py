"""Chronological diagnostics and a durable log of user-requested historical replays."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sqlite3

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import r2_score, log_loss
from .ui_backend import FEATURES, WIN_FEATURES, json_safe, save_cache
from .optimizerr import ROLES, select_dream11_team

ANALYTICS_VERSION = "dashboard-v1"


def point_metrics(data, column="predicted_points", tolerance=20):
    """No percentage 'accuracy' is fabricated for a regression target."""
    if data.empty:
        return {}
    valid = data[["total_fantasy_points", column]].dropna()
    valid = valid[np.isfinite(valid.to_numpy(float)).all(axis=1)]
    if valid.empty:
        return {}
    y, p = valid.total_fantasy_points.to_numpy(float), valid[column].to_numpy(float)
    error = p-y
    return {"rows": len(valid), "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error**2))),
            "r2": float(r2_score(y, p)) if len(y) > 1 and np.var(y) > 0 else None,
            "within_tolerance_percent": float(100*np.mean(np.abs(error) <= tolerance)),
            "tolerance": tolerance, "bias": float(error.mean())}


def winner_metrics(data):
    if data.empty:
        return {}
    valid = data.dropna(subset=["target", "probability_a"])
    if valid.empty:
        return {}
    y = valid.target.to_numpy(int); p = valid.probability_a.to_numpy(float)
    return {"matches": len(y), "correct": int(np.sum((p >= .5) == y)),
            "accuracy_percent": float(100*np.mean((p >= .5) == y)),
            "brier": float(np.mean((p-y)**2)), "neutral_brier": .25,
            "log_loss": float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1]))}


def selection_summary(table):
    if table.empty:
        return {}
    valid = table[table.selection_status.eq("ok")]
    result = {"matches": len(table), "scored_matches": len(valid), "unavailable_matches": len(table)-len(valid)}
    if len(valid):
        result.update(mean_overlap=float(valid.overlap.mean()), mean_regret=float(valid.regret.mean()),
                      mean_capture_percent=float(valid.capture_percent.mean()),
                      subset_matches=int(valid.pool_scope.ne("all_recorded_participants").sum()))
    return json_safe(result)


class DashboardAnalytics:
    def __init__(self, service):
        self.service = service
        self.key = f"{ANALYTICS_VERSION}_{service.fingerprint}_{service.cutoff:%Y-%m-%d}_{sklearn.__version__}"
        self.folder = service.root / "outputs/dashboard" / self.key

    def build(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        cache = self.folder / "diagnostics.joblib"
        if cache.exists():
            return joblib.load(cache)
        service = self.service
        validation_start = service.cutoff.replace(month=1, day=1)
        definitions = [
            ("Validation", validation_start, service.data.date.ge(validation_start) & service.data.end_date.le(service.cutoff)),
            ("Held-out test", service.cutoff+pd.Timedelta(days=1), service.data.date.gt(service.cutoff))]
        players, wins, selections, metadata, importances = [], [], [], [], []
        for split, training_limit, mask in definitions:
            bundle = service._fit_bundle(training_limit)
            rows = service.data[mask].copy()
            cols = ["match_id", "player_id", "player", "date", "team_key", "role", "data_quality_notes", "total_fantasy_points", "avg_pts_last_5"]
            scored = rows[cols].copy()
            scored["split"] = split
            scored["predicted_points"] = bundle["regressor"].predict(rows[FEATURES]) if len(rows) and bundle["regressor"] is not None else np.nan
            scored["recent_average"] = rows.avg_pts_last_5.fillna(bundle["fallback_mean"] if bundle["fallback_mean"] is not None else np.nan)
            scored["error"] = scored.predicted_points-scored.total_fantasy_points
            scored["absolute_error"] = scored.error.abs()
            scored["flagged"] = scored.data_quality_notes.ne("")
            players.append(scored)
            win_rows = service.win_data[service.win_data.match_id.isin(rows.match_id.unique()) & service.win_data.target.notna()].copy()
            ws = win_rows[["match_id", "date", "team_a", "team_b", "target"]].copy()
            ws["split"] = split
            ws["probability_a"] = bundle["win_model"].predict_proba(win_rows[WIN_FEATURES])[:, 1] if len(win_rows) and bundle["win_model"] is not None else np.nan
            ws["predicted_winner"] = np.where(ws.probability_a >= .5, ws.team_a, ws.team_b)
            ws["actual_winner"] = np.where(ws.target == 1, ws.team_a, ws.team_b)
            ws.loc[ws.probability_a.isna(), "predicted_winner"] = None
            wins.append(ws)
            selections.extend(self._select_matches(scored, split))
            metadata.append({"split": split, "training_end_exclusive": bundle["limit_exclusive"],
                "training_rows": bundle["training_rows"], "excluded_training_rows": bundle["excluded_training_rows"],
                "evaluation_start": rows.date.min(), "evaluation_end": rows.end_date.max(),
                "evaluation_matches": rows.match_id.nunique(), "evaluation_rows": len(rows),
                "win_training_matches": bundle["win_training_matches"], "calibration_matches": bundle["calibration_matches"]})
            if bundle["regressor"] is not None:
                model = bundle["regressor"]
                names = model.named_steps["imputer"].get_feature_names_out(FEATURES)
                importances.extend({"split": split, "feature": n, "importance": float(v)}
                    for n, v in zip(names, model.named_steps["model"].feature_importances_))
        result = {"players": pd.concat(players, ignore_index=True), "wins": pd.concat(wins, ignore_index=True),
                  "selections": pd.DataFrame(selections), "metadata": json_safe(metadata),
                  "importances": pd.DataFrame(importances), "key": self.key}
        metrics = {split: {"points": point_metrics(result["players"][result["players"].split.eq(split)]),
                          "recent_average": point_metrics(result["players"][result["players"].split.eq(split)], "recent_average"),
                          "winner": winner_metrics(result["wins"][result["wins"].split.eq(split)]),
                          "selection": selection_summary(result["selections"][result["selections"].split.eq(split)])}
                   for split, _, _ in definitions}
        for key in ["players", "wins", "selections", "importances"]:
            result[key].to_csv(self.folder / f"{key}.csv", index=False)
        (self.folder / "metrics.json").write_text(json.dumps(json_safe({"splits": result["metadata"], "metrics": metrics}), indent=2, allow_nan=False), encoding="utf-8")
        save_cache(result, cache)
        return result

    def _select_matches(self, scored, split):
        records = []
        for mid, group in scored.groupby("match_id", sort=False):
            meta = self.service.metadata.get(mid, {})
            roster = meta.get("roster", {})
            pool = group[group.player_id.isin(roster)].copy() if roster else group.copy()
            missing = len(set(roster)-set(group.player_id)) if roster else 0
            unknown = int((~pool.role.isin(ROLES)).sum())
            pool = pool[pool.role.isin(ROLES)].copy(); pool["team"] = pool.team_key
            item = {"match_id": mid, "date": group.date.iloc[0], "split": split,
                    "pool_scope": "known_role_subset" if missing or unknown else "all_recorded_participants",
                    "unknown_roles": unknown, "missing_excel_players": missing,
                    "quality_flagged": bool(pool.flagged.any()), "selection_status": "unavailable"}
            try:
                pred = select_dream11_team(pool, "predicted_points")
                actual = select_dream11_team(pool, "total_fantasy_points")
                baseline = select_dream11_team(pool, "recent_average")
                optimum = float(actual.total_fantasy_points.sum()); realized = float(pred.total_fantasy_points.sum())
                item.update(selection_status="ok", predicted_xi_actual=realized, oracle_points=optimum,
                            baseline_xi_actual=float(baseline.total_fantasy_points.sum()),
                            overlap=len(set(pred.player_id)&set(actual.player_id)), regret=optimum-realized,
                            capture_percent=100*realized/optimum if optimum > 0 else None)
            except ValueError as e:
                item["reason"] = str(e)
            records.append(item)
        return records


class RequestLog:
    """One result per input/model/date/match, regardless of repeated clicks."""
    def __init__(self, root):
        self.path = Path(root) / "outputs/request_history.sqlite3"

    def record(self, service, report):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        signature = f"{service.fingerprint}:{sklearn.__version__}:{service.cutoff}:{report['model']['training_end_exclusive']}:{report['match']['match_id']}"
        key = hashlib.sha256(signature.encode()).hexdigest()
        row = {"request_key": key, "model_fingerprint": service.fingerprint, "sklearn_version": sklearn.__version__,
               "configured_cutoff": service.cutoff.strftime("%Y-%m-%d"), "match_id": report["match"]["match_id"],
               "match_date": report["match"]["date"], "query_mode": report["model"]["query_mode"],
               "requested_date": report["match"]["requested_date"], "algorithm": report["model"]["algorithm"],
               "model_training_end_exclusive": report["model"]["training_end_exclusive"],
               "pool_scope": report["candidate_pool"]["scope"], "comparison": report["comparison"],
               "player_results": report.get("player_predictions", []),
               "team_win_probabilities": report["team_win_probabilities"], "actual_winner": report["match"]["actual_winner"],
               "evaluation_kind": "historical_replay_requested_in_ui"}
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.path) as con:
            con.execute("CREATE TABLE IF NOT EXISTS requests (request_key TEXT PRIMARY KEY, first_requested_at TEXT, last_requested_at TEXT, request_count INTEGER, payload TEXT)")
            con.execute("""INSERT INTO requests VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(request_key) DO UPDATE SET last_requested_at=excluded.last_requested_at,
                request_count=requests.request_count+1, payload=excluded.payload""",
                (key, now, now, json.dumps(json_safe(row), allow_nan=False)))
        return key

    def read(self, service):
        if not self.path.exists():
            return {"requests": pd.DataFrame(), "players": pd.DataFrame(), "wins": pd.DataFrame()}
        requests, players, wins = [], [], []
        with sqlite3.connect(self.path) as con:
            records = con.execute("SELECT first_requested_at,last_requested_at,request_count,payload FROM requests ORDER BY first_requested_at").fetchall()
        for first, last, count, payload in records:
            row = json.loads(payload)
            if row["model_fingerprint"] != service.fingerprint or row["configured_cutoff"] != service.cutoff.strftime("%Y-%m-%d") or row["sklearn_version"] != sklearn.__version__:
                continue
            common = {"request_key": row["request_key"], "match_id": row["match_id"], "date": row["match_date"], "query_mode": row["query_mode"]}
            frame = pd.DataFrame(row["player_results"])
            pm = point_metrics(frame)
            win = row["team_win_probabilities"]["probabilities"]
            if len(win) == 2 and row["actual_winner"] in [x["team"] for x in win]:
                wins.append({**common, "team_a": win[0]["team"], "team_b": win[1]["team"],
                             "probability_a": win[0].get("probability", win[0]["win_percent"]/100), "target": int(row["actual_winner"] == win[0]["team"])})
            for player in row["player_results"]:
                players.append({**player, **common})
            comparison = row["comparison"] or {}
            best = comparison.get("actual_best_team_points", 0)
            requests.append({**common, "first_requested_at": first, "last_requested_at": last, "request_count": count,
                             "algorithm": row["algorithm"], "pool_scope": row["pool_scope"], "player_mae": pm.get("mae"),
                             "overlap": comparison.get("overlap"), "regret": comparison.get("selection_regret"),
                             "predicted_xi_actual": comparison.get("predicted_team_actual_points"),
                             "oracle_points": comparison.get("actual_best_team_points"),
                             "capture_percent": 100*comparison["predicted_team_actual_points"]/best if best and best > 0 else None})
        return {"requests": pd.DataFrame(requests), "players": pd.DataFrame(players), "wins": pd.DataFrame(wins)}
