"""Exact XI selection using the original project's PuLP solver."""
import numpy as np
import pulp

ROLES = ("BAT", "BOWL", "AR", "WK")


def select_dream11_team(df, points_col):
    data = df.copy().reset_index(drop=True)
    if data.player_id.duplicated().any():
        raise ValueError("Duplicate candidate player IDs")
    if len(data) < 11 or data.team.nunique() != 2:
        raise ValueError("Selection requires at least 11 candidates from two teams")
    if not data.role.isin(ROLES).all():
        raise ValueError("Resolve unknown roles before selecting from this pool")
    if not np.isfinite(data[points_col].to_numpy(float)).all():
        raise ValueError("Every candidate needs a finite score")
    missing = set(ROLES) - set(data.role)
    if missing:
        raise ValueError(f"No candidate for required roles: {sorted(missing)}")
    problem = pulp.LpProblem("Fantasy_XI", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"player_{i}", cat="Binary") for i in data.index}
    problem += pulp.lpSum(x[i] * float(data.loc[i, points_col]) for i in data.index)
    problem += pulp.lpSum(x.values()) == 11
    for role in ROLES:
        count = pulp.lpSum(x[i] for i in data.index[data.role.eq(role)])
        problem += count >= 1
        problem += count <= 8
    for team in data.team.unique():
        problem += pulp.lpSum(x[i] for i in data.index[data.team.eq(team)]) >= 1
    # CBC's worker-thread mode can stall on these tiny problems; use its serial solver.
    problem.solve(pulp.PULP_CBC_CMD(msg=False, threads=0, timeLimit=30))
    if pulp.LpStatus[problem.status] != "Optimal":
        raise ValueError(f"No optimal valid XI: {pulp.LpStatus[problem.status]}")
    selected = data.loc[[i for i in data.index if pulp.value(x[i]) > .5]]
    if len(selected) != 11:
        raise RuntimeError("Solver returned an invalid XI")
    return selected.sort_values([points_col, "player_id"], ascending=[False, True])