import numpy as np
import pulp

# Role limits: (minimum required, maximum allowed)
ROLE_BOUNDS = {
    "WK": (1, 8),
    "BAT": (3, 8),
    "AR": (1, 8),
    "BOWL": (3, 8),
}


def select_dream11_team(df, points_col):
    data = df.copy().reset_index(drop=True)
    if data.player_id.duplicated().any():
        raise ValueError("Duplicate candidate player IDs")
    if len(data) < 11 or data.team.nunique() != 2:
        raise ValueError("Selection requires at least 11 candidates from two teams")
    if not data.role.isin(ROLE_BOUNDS.keys()).all():
        raise ValueError("Resolve unknown roles before selecting from this pool")
    if not np.isfinite(data[points_col].to_numpy(float)).all():
        raise ValueError("Every candidate needs a finite score")

    # Validate that the candidate pool contains enough players for each role minimum
    for role, (min_req, _) in ROLE_BOUNDS.items():
        if (data.role == role).sum() < min_req:
            raise ValueError(f"Insufficient candidates for role '{role}': required at least {min_req}")

    problem = pulp.LpProblem("Fantasy_XI", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"player_{i}", cat="Binary") for i in data.index}
    problem += pulp.lpSum(x[i] * float(data.loc[i, points_col]) for i in data.index)
    problem += pulp.lpSum(x.values()) == 11

    # Enforce role-specific minimums and maximums
    for role, (min_req, max_req) in ROLE_BOUNDS.items():
        count = pulp.lpSum(x[i] for i in data.index[data.role.eq(role)])
        problem += count >= min_req
        problem += count <= max_req

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