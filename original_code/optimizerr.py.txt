import pulp
import pandas as pd

def select_dream11_team(df, points_col):
    """
    Selects 11 players maximizing `points_col` while satisfying Dream11 rules.
    `df` must contain: ['player_name', 'team', 'role', points_col]
    """
    match_df = df.copy().reset_index(drop=True)
    prob = pulp.LpProblem("Dream11_Selection", pulp.LpMaximize)
    
    # Binary variables: x_i = 1 if player selected, else 0
    indices = match_df.index
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in indices}
    
    # Objective: Maximize total points
    prob += pulp.lpSum([x[i] * match_df.loc[i, points_col] for i in indices])
    
    # Constraint 1: Total 11 players
    prob += pulp.lpSum([x[i] for i in indices]) == 11
    
    # Constraint 2: Role limits (1 to 8 players per role)
    roles = ['BAT', 'BOWL', 'AR', 'WK']
    for role in roles:
        role_indices = match_df[match_df['role'].astype(str).str.upper() == role].index
        if len(role_indices) > 0:
            prob += pulp.lpSum([x[i] for i in role_indices]) >= 1
            prob += pulp.lpSum([x[i] for i in role_indices]) <= 8

    # Constraint 3: At least 1 player from each team
    teams = match_df['team'].unique()
    for t in teams:
        team_indices = match_df[match_df['team'] == t].index
        prob += pulp.lpSum([x[i] for i in team_indices]) >= 1

    # Solve ILP
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    selected_indices = [i for i in indices if pulp.value(x[i]) == 1]
    return match_df.loc[selected_indices].sort_values(by=points_col, ascending=False)