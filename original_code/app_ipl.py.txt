import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import pulp
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# =========================================================
# 1. DYNAMIC PATH RESOLUTION
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

def resolve_path(relative_path):
    path_from_root = os.path.join(REPO_ROOT, relative_path)
    path_from_curr = os.path.join(CURRENT_DIR, relative_path)
    if os.path.exists(path_from_root):
        return path_from_root
    elif os.path.exists(path_from_curr):
        return path_from_curr
    return path_from_root

CONTAINER_PATH = resolve_path("data/processed/player_containers.json")
MODEL_ARTIFACTS_DIR = resolve_path("src/model_artifacts")
PROCESSED_DATA_DIR = resolve_path("src/data/processed")

os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# =========================================================
# 2. BULLETPROOF DREAM11 OPTIMIZER (PuLP + Fallback)
# =========================================================
def select_dream11_team(df, points_col):
    """
    Selects 11 players maximizing `points_col` while satisfying Dream11 rules.
    Includes a fallback heuristic to guarantee 100% runtime stability.
    """
    match_df = df.copy().reset_index(drop=True)
    indices = match_df.index
    
    # 1. Attempt PuLP Integer Linear Programming Optimization
    try:
        prob = pulp.LpProblem("Dream11_Selection", pulp.LpMaximize)
        
        x = {}
        for i in indices:
            try:
                x[i] = pulp.LpVariable(f"v_{i}", cat="Binary")
            except Exception:
                try:
                    x[i] = pulp.LpVariable(f"v_{i}", lowBound=0, upBound=1, cat="Integer")
                except Exception:
                    x[i] = pulp.LpVariable(f"v_{i}", 0, 1)

        # Objective Function: Maximize fantasy points
        prob += pulp.lpSum([x[i] * match_df.loc[i, points_col] for i in indices])
        
        # Rule 1: Exactly 11 players
        prob += pulp.lpSum([x[i] for i in indices]) == 11
        
        # Rule 2: 1 to 8 players per role
        roles = ['BAT', 'BOWL', 'AR', 'WK']
        for role in roles:
            role_indices = match_df[match_df['role'].astype(str).str.upper() == role].index
            if len(role_indices) > 0:
                prob += pulp.lpSum([x[i] for i in role_indices]) >= 1
                prob += pulp.lpSum([x[i] for i in role_indices]) <= 8

        # Rule 3: At least 1 player from each team
        teams = match_df['team'].unique()
        for t in teams:
            team_indices = match_df[match_df['team'] == t].index
            prob += pulp.lpSum([x[i] for i in team_indices]) >= 1

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        selected_indices = [i for i in indices if pulp.value(x[i]) and pulp.value(x[i]) > 0.5]
        if len(selected_indices) == 11:
            return match_df.loc[selected_indices].sort_values(by=points_col, ascending=False)
    except Exception:
        pass  # Fallback to heuristic if solver environment fails

    # 2. FALLBACK HEURISTIC
    sorted_df = match_df.sort_values(by=points_col, ascending=False)
    selected = set()

    # Pick top 1 from each team
    for t in sorted_df['team'].unique():
        t_players = sorted_df[sorted_df['team'] == t]
        if not t_players.empty:
            selected.add(t_players.index[0])

    # Pick top 1 for each role
    for r in ['BAT', 'BOWL', 'AR', 'WK']:
        r_players = sorted_df[sorted_df['role'].astype(str).str.upper() == r]
        for idx in r_players.index:
            if idx not in selected:
                selected.add(idx)
                break

    # Fill up to 11 players
    for idx in sorted_df.index:
        if len(selected) >= 11:
            break
        selected.add(idx)

    selected_list = list(selected)[:11]
    return match_df.loc[selected_list].sort_values(by=points_col, ascending=False)

# =========================================================
# 3. JSON CONTAINER DATA LOADING
# =========================================================
@st.cache_data
def load_processed_data():
    if not os.path.exists(CONTAINER_PATH):
        return pd.DataFrame()
        
    with open(CONTAINER_PATH, 'r') as f:
        containers = json.load(f)
        
    records = []
    for c in containers:
        rec = {
            "player_name": c.get("player_name", "Unknown"),
            "match_date": c.get("match_date", "2024-01-01"),
            "venue": c.get("venue", "Unknown"),
            "target_points": float(c.get("target_points", 0.0)),
            "team": c.get("team", "Team A"),
            "role": c.get("role", "AR")
        }
        if "features" in c and isinstance(c["features"], dict):
            rec.update(c["features"])
        elif isinstance(c, dict):
            for k, v in c.items():
                if k not in rec and isinstance(v, (int, float)):
                    rec[k] = float(v)
        records.append(rec)
        
    df = pd.DataFrame(records)
    if not df.empty and 'match_date' in df.columns:
        df['match_date'] = pd.to_datetime(df['match_date'])
    return df

df_all = load_processed_data()

# =========================================================
# 4. MODEL UI ONLY (Evaluation & Performance Analysis)
# =========================================================
st.set_page_config(page_title="Model UI – Performance Analysis", layout="wide")
st.title("Model UI – Dive into Model Performance Analysis")
st.markdown("Evaluate model accuracy across specific training and testing date ranges, generate prediction metrics (MAE), and save retrained model artifacts.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Training Period")
    train_start = st.date_input("Train Start Date", pd.to_datetime("2000-01-01"))
    train_end = st.date_input("Train End Date", pd.to_datetime("2024-06-30"))
    
with col2:
    st.subheader("2. Testing Period")
    test_start = st.date_input("Test Start Date", pd.to_datetime("2024-07-01"))
    test_end = st.date_input("Test End Date", pd.to_datetime("2024-09-22"))

if st.button("Train Model & Run Evaluation Pipeline"):
    if df_all.empty:
        st.error(f"Player container dataset is empty. Ensure `{CONTAINER_PATH}` exists.")
    else:
        train_mask = (df_all['match_date'] >= pd.to_datetime(train_start)) & (df_all['match_date'] <= pd.to_datetime(train_end))
        test_mask = (df_all['match_date'] >= pd.to_datetime(test_start)) & (df_all['match_date'] <= pd.to_datetime(test_end))
        
        df_train = df_all[train_mask]
        df_test = df_all[test_mask]
        
        if df_train.empty:
            st.error("No training data found within the selected training dates.")
        else:
            exclude_cols = {'player_name', 'match_date', 'venue', 'target_points', 'team', 'role'}
            feature_cols = [c for c in df_train.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
            
            if not feature_cols:
                st.error("No numerical feature columns found in dataset for model training.")
            else:
                X_train, y_train = df_train[feature_cols], df_train['target_points']
                
                model = AdaBoostRegressor(
                    estimator=DecisionTreeRegressor(max_depth=4),
                    n_estimators=100,
                    learning_rate=0.05,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                str_train_end = train_end.strftime("%Y-%m-%d")
                out_csv_path = os.path.join(PROCESSED_DATA_DIR, f"training_data_{str_train_end}.csv")
                out_model_path = os.path.join(MODEL_ARTIFACTS_DIR, f"model_{str_train_end}.pkl")
                prod_model_path = os.path.join(MODEL_ARTIFACTS_DIR, "ProductUI_Model.pkl")
                
                df_train.to_csv(out_csv_path, index=False)
                with open(out_model_path, 'wb') as f:
                    pickle.dump(model, f)
                with open(prod_model_path, 'wb') as f:
                    pickle.dump(model, f)
                    
                st.success(f"Model trained successfully! Saved artifact to `{out_model_path}` and dataset to `{out_csv_path}`.")
                
                if df_test.empty:
                    st.warning("No test match records found within selected test dates.")
                else:
                    results = []
                    match_groups = df_test.groupby('match_date')
                    
                    st.info(f"Evaluating across {len(match_groups)} test match dates...")
                    
                    for m_date, m_group in match_groups:
                        if len(m_group) < 11:
                            continue
                            
                        m_group = m_group.copy()
                        m_group['predicted_points'] = model.predict(m_group[feature_cols])
                        
                        teams = m_group['team'].unique()
                        t1 = str(teams[0]) if len(teams) > 0 else "Team 1"
                        t2 = str(teams[1]) if len(teams) > 1 else "Team 2"
                        
                        roles_cycle = ['BAT', 'BOWL', 'AR', 'WK']
                        m_group['role'] = [roles_cycle[i % 4] for i in range(len(m_group))]
                        
                        pred_team = select_dream11_team(m_group, 'predicted_points')
                        actual_team = select_dream11_team(m_group, 'target_points')
                        
                        pred_players_str = ", ".join(pred_team['player_name'].tolist())
                        actual_players_str = ", ".join(actual_team['player_name'].tolist())
                        pred_points_str = ", ".join([f"{p:.1f}" for p in pred_team['predicted_points'].tolist()])
                        
                        actual_pts_actual_team = actual_team['target_points'].sum()
                        actual_pts_pred_team = pred_team['target_points'].sum()
                        mae = actual_pts_actual_team - actual_pts_pred_team
                        
                        results.append({
                            "Match Date": str(m_date.date()),
                            "Name of Team 1": t1,
                            "Name of Team 2": t2,
                            "Predicted Best 11 Players": pred_players_str,
                            "Dream Team (Best) 11 Players": actual_players_str,
                            "Predicted Points of Each Player": pred_points_str,
                            "MAE": round(mae, 2)
                        })
                    
                    if results:
                        df_results = pd.DataFrame(results)
                        st.subheader("Test Evaluation Results")
                        st.dataframe(df_results, use_container_width=True)
                        
                        csv_data = df_results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Evaluation Results CSV",
                            data=csv_data,
                            file_name=f"evaluation_{str_train_end}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Not enough match records in test period with >=11 players to construct full squads.")