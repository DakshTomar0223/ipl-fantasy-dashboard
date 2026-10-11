import os
import json
import pickle
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

def train_base_product_model():
    os.makedirs("src/model_artifacts", exist_ok=True)
    container_path = "./data/processed/player_containers.json"
    
    if not os.path.exists(container_path):
        print(f"Error: {container_path} not found. Run preprocessing first.")
        return

    with open(container_path, 'r') as f:
        containers = json.load(f)

    records = []
    for c in containers:
        rec = {
            "player_name": c["player_name"],
            "match_date": c["match_date"],
            "target_points": c.get("target_points", 0.0)
        }
        rec.update(c["features"])
        records.append(rec)

    df = pd.DataFrame(records)
    df['match_date'] = pd.to_datetime(df['match_date'])
    
    # STRICT CUTOFF: No data after 2024-06-30
    df_cutoff = df[df['match_date'] <= pd.to_datetime("2024-06-30")]
    
    feature_cols = [c for c in df_cutoff.columns if c.startswith("avg_") or c.startswith("max_") or c.startswith("std_")]
    X = df_cutoff[feature_cols]
    y = df_cutoff['target_points']

    model = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4),
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X, y)

    # Save to src/model_artifacts/ProductUI_Model.pkl
    out_path = "src/model_artifacts/ProductUI_Model.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Successfully trained and saved pretrained model to {out_path}")

if __name__ == "__main__":
    train_base_product_model()