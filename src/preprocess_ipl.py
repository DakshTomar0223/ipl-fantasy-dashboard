"""Rebuild the UI feature cache from the supplied fantasy workbook."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ui_backend import FantasyService, FEATURES, WIN_FEATURES


def build_containers():
    service = FantasyService()
    out = service.root / "data/processed"
    service.data[["match_id", "player_id", "date", "total_fantasy_points", *FEATURES]].to_csv(out / "ui_player_features.csv", index=False)
    service.win_data.drop(columns="evidence").to_csv(out / "ui_match_features.csv", index=False)
    print(f"Saved {len(service.data):,} player-match rows and {len(service.win_data):,} match rows to {out}")


if __name__ == "__main__":
    build_containers()
