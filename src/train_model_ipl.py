"""Fit AdaBoost on Excel targets with chronological probability calibration."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ui_backend import FantasyService


def train_base_product_model():
    service = FantasyService()
    print(json.dumps(service.prepare(), indent=2))
    print(f"Model cache: {service.models_dir}")


if __name__ == "__main__":
    train_base_product_model()
