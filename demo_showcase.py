#!/usr/bin/env python3
"""Quick project demo for the Four Worlds Adversarial Fraud Detection Lab.

This script is intentionally lightweight so it can run quickly in a demo or
presentation setting while still showcasing the main idea of the project:

- load the PaySim dataset
- engineer transaction features
- train a fraud detector
- generate adversarial campaign attacks
- evaluate campaign outcomes and print a compact summary
"""

from __future__ import annotations

import json
from typing import Dict, List

from src.campaigns.static_generator import StaticCampaignGenerator
from src.data_processor import DataProcessor
from src.evaluation.metrics import Evaluator
from src.models.xgboost_detector import XGBoostDetector
from src.simulation.simulator import SimulationEngine


def run_demo() -> Dict[str, Dict[str, float]]:
    print("=" * 72)
    print("Four Worlds Adversarial Fraud Detection Lab - Demo")
    print("=" * 72)

    processor = DataProcessor()
    raw_df = processor.load_and_sample()
    feat_df = processor.engineer_features(raw_df)
    (X_train, y_train, _), (X_test, y_test, _) = processor.prepare_world1_split(feat_df)

    legit_bg = X_test[y_test == 0].copy()
    legit_bg = legit_bg.sample(n=min(len(legit_bg), 500), random_state=42).reset_index(drop=True)

    print("\nTraining detector...")
    detector = XGBoostDetector()
    detector.fit(X_train, y_train)

    sim_engine = SimulationEngine(detector, scaler=processor.scaler)
    generator = StaticCampaignGenerator()

    world_results: Dict[str, Dict[str, float]] = {}
    for world_name, campaign_factory in [
        ("World 1", lambda: generator.generate_known_campaigns(num_campaigns=3)),
        ("World 3", lambda: generator.generate_unknown_campaigns(num_campaigns=3)),
    ]:
        campaigns = campaign_factory()
        evaluated = sim_engine.run_campaigns(campaigns)
        metrics = Evaluator.compute_campaign_metrics(
            evaluated,
            detector=detector,
            bg_test_df=legit_bg,
            scaler=processor.scaler,
            seed=42,
        )
        world_results[world_name] = metrics

        print(f"\n{world_name}:")
        for key in ["CSR", "CFE", "IT", "F1", "Precision", "Recall"]:
            print(f"  {key}: {metrics[key]:.4f}")

    print("\nSummary:")
    print(json.dumps(world_results, indent=2, sort_keys=True))
    return world_results


if __name__ == "__main__":
    run_demo()
