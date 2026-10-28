#!/usr/bin/env python3
"""Presentation-ready demo for the Four Worlds fraud lab.

This version is intentionally compact, readable, and designed for a live demo or
short presentation. It shows the core storyline of the project:

1. load fraud data
2. engineer transaction features
3. train a detector
4. simulate known and unknown fraud campaigns
5. summarize the results in a compact report
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from src.campaigns.static_generator import StaticCampaignGenerator
from src.data_processor import DataProcessor
from src.evaluation.metrics import Evaluator
from src.models.xgboost_detector import XGBoostDetector
from src.simulation.simulator import SimulationEngine


@dataclass
class DemoResult:
    world: str
    csr: float
    cfe: float
    it: float
    f1: float
    precision: float
    recall: float


def print_banner() -> None:
    print("=" * 88)
    print("FOUR WORLDS ADVERSARIAL FRAUD DETECTION LAB")
    print("Demo showcase for business / research presentation")
    print("=" * 88)
    print("Goal: show how fraud detection behaves under known vs unknown attack patterns.")
    print("Detects: XGBoost baseline on engineered PaySim transaction features")
    print("Attack modes: static campaign generation + simulation-based evaluation")
    print("=" * 88)


def evaluate_world(name: str, campaigns: List, detector, processor: DataProcessor, legit_bg: pd.DataFrame) -> DemoResult:
    sim_engine = SimulationEngine(detector, scaler=processor.scaler)
    evaluated = sim_engine.run_campaigns(campaigns)
    metrics = Evaluator.compute_campaign_metrics(
        evaluated,
        detector=detector,
        bg_test_df=legit_bg,
        scaler=processor.scaler,
        seed=42,
    )
    return DemoResult(
        world=name,
        csr=float(metrics["CSR"]),
        cfe=float(metrics["CFE"]),
        it=float(metrics["IT"]),
        f1=float(metrics["F1"]),
        precision=float(metrics["Precision"]),
        recall=float(metrics["Recall"]),
    )


def main() -> None:
    print_banner()

    processor = DataProcessor()
    raw_df = processor.load_and_sample()
    feat_df = processor.engineer_features(raw_df)
    (X_train, y_train, _), (X_test, y_test, _) = processor.prepare_world1_split(feat_df)

    legit_bg = X_test[y_test == 0].copy()
    legit_bg = legit_bg.sample(n=min(len(legit_bg), 500), random_state=42).reset_index(drop=True)

    print("\n[1/4] Training the fraud detector...")
    detector = XGBoostDetector()
    detector.fit(X_train, y_train)

    generator = StaticCampaignGenerator()
    world_configs = [
        ("World 1", generator.generate_known_campaigns(num_campaigns=3)),
        ("World 3", generator.generate_unknown_campaigns(num_campaigns=3)),
    ]

    results: List[DemoResult] = []
    for world_name, campaign_list in world_configs:
        print(f"\n[2/4] Running {world_name}...")
        result = evaluate_world(world_name, campaign_list, detector, processor, legit_bg)
        results.append(result)

        print(f"  Campaign success rate (CSR): {result.csr:.3f}")
        print(f"  Cumulative exposure (CFE):  {result.cfe:.0f}")
        print(f"  Intervention timing (IT):    {result.it:.2f}")
        print(f"  F1 score:                   {result.f1:.3f}")
        print(f"  Precision:                  {result.precision:.3f}")
        print(f"  Recall:                     {result.recall:.3f}")

    print("\n[3/4] Takeaway")
    print("The key story of the lab is not only one raw accuracy number.")
    print("It is how the detector behaves when fraud changes from familiar to unseen patterns.")
    print("A model can look strong on known fraud, but degrade sharply under adversarial evolution.")

    print("\n[4/4] Compact result table")
    table = pd.DataFrame([
        {
            "World": r.world,
            "CSR": round(r.csr, 3),
            "CFE": round(r.cfe, 1),
            "IT": round(r.it, 2),
            "F1": round(r.f1, 3),
            "Precision": round(r.precision, 3),
            "Recall": round(r.recall, 3),
        }
        for r in results
    ])
    print(table.to_string(index=False))

    print("\nDemo complete. This project is a research lab for measuring adversarial robustness in fraud detection.")


if __name__ == "__main__":
    main()
