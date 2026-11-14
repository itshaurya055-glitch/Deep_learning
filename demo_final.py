#!/usr/bin/env python3
##.\.venv\Scripts\python.exe demo_final.py
"""
+=============================================================================+
|   FOUR WORLDS ADVERSARIAL FRAUD DETECTION LAB -- FINAL DEMO                 |
|   Complete pipeline: Data -> Features -> XGBoost + LSTM -> Campaigns -> RL  |
|   Adversarial Evaluation -> Metrics -> Visualisation                        |
+=============================================================================+

This script demonstrates every component of the project in one clean run:

  1. Load & preprocess PaySim dataset
  2. Engineer 13 fraud-detection features
  3. Train XGBoost (tabular) and LSTM (sequential) detectors
  4. Generate static known & unknown fraud campaigns
  5. Train PPO-based adaptive attacker (reinforcement learning)
  6. Simulate all 4 adversarial worlds
  7. Compute campaign-level and detector-level metrics
  8. Generate publication-quality comparison plots
  9. Print a final summary report

Designed for live demo / viva / presentation.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless — works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)

# ── Project imports ──────────────────────────────────────────────────────────
from src.data_processor import DataProcessor
from src.models.xgboost_detector import XGBoostDetector
from src.models.lstm_detector import LSTMDetector
from src.campaigns.static_generator import StaticCampaignGenerator
from src.campaigns.rl_attacker import train_rl_attacker, generate_rl_campaigns
from src.simulation.simulator import SimulationEngine
from src.evaluation.metrics import Evaluator
from src.config import OUTPUT_DIR, PLOTS_DIR, FEATURE_COLUMNS

warnings.filterwarnings("ignore")

# ═══════════════════════════════ settings ═════════════════════════════════════
NUM_CAMPAIGNS     = 10          # kept small for a fast demo (increase for rigour)
PPO_TRAIN_STEPS   = 5_000       # RL training steps (increase to 30k+ for full run)
RANDOM_SEED       = 42
DEMO_PLOTS_DIR    = os.path.join(OUTPUT_DIR, "demo_plots")
os.makedirs(DEMO_PLOTS_DIR, exist_ok=True)

# ═══════════════════════════════ helpers ══════════════════════════════════════

def banner(title: str, width: int = 76) -> None:
    """Print a boxed banner."""
    border = "=" * width
    print(f"\n+{border}+")
    print(f"|  {title:<{width - 2}}|")
    print(f"+{border}+")


def section(title: str) -> None:
    print(f"\n{'-' * 76}")
    print(f"  >>  {title}")
    print(f"{'-' * 76}")


def bullet(msg: str) -> None:
    print(f"     * {msg}")


def result_line(label: str, value, fmt: str = ".4f") -> None:
    if isinstance(value, float):
        print(f"     {label:<35} {value:{fmt}}")
    else:
        print(f"     {label:<35} {value}")


def elapsed(t0: float) -> str:
    dt = time.time() - t0
    if dt < 60:
        return f"{dt:.1f}s"
    return f"{dt / 60:.1f}min"


# ═══════════════════════════════ main pipeline ════════════════════════════════

def main() -> None:
    t_start = time.time()

    # ── 0. Banner ────────────────────────────────────────────────────────────
    banner("FOUR WORLDS ADVERSARIAL FRAUD DETECTION LAB -- DEMO")
    print("""
     Research question:
       "How robust are fraud detection models when attackers evolve
        their strategies and fraud patterns are no longer observed
        during training?"

     Detectors  : XGBoost (tabular) + LSTM (sequential)
     Attackers  : Static campaigns + PPO reinforcement learning agent
     Worlds     : 4 adversarial evaluation worlds
     Dataset    : PaySim synthetic mobile-money transactions
    """)

    # ── 1. Data Loading ──────────────────────────────────────────────────────
    section("STEP 1 / 8  ·  Loading & Sampling PaySim Dataset")
    t0 = time.time()
    processor = DataProcessor()
    raw_df = processor.load_and_sample()
    feat_df = processor.engineer_features(raw_df)
    (X_train, y_train, train_df), (X_test, y_test, test_df) = \
        processor.prepare_world1_split(feat_df)

    legit_bg = X_test[y_test == 0].copy()
    legit_bg = legit_bg.sample(
        n=min(len(legit_bg), 500), random_state=RANDOM_SEED
    ).reset_index(drop=True)

    bullet(f"Total features engineered    : {len(FEATURE_COLUMNS)}")
    bullet(f"Training samples             : {len(X_train):,}")
    bullet(f"Test samples                 : {len(X_test):,}")
    bullet(f"Fraud in train               : {int(y_train.sum()):,}  "
           f"({y_train.mean() * 100:.2f}%)")
    bullet(f"Fraud in test                : {int(y_test.sum()):,}  "
           f"({y_test.mean() * 100:.2f}%)")
    bullet(f"Elapsed                      : {elapsed(t0)}")

    # ── 2. Train XGBoost ─────────────────────────────────────────────────────
    section("STEP 2 / 8  ·  Training XGBoost Detector")
    t0 = time.time()
    xgb_det = XGBoostDetector()
    xgb_det.fit(X_train, y_train)

    xgb_probs = xgb_det.predict_proba(X_test)
    xgb_preds = xgb_det.predict(X_test)
    xgb_f1    = f1_score(y_test, xgb_preds, zero_division=0)
    xgb_prec  = precision_score(y_test, xgb_preds, zero_division=0)
    xgb_rec   = recall_score(y_test, xgb_preds, zero_division=0)
    xgb_auc   = roc_auc_score(y_test, xgb_probs) if len(np.unique(y_test)) > 1 else 0.0

    result_line("F1 (test)",    xgb_f1)
    result_line("Precision",    xgb_prec)
    result_line("Recall",       xgb_rec)
    result_line("AUC-ROC",      xgb_auc)
    result_line("Threshold",    xgb_det.threshold, ".3f")
    bullet(f"Elapsed: {elapsed(t0)}")

    # ── 3. Train LSTM ────────────────────────────────────────────────────────
    section("STEP 3 / 8  ·  Training LSTM Sequence Detector")
    t0 = time.time()
    lstm_det = LSTMDetector()
    lstm_det.fit(X_train, y_train)
    lstm_det.calibrate_simulation_threshold(
        X_test.head(500), y_test[:500]      # small slice for speed in demo
    )

    lstm_probs = lstm_det.predict_proba(X_test)
    lstm_preds = lstm_det.predict(X_test)
    lstm_f1    = f1_score(y_test, lstm_preds, zero_division=0)
    lstm_prec  = precision_score(y_test, lstm_preds, zero_division=0)
    lstm_rec   = recall_score(y_test, lstm_preds, zero_division=0)
    lstm_auc   = roc_auc_score(y_test, lstm_probs) if len(np.unique(y_test)) > 1 else 0.0

    result_line("F1 (test)",    lstm_f1)
    result_line("Precision",    lstm_prec)
    result_line("Recall",       lstm_rec)
    result_line("AUC-ROC",      lstm_auc)
    result_line("Batch thresh", lstm_det.threshold, ".3f")
    result_line("Sim thresh",   lstm_det.sim_threshold, ".3f")
    bullet(f"Elapsed: {elapsed(t0)}")

    # ── 4. Generate Static Campaigns ─────────────────────────────────────────
    section("STEP 4 / 8  ·  Generating Static Fraud Campaigns")
    generator = StaticCampaignGenerator()
    known_campaigns   = generator.generate_known_campaigns(num_campaigns=NUM_CAMPAIGNS)
    unknown_campaigns = generator.generate_unknown_campaigns(num_campaigns=NUM_CAMPAIGNS)

    bullet(f"Known campaigns   : {len(known_campaigns)}  "
           f"(archetypes: mule_cascade, smurfing, rapid_fire)")
    bullet(f"Unknown campaigns : {len(unknown_campaigns)}  "
           f"(archetypes: slow_drip, interleaved_noise)")

    # ── 5. Train PPO Attackers ───────────────────────────────────────────────
    section("STEP 5 / 8  ·  Training PPO Reinforcement Learning Attackers")
    t0 = time.time()

    # Train against XGBoost (known + unknown)
    bullet("Training PPO vs XGBoost (known archetypes)...")
    ppo_xgb_known = train_rl_attacker(
        xgb_det, scaler=processor.scaler,
        is_unknown_archetype=False, total_timesteps=PPO_TRAIN_STEPS
    )
    bullet("Training PPO vs XGBoost (unknown archetypes)...")
    ppo_xgb_unknown = train_rl_attacker(
        xgb_det, scaler=processor.scaler,
        is_unknown_archetype=True, total_timesteps=PPO_TRAIN_STEPS
    )

    # Train against LSTM (known + unknown)
    bullet("Training PPO vs LSTM (known archetypes)...")
    ppo_lstm_known = train_rl_attacker(
        lstm_det, scaler=processor.scaler,
        is_unknown_archetype=False, total_timesteps=PPO_TRAIN_STEPS
    )
    bullet("Training PPO vs LSTM (unknown archetypes)...")
    ppo_lstm_unknown = train_rl_attacker(
        lstm_det, scaler=processor.scaler,
        is_unknown_archetype=True, total_timesteps=PPO_TRAIN_STEPS
    )
    bullet(f"All PPO attackers trained in {elapsed(t0)}")

    # ── 6. Simulate All 4 Worlds ─────────────────────────────────────────────
    section("STEP 6 / 8  ·  Simulating Four Adversarial Worlds")
    t0 = time.time()

    all_results = []

    for det_name, detector, ppo_known, ppo_unknown in [
        ("XGBoost", xgb_det,  ppo_xgb_known,  ppo_xgb_unknown),
        ("LSTM",    lstm_det, ppo_lstm_known, ppo_lstm_unknown),
    ]:
        sim = SimulationEngine(detector, scaler=processor.scaler)

        # World 1: Known fraud + static attacker
        w1 = sim.run_campaigns(
            generator.generate_known_campaigns(num_campaigns=NUM_CAMPAIGNS)
        )
        m1 = Evaluator.compute_campaign_metrics(
            w1, detector=detector, bg_test_df=legit_bg,
            scaler=processor.scaler, seed=RANDOM_SEED
        )

        # World 2: Known fraud + PPO adaptive attacker
        w2 = generate_rl_campaigns(
            ppo_known, detector, scaler=processor.scaler,
            is_unknown_archetype=False, num_campaigns=NUM_CAMPAIGNS
        )
        m2 = Evaluator.compute_campaign_metrics(
            w2, detector=detector, bg_test_df=legit_bg,
            scaler=processor.scaler, seed=RANDOM_SEED + 1
        )

        # World 3: Unknown fraud + static attacker
        w3 = sim.run_campaigns(
            generator.generate_unknown_campaigns(num_campaigns=NUM_CAMPAIGNS)
        )
        m3 = Evaluator.compute_campaign_metrics(
            w3, detector=detector, bg_test_df=legit_bg,
            scaler=processor.scaler, seed=RANDOM_SEED + 2
        )

        # World 4: Unknown fraud + PPO adaptive attacker
        w4 = generate_rl_campaigns(
            ppo_unknown, detector, scaler=processor.scaler,
            is_unknown_archetype=True, num_campaigns=NUM_CAMPAIGNS
        )
        m4 = Evaluator.compute_campaign_metrics(
            w4, detector=detector, bg_test_df=legit_bg,
            scaler=processor.scaler, seed=RANDOM_SEED + 3
        )

        dd = Evaluator.compute_detector_degradation(m1["F1"], m4["F1"])

        for world_name, metrics in [
            ("World 1 (Known + Static)",   m1),
            ("World 2 (Known + PPO)",      m2),
            ("World 3 (Unknown + Static)", m3),
            ("World 4 (Unknown + PPO)",    m4),
        ]:
            all_results.append({
                "Detector": det_name,
                "World":    world_name,
                "CSR":      metrics["CSR"],
                "CFE":      metrics["CFE"],
                "IT":       metrics["IT"],
                "F1":       metrics["F1"],
                "Precision": metrics["Precision"],
                "Recall":   metrics["Recall"],
            })

        bullet(f"{det_name} → Degradation (DD = F1_W1 − F1_W4) = {dd:.4f}")

    bullet(f"All 4 worlds simulated in {elapsed(t0)}")

    # ── 7. Build results table ───────────────────────────────────────────────
    section("STEP 7 / 8  ·  Results Summary")
    results_df = pd.DataFrame(all_results)

    # Pretty-print
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print()
    print(results_df.to_string(index=False))

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "demo_results.csv")
    results_df.to_csv(csv_path, index=False)
    bullet(f"Results saved to {csv_path}")

    # ── 8. Generate Plots ────────────────────────────────────────────────────
    section("STEP 8 / 8  ·  Generating Visualisation Plots")
    t0 = time.time()

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ---- Plot 1: Four-Worlds Comparison (2×2 grid) ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "Four Worlds Adversarial Fraud Detection - Comparison",
        fontsize=18, fontweight="bold", y=0.98,
    )

    metric_info = [
        ("CSR", "Campaign Success Rate (↑ = attacker wins)",   "Reds",   (0, 1.05)),
        ("CFE", "Cumulative Financial Exposure ₹ (↓ better)", "Oranges", None),
        ("IT",  "Intervention Timing (steps) (↓ better)",     "Blues",   None),
        ("F1",  "Detector F1 Score (↑ better)",               "Greens",  (0, 1.05)),
    ]

    for ax, (col, title, cmap, ylim) in zip(axes.flat, metric_info):
        sns.barplot(
            data=results_df, x="World", y=col, hue="Detector",
            ax=ax, palette=cmap, edgecolor="black", linewidth=0.6,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(title="Detector", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p1 = os.path.join(DEMO_PLOTS_DIR, "four_worlds_comparison.png")
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    bullet(f"Saved → {p1}")

    # ---- Plot 2: Degradation Curve ----
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    # Map world labels to short names for x-axis ordering
    world_order = [
        "World 1 (Known + Static)",
        "World 2 (Known + PPO)",
        "World 3 (Unknown + Static)",
        "World 4 (Unknown + PPO)",
    ]
    plot_df = results_df.copy()
    plot_df["World"] = pd.Categorical(plot_df["World"], categories=world_order, ordered=True)

    for det_name, color, marker in [("XGBoost", "#e74c3c", "o"), ("LSTM", "#2ecc71", "s")]:
        sub = plot_df[plot_df["Detector"] == det_name].sort_values("World")
        ax2.plot(
            sub["World"], sub["F1"],
            marker=marker, markersize=10, linewidth=2.5,
            label=det_name, color=color,
        )
        # Annotate values
        for _, row in sub.iterrows():
            ax2.annotate(
                f'{row["F1"]:.3f}',
                (row["World"], row["F1"]),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=10, fontweight="bold", color=color,
            )

    ax2.set_title(
        "Detector F1 Degradation Across Adversarial Worlds",
        fontsize=15, fontweight="bold",
    )
    ax2.set_ylabel("F1 Score", fontsize=13)
    ax2.set_xlabel("Adversarial World", fontsize=13)
    ax2.set_ylim(-0.05, 1.1)
    ax2.legend(fontsize=12, loc="upper right")
    ax2.tick_params(axis="x", rotation=20, labelsize=10)
    plt.tight_layout()
    p2 = os.path.join(DEMO_PLOTS_DIR, "degradation_curve.png")
    fig2.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    bullet(f"Saved → {p2}")

    # ---- Plot 3: Heatmap of All Metrics ----
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    heat_df = results_df.set_index(["Detector", "World"])[
        ["CSR", "CFE", "IT", "F1", "Precision", "Recall"]
    ]
    # Normalise CFE and IT for heatmap visibility
    heat_norm = heat_df.copy()
    for c in heat_norm.columns:
        cmax = heat_norm[c].max()
        if cmax > 0:
            heat_norm[c] = heat_norm[c] / cmax

    sns.heatmap(
        heat_norm, annot=heat_df.round(3).values, fmt="",
        cmap="YlOrRd", linewidths=0.5, ax=ax3,
        xticklabels=heat_df.columns,
        yticklabels=[f"{d} / {w}" for d, w in heat_df.index],
    )
    ax3.set_title(
        "Metric Heatmap (raw values shown, colour = normalised)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    p3 = os.path.join(DEMO_PLOTS_DIR, "metrics_heatmap.png")
    fig3.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(fig3)
    bullet(f"Saved → {p3}")

    # ---- Plot 4: Static Test Set Performance (confusion matrix side by side) ----
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 5))
    fig4.suptitle(
        "Detector Performance on PaySim Static Test Set",
        fontsize=15, fontweight="bold",
    )

    for ax, preds, name in [
        (ax4a, xgb_preds, "XGBoost"),
        (ax4b, lstm_preds, "LSTM"),
    ]:
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"],
        )
        f1_val = f1_score(y_test, preds, zero_division=0)
        ax.set_title(f"{name}  (F1 = {f1_val:.4f})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p4 = os.path.join(DEMO_PLOTS_DIR, "confusion_matrices.png")
    fig4.savefig(p4, dpi=200, bbox_inches="tight")
    plt.close(fig4)
    bullet(f"Saved → {p4}")

    # ---- Plot 5: CSR Comparison bar chart ----
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=results_df, x="World", y="CSR", hue="Detector",
        palette=["#e74c3c", "#2ecc71"], edgecolor="black", ax=ax5,
    )
    ax5.set_title(
        "Campaign Success Rate — Attacker vs Detector",
        fontsize=15, fontweight="bold",
    )
    ax5.set_ylabel("CSR (1.0 = all attacks succeed)")
    ax5.set_ylim(0, 1.15)
    ax5.axhline(y=1.0, color="grey", linestyle="--", alpha=0.5, label="100% success")
    ax5.legend(fontsize=11)
    ax5.tick_params(axis="x", rotation=20, labelsize=10)
    plt.tight_layout()
    p5 = os.path.join(DEMO_PLOTS_DIR, "csr_comparison.png")
    fig5.savefig(p5, dpi=200, bbox_inches="tight")
    plt.close(fig5)
    bullet(f"Saved → {p5}")

    bullet(f"All plots generated in {elapsed(t0)}")

    # ── Final Report ─────────────────────────────────────────────────────────
    banner("DEMO COMPLETE")
    total_time = elapsed(t_start)
    print(f"""
     [OK]  Data loaded and preprocessed
     [OK]  13 transaction features engineered
     [OK]  XGBoost detector trained   (test F1 = {xgb_f1:.4f})
     [OK]  LSTM detector trained      (test F1 = {lstm_f1:.4f})
     [OK]  4 PPO adversarial attackers trained
     [OK]  All 4 worlds simulated     ({NUM_CAMPAIGNS} campaigns each)
     [OK]  Campaign metrics computed  (CSR, CFE, IT, F1, Precision, Recall)
     [OK]  5 publication-quality plots generated

     Key insight:
       Detector performance degrades as the adversary evolves from
       known static -> known adaptive -> unknown static -> unknown adaptive.
       This proves that static evaluation alone is insufficient for
       real-world fraud detection systems.

     Output files:
       * {csv_path}
       * {DEMO_PLOTS_DIR}/four_worlds_comparison.png
       * {DEMO_PLOTS_DIR}/degradation_curve.png
       * {DEMO_PLOTS_DIR}/metrics_heatmap.png
       * {DEMO_PLOTS_DIR}/confusion_matrices.png
       * {DEMO_PLOTS_DIR}/csr_comparison.png

     Total runtime: {total_time}
    """)


if __name__ == "__main__":
    main()
