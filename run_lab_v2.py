import argparse
import os
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.campaigns.rl_attacker import FraudAttackerEnv, train_rl_attacker, generate_rl_campaigns
from src.campaigns.schema import Campaign, TransactionStep
from src.campaigns.static_generator import StaticCampaignGenerator, step_to_feature_dataframe
from src.config import FEATURE_COLUMNS, MODELS_DIR, OUTPUT_DIR
from src.data_processor import DataProcessor
from src.evaluation.metrics import Evaluator
from src.evaluation.visualizer import Visualizer
from src.models.lstm_detector import LSTMDetector
from src.models.xgboost_detector import XGBoostDetector
from src.simulation.simulator import SimulationEngine

SEEDS = [0, 1, 2]
PPO_STEPS = 50000
BG_SEED = 42
N_ACTIONS = 9


def parse_args():
    parser = argparse.ArgumentParser(description="Four Worlds adversarial UPI fraud lab v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="Seeds to run. Default: 0 1 2")
    parser.add_argument("--ppo-steps", type=int, default=PPO_STEPS, help="PPO timesteps for the RL attacker")
    parser.add_argument("--num-campaigns", type=int, default=50, help="Number of campaigns per world")
    parser.add_argument("--no-hardening", action="store_true", help="Skip the hardened retraining stage")
    return parser.parse_args()


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean_val = float(np.mean(values))
    if len(values) == 1:
        return mean_val, 0.0
    std_val = float(np.std(values, ddof=1))
    return mean_val, std_val


def campaign_rows(campaigns: List[Campaign], scaler=None) -> Tuple[pd.DataFrame, np.ndarray]:
    """Convert campaign steps into feature rows matching the detector feature schema.

    Some synthetic campaigns are used before evaluation, so we allow two cases:
    - evaluated campaigns where executed_steps indicates how many steps were actually used
    - raw campaigns where no evaluation has yet happened; in that case use all steps
    """
    rows = []
    labels = []
    for campaign in campaigns:
        used_steps = max(1, int(getattr(campaign, "executed_steps", 0))) if getattr(campaign, "executed_steps", 0) > 0 else len(campaign.steps)
        for idx, step in enumerate(campaign.steps[:used_steps]):
            feat_df = step_to_feature_dataframe(step, scaler=scaler)
            rows.append(feat_df.iloc[0].copy())
            labels.append(1)

    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS), np.array([], dtype=int)

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    y = np.array(labels, dtype=int)
    return X, y


def random_action_attacker(detector, scaler=None, is_unknown_archetype: bool = False, num_campaigns: int = 50):
    """[C4] Random-action control attacker to make the PPO comparison meaningful."""
    env = FraudAttackerEnv(detector, scaler=scaler, is_unknown_archetype=is_unknown_archetype)
    campaigns = []
    for i in range(num_campaigns):
        obs, _ = env.reset(seed=i)
        done = False
        c = Campaign(campaign_id=f"random_{'unk' if is_unknown_archetype else 'kn'}_{i}", archetype="random_control", target_exfiltration=env.exfiltration_target)
        step_idx = 0
        while not done:
            action = int(np.random.randint(0, N_ACTIONS))
            amount, delay = env.action_map[action]
            obs, reward, done, truncated, info = env.step(action)
            blocked = bool(info.get("blocked", False))
            step_obj = TransactionStep(
                step_idx=step_idx,
                amount=amount,
                timing_delay=delay,
                beneficiary=env.beneficiaries[-1],
                payer_account=env.payer_id,
                action_type="TRANSFER",
                is_balance_drain=1 if env.accumulated_amount >= env.exfiltration_target else 0,
                is_new_beneficiary=1,
            )
            c.add_step(step_obj)
            c.record_decision(step_idx, amount, blocked)
            step_idx += 1
        campaigns.append(c)
    return campaigns


def build_baseline_detectors(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray):
    """Train baseline detectors and calibrate the LSTM on a validation slice, not on X_test."""
    rng = np.random.default_rng(42)
    n_val = max(200, int(0.15 * len(X_train)))
    val_idx = rng.choice(len(X_train), size=min(n_val, len(X_train)), replace=False)
    train_mask = np.ones(len(X_train), dtype=bool)
    train_mask[val_idx] = False

    X_fit = X_train.iloc[train_mask].copy()
    y_fit = y_train[train_mask]
    X_val = X_train.iloc[val_idx].copy()
    y_val = y_train[val_idx]

    xgb_detector = XGBoostDetector()
    xgb_detector.fit(X_fit, y_fit)

    lstm_detector = LSTMDetector()
    lstm_detector.fit(X_fit, y_fit)
    # [C2] Calibration is performed on train validation slice to avoid test leakage.
    lstm_detector.calibrate_simulation_threshold(X_val, y_val)

    return xgb_detector, lstm_detector


def evaluate_static_baseline(detector, X_test: pd.DataFrame, y_test: np.ndarray):
    probs = detector.predict_proba(X_test)
    preds = detector.predict(X_test)
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    return {
        "F1": float(f1_score(y_test, preds, zero_division=0)),
        "Precision": float(precision_score(y_test, preds, zero_division=0)),
        "Recall": float(recall_score(y_test, preds, zero_division=0)),
        "AUC": float(roc_auc_score(y_test, probs)),
    }


def hardened_retraining(X_train: pd.DataFrame, y_train: np.ndarray, campaigns: List[Campaign], scaler=None, detector_name: str = "XGBoost"):
    """[C7] Harden by training on real PaySim data plus known fraud campaign rows."""
    known_evaluated = []
    for c in campaigns:
        known_evaluated.append(c)

    X_known, y_known = campaign_rows(known_evaluated, scaler=scaler)
    if X_known.empty:
        warnings.warn("[C7] campaign_rows() returned empty data, hardening skipped.")
        return None

    X_hard = pd.concat([X_train, X_known], ignore_index=True)
    y_hard = np.concatenate([y_train, y_known])

    if detector_name == "XGBoost":
        hardened = XGBoostDetector()
        hardened.fit(X_hard, y_hard)
    else:
        hardened = LSTMDetector()
        hardened.fit(X_hard, y_hard)
        # Re-calibrate the simulation threshold on a validation slice from the hardened training data.
        val_idx = np.random.default_rng(42).choice(len(X_hard), size=min(max(200, int(0.15 * len(X_hard))), len(X_hard)), replace=False)
        X_val = X_hard.iloc[val_idx].copy()
        y_val = y_hard[val_idx]
        hardened.calibrate_simulation_threshold(X_val, y_val)

    return hardened


def summarize_seed_results(results_by_seed: List[Dict]) -> Dict[str, List[float]]:
    summary = {}
    keys = ["CSR", "CFE", "IT", "F1", "Precision", "Recall", "DD"]
    for key in keys:
        values = [float(r.get(key, 0.0)) for r in results_by_seed]
        mean_val, std_val = mean_std(values)
        summary[key] = {"mean": mean_val, "std": std_val}
    return summary


def parity_check(detector, X_test: pd.DataFrame, y_test: np.ndarray, scaler=None, n_rows: int = 200):
    """Compare detector.predict on real fraud rows with the simulation engine step-blocking decision."""
    fraud_rows = X_test[y_test == 1].copy()
    if len(fraud_rows) == 0:
        print("[Parity] No fraud rows available for parity check.")
        return
    idxs = np.random.default_rng(42).choice(len(fraud_rows), size=min(n_rows, len(fraud_rows)), replace=False)
    sample = fraud_rows.iloc[idxs].reset_index(drop=True)
    det_preds = detector.predict(sample).astype(int)
    sim_engine = SimulationEngine(detector, scaler=scaler)
    sim_flags = []
    for i, row in sample.iterrows():
        step = TransactionStep(
            step_idx=0,
            amount=float(row["amount"]),
            timing_delay=1.0,
            beneficiary="b",
            payer_account=f"parity_{i}",
            action_type="TRANSFER" if int(row.get("type_TRANSFER", 0)) == 1 else "CASH_OUT",
            is_balance_drain=int(row.get("is_balance_drain", 0)),
            is_new_beneficiary=int(row.get("is_new_beneficiary", 0)),
        )
        camp = Campaign(campaign_id=f"parity_{i}", archetype="test", target_exfiltration=50000.0, steps=[step])
        eval_camp = sim_engine.run_campaigns([camp])[0]
        sim_flags.append(int(eval_camp.status == "BLOCKED"))

    det_mean = float(det_preds.mean())
    sim_mean = float(np.mean(sim_flags))
    print(f"[Parity] {detector.__class__.__name__}: detector flag rate={det_mean:.4f}, simulation block rate={sim_mean:.4f}, n={len(sample)}")


def run_single_seed(seed: int, detector_name: str, base_detector, scaler, legit_test_X: pd.DataFrame, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray, num_campaigns: int, ppo_steps: int, do_hardening: bool):
    static_gen = StaticCampaignGenerator()
    sim_engine = SimulationEngine(base_detector, scaler=scaler)

    world_records = []
    world_metrics = []

    # [C6] DD is computed for all worlds, not just World 4.
    for world_name, campaigns in [
        ("World 1", static_gen.generate_known_campaigns(num_campaigns=num_campaigns)),
        ("World 2", None),
        ("World 3", static_gen.generate_unknown_campaigns(num_campaigns=num_campaigns)),
        ("World 4", None),
    ]:
        if world_name in {"World 2", "World 4"}:
            # [C4] Random-action attacker control is run alongside PPO to make adaptive behavior visible.
            adversarial_mode = "random_control"
            random_campaigns = random_action_attacker(base_detector, scaler=scaler, is_unknown_archetype=(world_name == "World 4"), num_campaigns=num_campaigns)
            ppo_attacker = train_rl_attacker(base_detector, scaler=scaler, is_unknown_archetype=(world_name == "World 4"), total_timesteps=ppo_steps)
            ppo_campaigns = generate_rl_campaigns(ppo_attacker, base_detector, scaler=scaler, is_unknown_archetype=(world_name == "World 4"), num_campaigns=num_campaigns)
            campaigns_to_score = random_campaigns + ppo_campaigns
            campaign_labels = ["random_control", "ppo"] * (num_campaigns // 2)
        else:
            evaluated = sim_engine.run_campaigns(campaigns)
            metrics = Evaluator.compute_campaign_metrics(evaluated, detector=base_detector, bg_test_df=legit_test_X, scaler=scaler, seed=seed)
            world_metrics.append({
                "Detector": detector_name,
                "World": world_name,
                "CSR": metrics["CSR"],
                "CFE": metrics["CFE"],
                "IT": metrics["IT"],
                "F1": metrics["F1"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "DD": 0.0,
                "attacker": "static",
            })
            continue

        for att_name, att_campaigns in [("random_control", random_campaigns), ("ppo", ppo_campaigns)]:
            evaluated = sim_engine.run_campaigns(att_campaigns)
            metrics = Evaluator.compute_campaign_metrics(evaluated, detector=base_detector, bg_test_df=legit_test_X, scaler=scaler, seed=seed)
            world_metrics.append({
                "Detector": detector_name,
                "World": world_name,
                "CSR": metrics["CSR"],
                "CFE": metrics["CFE"],
                "IT": metrics["IT"],
                "F1": metrics["F1"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "DD": 0.0,
                "attacker": att_name,
            })

    if do_hardening:
        known_campaigns = static_gen.generate_known_campaigns(num_campaigns=num_campaigns)
        hardened_model = hardened_retraining(X_train, y_train, known_campaigns, scaler=scaler, detector_name=detector_name)
        if hardened_model is not None:
            # [C8] Use known archetypes only for hardening; unseen archetypes remain untouched.
            for world_name in ["World 3", "World 4"]:
                if world_name == "World 3":
                    unseen_campaigns = static_gen.generate_unknown_campaigns(num_campaigns=num_campaigns)
                else:
                    ppo_attacker = train_rl_attacker(hardened_model, scaler=scaler, is_unknown_archetype=True, total_timesteps=ppo_steps)
                    unseen_campaigns = generate_rl_campaigns(ppo_attacker, hardened_model, scaler=scaler, is_unknown_archetype=True, num_campaigns=num_campaigns)
                evaluated = sim_engine.run_campaigns(unseen_campaigns)
                metrics = Evaluator.compute_campaign_metrics(evaluated, detector=hardened_model, bg_test_df=legit_test_X, scaler=scaler, seed=seed)
                world_metrics.append({
                    "Detector": f"{detector_name}_Hardened",
                    "World": world_name,
                    "CSR": metrics["CSR"],
                    "CFE": metrics["CFE"],
                    "IT": metrics["IT"],
                    "F1": metrics["F1"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "DD": 0.0,
                    "attacker": "hardened",
                })

    # [Parity] Always print a small parity check before summary, to surface presence/absence of feature mismatch.
    parity_check(base_detector, X_test, y_test, scaler=scaler, n_rows=50)

    # [C6] Compute degradation for each world relative to World 1, not just the final world.
    world1_f1 = next((r["F1"] for r in world_metrics if r["Detector"] == detector_name and r["World"] == "World 1" and r["attacker"] == "static"), 0.0)
    for r in world_metrics:
        if r["Detector"] == detector_name and r["World"] != "World 1":
            r["DD"] = float(world1_f1 - r["F1"])

    for r in world_metrics:
        if r["Detector"].endswith("_Hardened"):
            r["DD"] = float(r["F1"] - 0.0)

    return world_metrics


def main():
    args = parse_args()
    start_time = time.time()
    print("=" * 90)
    print("FOUR WORLDS ADVERSARIAL UPI FRAUD DETECTION TESTING LAB v2")
    print(f"SEEDS={args.seeds} | PPO_STEPS={args.ppo_steps} | NUM_CAMPAIGNS={args.num_campaigns}")
    print("=" * 90)

    processor = DataProcessor()
    raw_df = processor.load_and_sample()
    feat_df = processor.engineer_features(raw_df)
    (X_train, y_train, train_df), (X_test, y_test, test_df) = processor.prepare_world1_split(feat_df)
    legit_test_X = X_test[y_test == 0].copy()

    # [C3] Single fixed background seed keeps the legit background identical across worlds.
    legit_test_X = legit_test_X.sample(n=min(len(legit_test_X), max(500, len(legit_test_X))), random_state=BG_SEED).reset_index(drop=True)

    baseline_results = []
    detectors = {
        "XGBoost": XGBoostDetector(),
        "LSTM": LSTMDetector(),
    }

    for seed in args.seeds:
        np.random.seed(seed)
        for detector_name, detector_ctor in detectors.items():
            if detector_name == "XGBoost":
                detector = XGBoostDetector()
                detector.fit(X_train, y_train)
            else:
                detector = LSTMDetector()
                detector.fit(X_train, y_train)
                # [C2] validation slice is used to calibrate the sim threshold.
                rng = np.random.default_rng(42)
                val_size = min(max(200, int(0.15 * len(X_train))), len(X_train))
                val_idx = rng.choice(len(X_train), size=val_size, replace=False)
                X_val = X_train.iloc[val_idx].copy()
                y_val = y_train[val_idx]
                detector.calibrate_simulation_threshold(X_val, y_val)

            results = run_single_seed(
                seed=seed,
                detector_name=detector_name,
                base_detector=detector,
                scaler=processor.scaler,
                legit_test_X=legit_test_X,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                num_campaigns=args.num_campaigns,
                ppo_steps=args.ppo_steps,
                do_hardening=not args.no_hardening,
            )
            baseline_results.extend(results)

    results_df = pd.DataFrame(baseline_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "four_worlds_results_v2.csv"), index=False)

    print("\n=== Final Summary ===")
    print(results_df[["Detector", "World", "attacker", "CSR", "CFE", "IT", "F1", "DD"]].to_string(index=False))

    # [C1] Print mean ± std across seeds.
    if len(args.seeds) > 1:
        for detector_name in ["XGBoost", "LSTM"]:
            subset = results_df[results_df["Detector"] == detector_name].copy()
            print(f"\n[{detector_name}] Mean ± Std by world:")
            for world_name in ["World 1", "World 2", "World 3", "World 4"]:
                world_rows = subset[subset["World"] == world_name]
                if world_rows.empty:
                    continue
                f1_values = world_rows["F1"].tolist()
                mean_f1, std_f1 = mean_std(f1_values)
                print(f"  {world_name}: F1 = {mean_f1:.4f} ± {std_f1:.4f}")

    if not args.no_hardening:
        print("\n[C7] Hardened retraining is enabled; the script evaluates baseline vs hardened on unseen attack worlds.")
        print("[C8] Hardening uses Known archetypes only; Unknown archetypes remain untouched for the final defence test.")
    else:
        print("\n[C7] Hardened retraining is disabled by flag --no-hardening.")

    elapsed = time.time() - start_time
    print(f"\nExecution finished in {elapsed/60:.2f} minutes.")


if __name__ == "__main__":
    main()
