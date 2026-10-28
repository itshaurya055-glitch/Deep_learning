"""
run_lab_v2.py  -  Four Worlds lab, revised to close the research gaps.

Changes vs run_lab.py (search for the tags [C1]..[C8]):
 [C1] Multi-seed runs, mean +/- std reported            (statistical validity)
 [C2] Threshold calibration on a validation slice, not the test set (no leakage)
 [C3] ONE fixed legit background for all worlds          (DD no longer = sampling noise)
 [C4] Random-action attacker baseline                    (proves PPO is truly "adaptive")
 [C5] Longer PPO training                                (PPO_STEPS)
 [C6] Detector Degradation computed for every world      (not only World 4)
 [C7] Hardened detector = defence (adversarial retraining) -> Gap 4 / co-evolution papers
 [C8] "Known" really means seen in training (hardened detector sees known archetypes only;
      unknown archetypes stay unseen, so W3/W4 recovery = true generalisation)

ADAPT-ME: places that depend on files I could not see are marked  # ADAPT.
"""
import os
import time
import random
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.utils import set_random_seed
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.data_processor import DataProcessor
from src.models.xgboost_detector import XGBoostDetector
from src.models.lstm_detector import LSTMDetector
from src.campaigns.static_generator import StaticCampaignGenerator
from src.campaigns.rl_attacker import train_rl_attacker, generate_rl_campaigns
from src.simulation.simulator import SimulationEngine
from src.evaluation.metrics import Evaluator
from src.evaluation.visualizer import Visualizer
from src.config import OUTPUT_DIR, MODELS_DIR

# ---------------- experiment settings ----------------
SEEDS = [0, 1, 2]          # [C1] raise to 5 if time allows
N_CAMPAIGNS = 100          # was 50
PPO_STEPS = 50_000         # [C5] was 5000 (reduce to 20_000 if too slow)
BG_SEED = 123              # [C3] same background sample in every world
VAL_FRAC = 0.15            # [C2] last 15% of train (temporal) used for calibration
N_ACTIONS = 9              # 3 amounts x 3 delays  (rl_attacker.py)
DETECTORS = ["XGBoost", "LSTM"]
METRICS = ["CSR", "CFE", "IT", "F1", "Precision", "Recall"]


def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    set_random_seed(s)


# [C4] random attacker: same interface as an SB3 model's .predict()
class RandomAttacker:
    def __init__(self, n_actions=N_ACTIONS, seed=0):
        self.rng = np.random.default_rng(seed)
        self.n = n_actions

    def predict(self, obs, deterministic=True, **kwargs):
        return np.array(int(self.rng.integers(self.n))), None


# ---------------- small helpers ----------------
def _slice(a, start, stop):
    return a.iloc[start:stop] if hasattr(a, "iloc") else a[start:stop]


def _stack(a, b):
    if hasattr(a, "iloc"):
        return pd.concat([a, b], ignore_index=True)
    return np.vstack([a, b])


def _stack_y(a, b):
    if hasattr(a, "iloc"):
        return pd.concat([a, b], ignore_index=True)
    return np.concatenate([a, b])


def temporal_val_split(X, y):
    """[C2] assumes X/y are time-ordered (your split is temporal)."""
    cut = int(len(X) * (1 - VAL_FRAC))
    return _slice(X, 0, cut), _slice(y, 0, cut), _slice(X, cut, None), _slice(y, cut, None)


def build_detector(name, X_fit, y_fit, X_val, y_val):
    if name == "XGBoost":
        d = XGBoostDetector()
        d.fit(X_fit, y_fit)   # ADAPT: make sure its internal threshold calibration
                              # uses a slice of X_fit / X_val, NOT the test set.
    else:
        d = LSTMDetector()
        d.fit(X_fit, y_fit)
        d.calibrate_simulation_threshold(X_val, y_val)  # [C2] validation, not test
    return d


def campaign_rows(campaigns, sim_engine):
    """
    [C7] Turn executed campaign steps into (X_rows, y_rows) with the SAME columns/scaling
    as X_train, label = 1 (fraud).

    ADAPT: reuse the code in src/simulation/simulator.py that converts a TransactionStep
    into a feature vector (it already does this for every step), but return the rows
    instead of sending them to the detector. Also add rows for steps that were blocked.
    """
    raise NotImplementedError("Implement using the feature-building code from simulator.py")


# ---------------- one full 4-world evaluation of ONE detector ----------------
def run_worlds(det, det_name, variant, scaler, bg, static_gen, seed):
    rows, camps = [], {}
    sim = SimulationEngine(det, scaler=scaler)

    def score(campaigns, world):
        m = Evaluator.compute_campaign_metrics(
            campaigns, detector=det, bg_test_df=bg, scaler=scaler, seed=BG_SEED)  # [C3]
        rows.append({"Seed": seed, "Detector": det_name, "Variant": variant,
                     "World": world, **{k: m[k] for k in METRICS}})

    # World 1: known + static
    c1 = sim.run_campaigns(static_gen.generate_known_campaigns(num_campaigns=N_CAMPAIGNS))
    score(c1, "World 1")

    # World 2: known + PPO
    a2 = train_rl_attacker(det, scaler=scaler, is_unknown_archetype=False, total_timesteps=PPO_STEPS)
    a2.save(os.path.join(MODELS_DIR, f"ppo_{det_name.lower()}_{variant.lower()}_w2_s{seed}.zip"))
    c2 = generate_rl_campaigns(a2, det, scaler=scaler, is_unknown_archetype=False, num_campaigns=N_CAMPAIGNS)
    score(c2, "World 2")
    camps["W2"] = c2

    # World 3: unknown + static
    c3 = sim.run_campaigns(static_gen.generate_unknown_campaigns(num_campaigns=N_CAMPAIGNS))
    score(c3, "World 3")

    # World 4: unknown + PPO
    a4 = train_rl_attacker(det, scaler=scaler, is_unknown_archetype=True, total_timesteps=PPO_STEPS)
    a4.save(os.path.join(MODELS_DIR, f"ppo_{det_name.lower()}_{variant.lower()}_w4_s{seed}.zip"))
    c4 = generate_rl_campaigns(a4, det, scaler=scaler, is_unknown_archetype=True, num_campaigns=N_CAMPAIGNS)
    score(c4, "World 4")

    # [C4] random-attacker controls: PPO should beat these, otherwise "adaptive" is unproven
    for unk, wname in [(False, "World 2 (random)"), (True, "World 4 (random)")]:
        cr = generate_rl_campaigns(RandomAttacker(seed=seed), det, scaler=scaler,
                                   is_unknown_archetype=unk, num_campaigns=N_CAMPAIGNS)
        score(cr, wname)

    return rows, camps, sim


def main():
    t0 = time.time()
    print("=" * 80)
    print("FOUR WORLDS LAB v2 (multi-seed, leakage-free, with defence)")
    print("=" * 80)

    processor = DataProcessor()
    feat_df = processor.engineer_features(processor.load_and_sample())
    (X_train, y_train, _), (X_test, y_test, _) = processor.prepare_world1_split(feat_df)
    bg = X_test[y_test == 0].copy()
    scaler = processor.scaler
    static_gen = StaticCampaignGenerator()

    all_rows, clean_rows = [], []

    for seed in SEEDS:
        print(f"\n######## SEED {seed} ########")
        seed_everything(seed)
        X_fit, y_fit, X_val, y_val = temporal_val_split(X_train, y_train)

        for name in DETECTORS:
            print(f"\n--- {name}: baseline (trained on PaySim only) ---")
            base = build_detector(name, X_fit, y_fit, X_val, y_val)

            # clean PaySim test result (threshold was NOT tuned on this set)
            p, yhat = base.predict_proba(X_test), base.predict(X_test)
            clean_rows.append({"Seed": seed, "Detector": name,
                               "F1": f1_score(y_test, yhat), "Precision": precision_score(y_test, yhat),
                               "Recall": recall_score(y_test, yhat), "AUC": roc_auc_score(y_test, p)})

            rows, camps, sim = run_worlds(base, name, "Baseline", scaler, bg, static_gen, seed)
            all_rows += rows

            # [C7][C8] hardened detector: PaySim + KNOWN-archetype campaigns only
            print(f"--- {name}: hardened (adversarial retraining on known archetypes) ---")
            try:
                seed_everything(seed + 1000)  # fresh campaigns, disjoint from test campaigns
                known_train = sim.run_campaigns(static_gen.generate_known_campaigns(num_campaigns=N_CAMPAIGNS))
                Xa, ya = campaign_rows(known_train + camps["W2"], sim)  # static-known + PPO-known
                X_aug, y_aug = _stack(X_fit, Xa), _stack_y(y_fit, ya)
                seed_everything(seed)
                hard = build_detector(name, X_aug, y_aug, X_val, y_val)
                rows, _, _ = run_worlds(hard, name, "Hardened", scaler, bg, static_gen, seed)
                all_rows += rows   # PPO is RE-TRAINED against the hardened detector (fair)
            except NotImplementedError as e:
                print(f"[skip hardened] {e}")

    # ---------------- aggregate ----------------
    df = pd.DataFrame(all_rows)

    # [C6] degradation of every world relative to World 1, same seed/detector/variant
    w1 = df[df.World == "World 1"][["Seed", "Detector", "Variant", "F1"]].rename(columns={"F1": "F1_W1"})
    df = df.merge(w1, on=["Seed", "Detector", "Variant"])
    df["Detector_Degradation"] = df["F1_W1"] - df["F1"]
    df.drop(columns="F1_W1", inplace=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "four_worlds_raw_by_seed.csv"), index=False)
    pd.DataFrame(clean_rows).to_csv(os.path.join(OUTPUT_DIR, "clean_test_by_seed.csv"), index=False)

    cols = METRICS + ["Detector_Degradation"]
    summ = df.groupby(["Detector", "Variant", "World"])[cols].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(os.path.join(OUTPUT_DIR, "four_worlds_summary_mean_std.csv"), index=False)

    # plots use the baseline, 4 main worlds, seed-averaged
    plot_df = (df[(df.Variant == "Baseline") & df.World.isin(["World 1", "World 2", "World 3", "World 4"])]
               .groupby(["Detector", "World"])[cols].mean().reset_index())
    viz = Visualizer()
    viz.plot_four_worlds_comparison(plot_df)
    viz.plot_degradation_curves(plot_df)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 80)
    print("SUMMARY (mean over seeds)")
    print("=" * 80)
    print(df.groupby(["Detector", "Variant", "World"])[["CSR", "CFE", "IT", "F1", "Detector_Degradation"]]
          .mean().round(3).to_string())
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()