import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from src.campaigns.schema import Campaign
from src.campaigns.static_generator import step_to_feature_dataframe

class Evaluator:
    @staticmethod
    def compute_campaign_metrics(
        campaigns: List[Campaign],
        detector: Any = None,
        bg_test_df: pd.DataFrame = None,
        scaler=None,
        seed: int = 42
    ) -> Dict[str, float]:
        if not campaigns:
            return {"CSR": 0.0, "IT": 0.0, "CFE": 0.0, "F1": 0.0,
                    "Precision": 0.0, "Recall": 0.0, "AUC": 0.0}

        total_campaigns = len(campaigns)
        successful_campaigns = sum(1 for c in campaigns if c.status == "SUCCESS")
        csr = float(successful_campaigns / total_campaigns)

        # Cumulative Financial Exposure (CFE)
        cfe = float(np.mean([c.cumulative_exfiltrated for c in campaigns]))

        # Intervention Timing: 1-indexed step of first block,
        # or total executed steps if never blocked
        intervention_steps = []
        for c in campaigns:
            if c.first_blocked_step is not None:
                intervention_steps.append(c.first_blocked_step + 1)
            else:
                intervention_steps.append(max(1, c.executed_steps))
        it = float(np.mean(intervention_steps))

        # ----------------------------------------------------------------
        # Pooled Transaction-level F1 — correct design:
        #   FRAUD class  = ALL steps actually executed in campaigns (both
        #                  allowed AND blocked steps; they are all fraud
        #                  attempts that the detector must identify).
        #   LEGIT class  = random sample from held-out legitimate test rows.
        #   Seed varies per call (passed in) so identical data cannot produce
        #   bit-for-bit identical F1 across worlds with different campaign sets.
        # ----------------------------------------------------------------
        if detector is not None and bg_test_df is not None:
            campaign_feats = []
            for c in campaigns:
                for idx, step in enumerate(c.steps):
                    # Include ALL steps that were presented to the detector,
                    # including the final blocked step — that's a true
                    # fraud attempt that ideally should be flagged.
                    if idx < c.executed_steps:
                        feat_df = step_to_feature_dataframe(step, scaler)
                        campaign_feats.append(feat_df)

            if campaign_feats:
                fraud_X = pd.concat(campaign_feats, ignore_index=True)
                fraud_y = np.ones(len(fraud_X), dtype=int)

                # Use a world-specific seed so each world samples a different
                # legitimate background slice → different (not identical) F1
                bg_n = min(len(bg_test_df), max(500, len(fraud_X) * 5))
                bg_sample = bg_test_df.sample(n=bg_n, random_state=seed)
                legit_X = bg_sample[fraud_X.columns].copy()
                legit_y = np.zeros(len(legit_X), dtype=int)

                pooled_X = pd.concat([fraud_X, legit_X], ignore_index=True)
                pooled_y = np.concatenate([fraud_y, legit_y])

                pooled_probs = detector.predict_proba(pooled_X)
                pooled_preds = detector.predict(pooled_X)

                f1   = float(f1_score(pooled_y, pooled_preds, zero_division=0))
                prec = float(precision_score(pooled_y, pooled_preds, zero_division=0))
                rec  = float(recall_score(pooled_y, pooled_preds, zero_division=0))
                # Guard against single-class pooled_y edge case
                if len(np.unique(pooled_y)) > 1:
                    auc = float(roc_auc_score(pooled_y, pooled_probs))
                else:
                    auc = 0.0
            else:
                f1, prec, rec, auc = 0.0, 0.0, 0.0, 0.0
        else:
            f1, prec, rec, auc = 0.0, 0.0, 0.0, 0.0

        return {
            "CSR": csr,
            "CFE": cfe,
            "IT": it,
            "F1": f1,
            "Precision": prec,
            "Recall": rec,
            "AUC": auc,
            "total_campaigns": total_campaigns,
            "successful_campaigns": successful_campaigns
        }

    @staticmethod
    def compute_detector_degradation(world1_f1: float, world4_f1: float) -> float:
        """Detector Degradation (DD) = F1(World 1) − F1(World 4)."""
        return float(world1_f1 - world4_f1)
