import pandas as pd
from typing import List, Any
from src.campaigns.schema import Campaign
from src.campaigns.static_generator import step_to_feature_dataframe

class SimulationEngine:
    def __init__(self, detector: Any, scaler=None):
        self.detector = detector
        self.scaler = scaler

    def run_campaigns(self, campaigns: List[Campaign]) -> List[Campaign]:
        """
        Runs static campaigns through the detector step-by-step.
        """
        if hasattr(self.detector, "reset_history"):
            self.detector.reset_history()

        evaluated_campaigns = []
        for c in campaigns:
            c_eval = Campaign(
                campaign_id=c.campaign_id,
                archetype=c.archetype,
                target_exfiltration=c.target_exfiltration,
                steps=[]
            )

            for step_idx, step in enumerate(c.steps):
                c_eval.add_step(step)
                feat_df = step_to_feature_dataframe(step, self.scaler)
                
                if hasattr(self.detector, "predict_step"):
                    prob, is_blocked = self.detector.predict_step(feat_df, step.payer_account)
                else:
                    prob = self.detector.predict_proba(feat_df)[0]
                    is_blocked = int(prob >= getattr(self.detector, "threshold", 0.5))

                blocked = bool(is_blocked == 1)
                c_eval.record_decision(step_idx, step.amount, blocked)

                if blocked:
                    break  # Campaign terminated upon first correct block

            evaluated_campaigns.append(c_eval)
        return evaluated_campaigns
