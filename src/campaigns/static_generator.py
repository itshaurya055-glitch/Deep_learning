import numpy as np
import pandas as pd
from typing import List
from src.campaigns.schema import Campaign, TransactionStep
from src.config import CAMPAIGN_EXFILTRATION_TARGET, MAX_CAMPAIGN_STEPS, FEATURE_COLUMNS

def step_to_feature_dataframe(step: TransactionStep, scaler=None) -> pd.DataFrame:
    """
    Converts a TransactionStep into a single-row DataFrame matching FEATURE_COLUMNS.
    """
    row_dict = {
        "amount": step.amount,
        "oldbalanceOrg": 50000.0,
        "newbalanceOrig": 0.0 if step.is_balance_drain else 50000.0 - step.amount,
        "oldbalanceDest": 10000.0,
        "newbalanceDest": 10000.0 + step.amount,
        "type_TRANSFER": 1 if step.action_type == "TRANSFER" else 0,
        "type_CASH_OUT": 1 if step.action_type == "CASH_OUT" else 0,
        "tx_count_last_N": min(5, step.step_idx + 1),
        "sum_amount_last_N": step.amount * min(5, step.step_idx + 1),
        "distinct_dest_last_N": 1 if step.is_new_beneficiary == 0 else min(5, step.step_idx + 1),
        "steps_since_last_tx": step.timing_delay,
        "is_balance_drain": step.is_balance_drain,
        "is_new_beneficiary": step.is_new_beneficiary
    }
    df = pd.DataFrame([row_dict])[FEATURE_COLUMNS]
    if scaler is not None:
        scaled_arr = scaler.transform(df)
        df = pd.DataFrame(scaled_arr, columns=FEATURE_COLUMNS)
    return df

class StaticCampaignGenerator:
    def __init__(self, exfiltration_target: float = CAMPAIGN_EXFILTRATION_TARGET):
        self.exfiltration_target = exfiltration_target

    def generate_known_campaigns(self, num_campaigns: int = 50) -> List[Campaign]:
        campaigns = []
        archetypes = ["mule_cascade", "smurfing", "rapid_fire"]
        
        for i in range(num_campaigns):
            arch = archetypes[i % len(archetypes)]
            c = Campaign(
                campaign_id=f"known_{i}_{arch}",
                archetype=arch,
                target_exfiltration=self.exfiltration_target
            )
            
            if arch == "mule_cascade":
                # Probe -> Large Transfer -> Cash out
                c.add_step(TransactionStep(0, 1000.0, 1.0, "dest_probe", f"payer_{i}", "TRANSFER", 0, 1))
                c.add_step(TransactionStep(1, 30000.0, 2.0, "dest_mule", f"payer_{i}", "TRANSFER", 1, 1))
                c.add_step(TransactionStep(2, 20000.0, 1.0, "dest_cash", f"payer_{i}", "CASH_OUT", 1, 0))
            
            elif arch == "smurfing":
                # 6 small transactions of ₹9,000 each
                for step_idx in range(6):
                    c.add_step(TransactionStep(
                        step_idx=step_idx,
                        amount=9000.0,
                        timing_delay=0.5,
                        beneficiary=f"dest_smurf_{step_idx}",
                        payer_account=f"payer_{i}",
                        action_type="TRANSFER",
                        is_balance_drain=0 if step_idx < 5 else 1,
                        is_new_beneficiary=1
                    ))
            
            elif arch == "rapid_fire":
                # 4 fast transactions of ₹13,000 each
                for step_idx in range(4):
                    c.add_step(TransactionStep(
                        step_idx=step_idx,
                        amount=13000.0,
                        timing_delay=0.1,
                        beneficiary=f"dest_rf_{step_idx}",
                        payer_account=f"payer_{i}",
                        action_type="TRANSFER",
                        is_balance_drain=0,
                        is_new_beneficiary=1
                    ))
            
            campaigns.append(c)
        return campaigns

    def generate_unknown_campaigns(self, num_campaigns: int = 50) -> List[Campaign]:
        campaigns = []
        archetypes = ["slow_drip", "interleaved_noise"]
        
        for i in range(num_campaigns):
            arch = archetypes[i % len(archetypes)]
            c = Campaign(
                campaign_id=f"unknown_{i}_{arch}",
                archetype=arch,
                target_exfiltration=self.exfiltration_target
            )
            
            if arch == "slow_drip":
                # 12 micro-transactions of ₹4,500 with long delay between steps
                for step_idx in range(12):
                    c.add_step(TransactionStep(
                        step_idx=step_idx,
                        amount=4500.0,
                        timing_delay=24.0,  # Long timing delay
                        beneficiary="dest_reused_mule",
                        payer_account=f"payer_unk_{i}",
                        action_type="TRANSFER",
                        is_balance_drain=0,
                        is_new_beneficiary=0 if step_idx > 0 else 1
                    ))
            
            elif arch == "interleaved_noise":
                # Interleaved low amounts and cashouts
                amounts = [3000.0, 12000.0, 2000.0, 15000.0, 4000.0, 16000.0]
                types = ["TRANSFER", "TRANSFER", "CASH_OUT", "TRANSFER", "CASH_OUT", "TRANSFER"]
                for step_idx in range(len(amounts)):
                    c.add_step(TransactionStep(
                        step_idx=step_idx,
                        amount=amounts[step_idx],
                        timing_delay=12.0,
                        beneficiary=f"dest_interleaved_{step_idx % 2}",
                        payer_account=f"payer_unk_{i}",
                        action_type=types[step_idx],
                        is_balance_drain=0,
                        is_new_beneficiary=1 if step_idx < 2 else 0
                    ))
            
            campaigns.append(c)
        return campaigns
