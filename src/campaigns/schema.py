from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class TransactionStep:
    step_idx: int
    amount: float
    timing_delay: float
    beneficiary: str
    payer_account: str
    action_type: str  # TRANSFER or CASH_OUT
    is_balance_drain: int
    is_new_beneficiary: int
    features: Optional[np.ndarray] = None

@dataclass
class Campaign:
    campaign_id: str
    archetype: str
    target_exfiltration: float
    steps: List[TransactionStep] = field(default_factory=list)
    status: str = "IN_PROGRESS"  # IN_PROGRESS, SUCCESS, BLOCKED
    cumulative_exfiltrated: float = 0.0
    first_blocked_step: Optional[int] = None
    executed_steps: int = 0

    def add_step(self, step: TransactionStep):
        self.steps.append(step)

    def record_decision(self, step_idx: int, amount: float, blocked: bool):
        self.executed_steps += 1
        if blocked:
            if self.status == "IN_PROGRESS":
                self.status = "BLOCKED"
                if self.first_blocked_step is None:
                    self.first_blocked_step = step_idx
        else:
            self.cumulative_exfiltrated += amount
            if self.cumulative_exfiltrated >= self.target_exfiltration and self.status == "IN_PROGRESS":
                self.status = "SUCCESS"
