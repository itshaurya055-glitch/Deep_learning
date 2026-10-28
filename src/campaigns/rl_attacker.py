import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from typing import Any, Dict, Tuple, List
from src.campaigns.schema import Campaign, TransactionStep
from src.campaigns.static_generator import step_to_feature_dataframe
from src.config import CAMPAIGN_EXFILTRATION_TARGET, MAX_CAMPAIGN_STEPS, PPO_PARAMS

class FraudAttackerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, detector: Any, scaler=None, is_unknown_archetype: bool = False):
        super(FraudAttackerEnv, self).__init__()
        self.detector = detector
        self.scaler = scaler
        self.is_unknown_archetype = is_unknown_archetype
        self.exfiltration_target = CAMPAIGN_EXFILTRATION_TARGET
        self.max_steps = MAX_CAMPAIGN_STEPS

        # State: [accumulated_ratio, step_ratio, last_decision (0=allowed, 1=blocked)]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Action: 9 Discrete combinations of (Amount x Timing Delay)
        # Amount: 0=Small (2k), 1=Medium (6k), 2=Large (15k)
        # Delay: 0=Short (0.5), 1=Medium (6.0), 2=Long (24.0)
        self.action_space = spaces.Discrete(9)

        self.action_map = {
            0: (2000.0, 0.5),
            1: (2000.0, 6.0),
            2: (2000.0, 24.0),
            3: (6000.0, 0.5),
            4: (6000.0, 6.0),
            5: (6000.0, 24.0),
            6: (15000.0, 0.5),
            7: (15000.0, 6.0),
            8: (15000.0, 24.0),
        }

        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.accumulated_amount = 0.0
        self.current_step_idx = 0
        self.last_decision = 0.0
        self.terminated = False
        self.payer_id = f"rl_payer_{np.random.randint(1000, 9999)}"
        self.beneficiaries = [f"dest_rl_{np.random.randint(100, 999)}"]
        
        if hasattr(self.detector, "reset_history"):
            self.detector.reset_history()

        obs = np.array([
            self.accumulated_amount / self.exfiltration_target,
            self.current_step_idx / self.max_steps,
            self.last_decision
        ], dtype=np.float32)

        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.terminated:
            obs = np.array([
                self.accumulated_amount / self.exfiltration_target,
                self.current_step_idx / self.max_steps,
                1.0
            ], dtype=np.float32)
            return obs, 0.0, True, False, {}

        amount, delay = self.action_map[action]
        
        is_new_ben = 1 if (np.random.rand() > 0.5 or len(self.beneficiaries) == 0) else 0
        if is_new_ben:
            ben = f"dest_rl_{np.random.randint(100, 999)}"
            self.beneficiaries.append(ben)
        else:
            ben = self.beneficiaries[0]

        is_drain = 1 if (self.accumulated_amount + amount >= self.exfiltration_target) else 0
        action_type = "TRANSFER" if not self.is_unknown_archetype else ("TRANSFER" if np.random.rand() > 0.3 else "CASH_OUT")

        step_obj = TransactionStep(
            step_idx=self.current_step_idx,
            amount=amount,
            timing_delay=delay,
            beneficiary=ben,
            payer_account=self.payer_id,
            action_type=action_type,
            is_balance_drain=is_drain,
            is_new_beneficiary=is_new_ben
        )

        feat_df = step_to_feature_dataframe(step_obj, self.scaler)
        
        if hasattr(self.detector, "predict_step"):
            prob, is_blocked = self.detector.predict_step(feat_df, self.payer_id)
        else:
            prob = self.detector.predict_proba(feat_df)[0]
            is_blocked = int(prob >= getattr(self.detector, "threshold", 0.5))

        blocked = bool(is_blocked == 1)
        self.current_step_idx += 1

        if blocked:
            self.last_decision = 1.0
            self.terminated = True
            reward = -10.0
        else:
            self.last_decision = 0.0
            self.accumulated_amount += amount
            reward = (amount / 2000.0) - 0.1

            if self.accumulated_amount >= self.exfiltration_target:
                reward += 20.0
                self.terminated = True
            elif self.current_step_idx >= self.max_steps:
                self.terminated = True

        obs = np.array([
            self.accumulated_amount / self.exfiltration_target,
            min(1.0, self.current_step_idx / self.max_steps),
            self.last_decision
        ], dtype=np.float32)

        return obs, reward, self.terminated, False, {"blocked": blocked, "amount": amount, "prob": prob}

def train_rl_attacker(detector: Any, scaler=None, is_unknown_archetype: bool = False, total_timesteps: int = 20000) -> PPO:
    print(f"Training PPO RL Attacker ({total_timesteps} steps, Unknown Archetype={is_unknown_archetype})...")
    env = FraudAttackerEnv(detector, scaler, is_unknown_archetype=is_unknown_archetype)
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=PPO_PARAMS["learning_rate"], seed=42)
    model.learn(total_timesteps=total_timesteps)
    return model

def generate_rl_campaigns(attacker_agent: PPO, detector: Any, scaler=None, is_unknown_archetype: bool = False, num_campaigns: int = 50) -> List[Campaign]:
    env = FraudAttackerEnv(detector, scaler, is_unknown_archetype=is_unknown_archetype)
    campaigns = []

    for i in range(num_campaigns):
        obs, _ = env.reset()
        done = False
        prefix = "unk" if is_unknown_archetype else "kn"
        c = Campaign(
            campaign_id=f"rl_{prefix}_{i}",
            archetype="rl_adaptive",
            target_exfiltration=env.exfiltration_target
        )

        step_idx = 0
        while not done:
            action, _ = attacker_agent.predict(obs, deterministic=True)
            amount, delay = env.action_map[int(action)]

            obs, reward, done, truncated, info = env.step(int(action))
            blocked = info.get("blocked", False)

            step_obj = TransactionStep(
                step_idx=step_idx,
                amount=amount,
                timing_delay=delay,
                beneficiary=env.beneficiaries[-1],
                payer_account=env.payer_id,
                action_type="TRANSFER",
                is_balance_drain=1 if env.accumulated_amount >= env.exfiltration_target else 0,
                is_new_beneficiary=1
            )
            c.add_step(step_obj)
            c.record_decision(step_idx, amount, blocked)
            step_idx += 1

        campaigns.append(c)

    return campaigns
