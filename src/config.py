import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR
PAYSIM_PATH = os.path.join(DATA_DIR, "paysim.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Random Seed
RANDOM_SEED = 42

# Data Processing Configs
LEGIT_SAMPLE_SIZE = 50000  # Stratified random sample of legitimate rows
FRAUD_ELIGIBLE_TYPES = ["TRANSFER", "CASH_OUT"]
VELOCITY_WINDOW_STEPS = 5

# Feature Lists
RAW_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"
]

FEATURE_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type_TRANSFER",
    "type_CASH_OUT",
    "tx_count_last_N",
    "sum_amount_last_N",
    "distinct_dest_last_N",
    "steps_since_last_tx",
    "is_balance_drain",
    "is_new_beneficiary"
]

# Campaign Configs
CAMPAIGN_EXFILTRATION_TARGET = 50000.0  # ₹50,000 objective
MAX_CAMPAIGN_STEPS = 10

# XGBoost Parameters
XGB_PARAMS = {
    "n_estimators": 50,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}

# LSTM Parameters
LSTM_PARAMS = {
    "input_dim": len(FEATURE_COLUMNS),
    "hidden_dim": 32,
    "num_layers": 1,
    "learning_rate": 0.001,
    "epochs": 3,
    "batch_size": 512,
    "sequence_length": 5
}

# RL Attacker Parameters
PPO_PARAMS = {
    "total_timesteps": 30000,
    "learning_rate": 0.0003,
    "verbose": 0
}
