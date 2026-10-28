import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.config import (
    PAYSIM_PATH, RANDOM_SEED, LEGIT_SAMPLE_SIZE,
    FRAUD_ELIGIBLE_TYPES, VELOCITY_WINDOW_STEPS, FEATURE_COLUMNS
)

class DataProcessor:
    def __init__(self, raw_path: str = PAYSIM_PATH):
        self.raw_path = raw_path
        self.scaler = StandardScaler()

    def load_and_sample(self) -> pd.DataFrame:
        print(f"Loading raw dataset from {self.raw_path}...")
        usecols = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud']
        df = pd.read_csv(self.raw_path, usecols=usecols)

        # 1. Rename to UPI framing
        df.rename(columns={
            "nameOrig": "payer_account",
            "nameDest": "beneficiary_account"
        }, inplace=True)

        # 2. Separate fraud and legitimate rows
        fraud_df = df[df["isFraud"] == 1].copy()
        legit_df = df[df["isFraud"] == 0].copy()

        print(f"Total Fraud rows: {len(fraud_df)}, Total Legitimate rows: {len(legit_df)}")

        # Stratified sample of legitimate rows across step and type
        n_sample = min(LEGIT_SAMPLE_SIZE, len(legit_df))
        sampled_legit = legit_df.sample(n=n_sample, random_state=RANDOM_SEED).reset_index(drop=True)

        print(f"Sampled Legitimate rows: {len(sampled_legit)}")

        # Combine sampled legitimate + 100% fraud
        combined_df = pd.concat([fraud_df, sampled_legit], ignore_index=True)
        combined_df.sort_values(by=["step"], inplace=True)
        combined_df.reset_index(drop=True, inplace=True)

        return combined_df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Engineering velocity, balance drain, and beneficiary features...")
        df = df.copy()

        # Balance drain flag
        df["is_balance_drain"] = ((df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)).astype(int)

        # One-hot encode types for TRANSFER and CASH_OUT
        df["type_TRANSFER"] = (df["type"] == "TRANSFER").astype(int)
        df["type_CASH_OUT"] = (df["type"] == "CASH_OUT").astype(int)

        # Beneficiary Novelty Flag
        df["pair_key"] = df["payer_account"] + "_" + df["beneficiary_account"]
        first_pair = df.groupby("pair_key")["step"].transform("min")
        df["is_new_beneficiary"] = (df["step"] == first_pair).astype(int)
        df.drop(columns=["pair_key"], inplace=True)

        # Velocity / Frequency over sliding window of steps per payer account
        df.sort_values(by=["payer_account", "step"], inplace=True)

        # Fast vectorized group operations without Python lambda overhead
        payer_group = df.groupby("payer_account")
        
        # Transaction count in last N
        df["tx_count_last_N"] = payer_group.cumcount().add(1).clip(upper=VELOCITY_WINDOW_STEPS)
        
        # Cumulative sum of amounts per account
        df["sum_amount_last_N"] = payer_group["amount"].cumsum()

        # Distinct beneficiaries in last N (cumsum of is_new_beneficiary)
        df["distinct_dest_last_N"] = payer_group["is_new_beneficiary"].cumsum().clip(lower=1, upper=VELOCITY_WINDOW_STEPS)
        
        # Steps since last transaction
        df["prev_step"] = payer_group["step"].shift(1)
        df["steps_since_last_tx"] = (df["step"] - df["prev_step"]).fillna(999.0)
        df.drop(columns=["prev_step"], inplace=True)

        # Re-sort chronologically
        df.sort_values(by=["step"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    def prepare_world1_split(self, df: pd.DataFrame, train_ratio: float = 0.8):
        """
        Filter to fraud-eligible transaction types (TRANSFER, CASH_OUT) for World 1 training
        to avoid type-cheating, and perform temporal split on step.
        """
        # Restrict to TRANSFER and CASH_OUT for detector training
        eligible_df = df[df["type"].isin(FRAUD_ELIGIBLE_TYPES)].copy().reset_index(drop=True)

        # Temporal split by step
        max_step = eligible_df["step"].max()
        split_step = eligible_df["step"].quantile(train_ratio)

        train_df = eligible_df[eligible_df["step"] <= split_step].copy()
        test_df = eligible_df[eligible_df["step"] > split_step].copy()

        print(f"World 1 Train set size: {len(train_df)} (Fraud: {train_df['isFraud'].sum()})")
        print(f"World 1 Static Test set size: {len(test_df)} (Fraud: {test_df['isFraud'].sum()})")

        X_train = train_df[FEATURE_COLUMNS].copy()
        y_train = train_df["isFraud"].values

        X_test = test_df[FEATURE_COLUMNS].copy()
        y_test = test_df["isFraud"].values

        # Fit scaler on train set
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=FEATURE_COLUMNS)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=FEATURE_COLUMNS)

        return (X_train_scaled, y_train, train_df), (X_test_scaled, y_test, test_df)

    def extract_empirical_distributions(self, df: pd.DataFrame) -> dict:
        """
        Extract empirical distributions for synthetic campaign generation (amount ranges, timing gaps, etc.)
        """
        fraud_df = df[df["isFraud"] == 1]
        legit_df = df[df["isFraud"] == 0]

        return {
            "fraud_amounts": {
                "mean": float(fraud_df["amount"].mean()),
                "std": float(fraud_df["amount"].std()),
                "min": float(fraud_df["amount"].min()),
                "max": float(fraud_df["amount"].max()),
                "p25": float(fraud_df["amount"].quantile(0.25)),
                "p50": float(fraud_df["amount"].quantile(0.50)),
                "p75": float(fraud_df["amount"].quantile(0.75)),
            },
            "legit_amounts": {
                "mean": float(legit_df["amount"].mean()),
                "p50": float(legit_df["amount"].quantile(0.50)),
            },
            "balance_orig_mean": float(df["oldbalanceOrg"].mean())
        }
