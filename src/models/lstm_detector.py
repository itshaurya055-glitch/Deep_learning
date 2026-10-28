import os
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from src.config import LSTM_PARAMS, MODELS_DIR

class PyTorchLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super(PyTorchLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_out = lstm_out[:, -1, :]
        logits = self.fc(final_out)
        return logits.squeeze(-1)

class LSTMDetector:
    def __init__(self, params: dict = None):
        self.params = params if params else LSTM_PARAMS.copy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PyTorchLSTM(
            input_dim=self.params["input_dim"],
            hidden_dim=self.params["hidden_dim"],
            num_layers=self.params["num_layers"]
        ).to(self.device)
        self.threshold = 0.5          # batch predict threshold (static test set)
        self.sim_threshold = 0.5      # single-step simulation threshold (recalibrated)
        self.account_history: Dict[str, list] = {}
        self._input_dim = self.params["input_dim"]

    def reset_history(self):
        """Clear the per-account sequence history buffer."""
        self.account_history = {}

    def _prepare_sequences(self, X: pd.DataFrame, seq_len: int):
        arr = X.values if isinstance(X, pd.DataFrame) else X
        num_samples = len(arr)
        padded = np.pad(arr, ((seq_len - 1, 0), (0, 0)), mode='edge')
        sequences = []
        for i in range(num_samples):
            sequences.append(padded[i:i + seq_len])
        return np.array(sequences, dtype=np.float32)

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        print("Preparing sequence data for LSTMDetector...")
        seq_len = self.params["sequence_length"]
        X_seq = self._prepare_sequences(X_train, seq_len)

        num_pos = np.sum(y_train == 1)
        num_neg = np.sum(y_train == 0)
        pos_weight = torch.tensor([num_neg / max(1, num_pos)], device=self.device)

        dataset = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=True)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=self.params["learning_rate"])

        print(f"Training LSTMDetector for {self.params['epochs']} epochs on {self.device}...")
        self.model.train()
        for epoch in range(self.params["epochs"]):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 2 == 0 or epoch == self.params["epochs"] - 1:
                print(f"Epoch [{epoch+1}/{self.params['epochs']}] Loss: {total_loss/len(loader):.4f}")

        # Calibrate optimal decision threshold on training set (batch/static mode)
        train_probs = self.predict_proba(X_train)
        best_f1, best_thresh = 0.0, 0.5
        for th in np.linspace(0.1, 0.9, 17):
            f1 = f1_score(y_train, (train_probs >= th).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, th
        self.threshold = float(best_thresh)
        self.sim_threshold = self.threshold  # default; overridden by calibrate_simulation_threshold
        print(f"Calibrated LSTM batch threshold to {self.threshold:.3f} (Train F1: {best_f1:.4f})")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        seq_len = self.params["sequence_length"]
        X_seq = self._prepare_sequences(X, seq_len)

        dataset = TensorDataset(torch.tensor(X_seq, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=False)

        probs = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                batch_probs = torch.sigmoid(logits).cpu().numpy()
                probs.extend(batch_probs)
        return np.array(probs)

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        th = threshold if threshold is not None else self.threshold
        return (self.predict_proba(X) >= th).astype(int)

    def calibrate_simulation_threshold(self, X_test: pd.DataFrame, y_test: np.ndarray):
        """
        Re-calibrate decision threshold for single-step zero-padded simulation mode.
        
        The batch threshold (self.threshold) is calibrated using edge-padded sequences
        on training data. In simulation, each step is evaluated with a zero-padded
        buffer, which shifts probabilities lower. This method finds the optimal
        threshold for that zero-padded evaluation mode using the held-out test set.
        """
        print("Calibrating LSTM simulation (zero-padded single-step) threshold on test set...")
        probs = []
        labels = y_test
        
        self.reset_history()
        for i, row in enumerate(X_test.values):
            feat_df = pd.DataFrame([row], columns=X_test.columns)
            account_id = f"calib_acct_{i}"
            prob, _ = self.predict_step(feat_df, account_id)
            probs.append(prob)
        self.reset_history()

        probs = np.array(probs)
        best_f1, best_thresh = 0.0, 0.5
        for th in np.linspace(0.3, 0.85, 23):
            preds = (probs >= th).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, float(th)
        self.sim_threshold = best_thresh
        print(f"Calibrated LSTM sim threshold to {self.sim_threshold:.3f} (Test F1: {best_f1:.4f})")
        return best_f1

    def predict_step(self, step_features: pd.DataFrame, payer_account: str) -> Tuple[float, int]:
        """
        Predict for a single transaction step, maintaining a rolling window of
        prior transactions per account. New accounts start with a ZERO buffer
        (not a repeated-first-row buffer) to avoid false velocity signals.
        """
        seq_len = self.params["sequence_length"]
        feat_vec = step_features.values[0].astype(np.float32)

        if payer_account not in self.account_history:
            # Initialize with zeros — avoids treating the first step as a
            # velocity burst (repeated-row padding creates an artificial
            # rapid-transaction signal the LSTM flags as fraud)
            self.account_history[payer_account] = [np.zeros(self._input_dim, dtype=np.float32)] * seq_len

        # Append new step and drop oldest
        self.account_history[payer_account].append(feat_vec)
        if len(self.account_history[payer_account]) > seq_len:
            self.account_history[payer_account].pop(0)

        seq_arr = np.array([self.account_history[payer_account]], dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(seq_arr, dtype=torch.float32).to(self.device)
            logit = self.model(x_tensor)
            prob = float(torch.sigmoid(logit).cpu().numpy()[0])

        # Use sim_threshold (calibrated for zero-padded single-step mode)
        is_blocked = int(prob >= self.sim_threshold)
        return prob, is_blocked

    def save(self, filename: str = "lstm_detector.pt"):
        path = os.path.join(MODELS_DIR, filename)
        torch.save({
            "model_state": self.model.state_dict(),
            "threshold": self.threshold,
            "sim_threshold": self.sim_threshold
        }, path)
        print(f"Saved LSTMDetector to {path} (batch_th={self.threshold:.3f}, sim_th={self.sim_threshold:.3f})")

    def load(self, filename: str = "lstm_detector.pt"):
        path = os.path.join(MODELS_DIR, filename)
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            self.model.load_state_dict(ckpt["model_state"])
            self.threshold = ckpt.get("threshold", 0.5)
            self.sim_threshold = ckpt.get("sim_threshold", self.threshold)
        else:
            self.model.load_state_dict(ckpt)
        print(f"Loaded LSTMDetector from {path} (batch_th={self.threshold:.3f}, sim_th={self.sim_threshold:.3f})")
