import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.config import PLOTS_DIR

class Visualizer:
    def __init__(self, output_dir: str = PLOTS_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="dark")

    def plot_four_worlds_comparison(self, results_df: pd.DataFrame):
        """
        results_df columns: ['Detector', 'World', 'CSR', 'F1', 'IT', 'CFE']
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Campaign Success Rate (CSR)
        sns.barplot(data=results_df, x="World", y="CSR", hue="Detector", ax=axes[0, 0], palette="crest")
        axes[0, 0].set_title("Campaign Success Rate (CSR) ↑", fontsize=14, fontweight="bold")
        axes[0, 0].set_ylabel("CSR (Fraction)")
        axes[0, 0].set_ylim(0, 1.05)

        # 2. Financial Exposure (CFE)
        sns.barplot(data=results_df, x="World", y="CFE", hue="Detector", ax=axes[0, 1], palette="flare")
        axes[0, 1].set_title("Cumulative Financial Exposure (CFE) ₹ ↓", fontsize=14, fontweight="bold")
        axes[0, 1].set_ylabel("Mean Exfiltrated ₹")

        # 3. Intervention Timing (IT)
        sns.barplot(data=results_df, x="World", y="IT", hue="Detector", ax=axes[1, 0], palette="viridis")
        axes[1, 0].set_title("Intervention Timing (IT) Steps ↓", fontsize=14, fontweight="bold")
        axes[1, 0].set_ylabel("Mean Steps Before Block")

        # 4. F1 Performance & Detector Degradation
        sns.barplot(data=results_df, x="World", y="F1", hue="Detector", ax=axes[1, 1], palette="magma")
        axes[1, 1].set_title("Detector F1 Score ↑", fontsize=14, fontweight="bold")
        axes[1, 1].set_ylabel("F1 Score")
        axes[1, 1].set_ylim(0, 1.05)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "four_worlds_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Saved Four Worlds comparison plot to {path}")

    def plot_degradation_curves(self, results_df: pd.DataFrame):
        """
        Line plot showing F1 performance degradation across Worlds 1 -> 2 -> 3 -> 4.
        """
        plt.figure(figsize=(9, 6))
        sns.lineplot(
            data=results_df,
            x="World",
            y="F1",
            hue="Detector",
            marker="o",
            linewidth=2.5,
            markersize=8
        )
        plt.title("Detector Degradation Curve Across Worlds", fontsize=15, fontweight="bold")
        plt.ylabel("F1 Score", fontsize=12)
        plt.xlabel("Simulation World", fontsize=12)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()

        path = os.path.join(self.output_dir, "degradation_curves.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Saved degradation curve plot to {path}")
