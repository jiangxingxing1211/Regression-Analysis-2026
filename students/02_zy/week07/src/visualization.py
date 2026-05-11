"""
Module: src.visualization
Purpose: Plot learning curves and save results.
"""
import matplotlib.pyplot as plt
from pathlib import Path


def plot_learning_curve(model_full, model_mini, save_dir: Path):
    plt.figure(figsize=(10, 6))
    plt.plot(model_full.loss_history_, label="Full Batch GD", color="steelblue")
    # 给Mini-Batch线加透明度，这样就能看到下面的蓝色线了
    plt.plot(model_mini.loss_history_, label="Mini-Batch GD", color="darkorange", alpha=0.6)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Learning Curve: Full Batch vs Mini-Batch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "learning_curve_full_vs_mini.png", dpi=150)
    plt.close()