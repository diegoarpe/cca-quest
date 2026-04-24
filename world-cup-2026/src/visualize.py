"""Charts summarising the model: top-20 probabilities and feature importance."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_top_contenders(probs: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    top = probs.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["team"], top["P(winner)"] * 100, color="#1f77b4")
    ax.set_xlabel("P(winner) %")
    ax.set_title(f"World Cup 2026 — top {top_n} contenders (Monte Carlo)")
    for i, v in enumerate(top["P(winner)"] * 100):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_round_probabilities(probs: pd.DataFrame, out_path: Path, top_n: int = 12) -> None:
    top = probs.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    cols = ["P(quarterfinal)", "P(semifinal)", "P(final)", "P(winner)"]
    labels = ["Quarterfinal", "Semifinal", "Final", "Winner"]
    colors = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    y = range(len(top))
    for col, label, color in zip(cols, labels, colors):
        ax.barh(y, top[col] * 100, color=color, label=label, height=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(top["team"])
    ax.set_xlabel("Probability %")
    ax.set_title(f"Stage-reaching probabilities — top {top_n}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_feature_importance(importance: pd.Series, out_path: Path) -> None:
    imp = importance.sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp.index, imp.values, color="#d95f02")
    ax.set_xlabel("Relative importance (blended Ridge + GBM)")
    ax.set_title("Feature importance")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
