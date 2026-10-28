"""Generates publication-quality figures for dual held-out language evaluation."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_dual_held_out_figures(
    results_file: str = "results/tables/dual_held_out_comprehensive_results.csv",
    plots_dir: str = "results/plots",
):
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found yet.")
        return

    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    df = pd.read_csv(results_file)

    # Filter to tuned threshold results
    df_tuned = df[df["threshold_mode"].str.contains("Tuned")].copy()

    # 1. Figure 4: Dual Held-Out Comparison (AUROC & Balanced Acc)
    fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(
        data=df_tuned,
        x="model",
        y="auroc",
        hue="split",
        ax=ax1,
        palette="viridis",
    )
    ax1.set_title("AUROC: Indo-Aryan Seen vs. Dravidian vs. Tibeto-Burman", fontweight="bold")
    ax1.set_ylabel("AUROC Score")
    ax1.set_xlabel("Architecture")
    ax1.set_ylim(0, 1.1)
    ax1.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="Random Guess (0.50)")
    ax1.legend(loc="lower left", fontsize=9)
    ax1.tick_params(axis="x", rotation=15)

    sns.barplot(
        data=df_tuned,
        x="model",
        y="balanced_acc",
        hue="split",
        ax=ax2,
        palette="magma",
    )
    ax2.set_title("Balanced Accuracy: Cross-Family Generalization", fontweight="bold")
    ax2.set_ylabel("Balanced Accuracy")
    ax2.set_xlabel("Architecture")
    ax2.set_ylim(0, 1.1)
    ax2.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="Random Guess (0.50)")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    fig4_path = os.path.join(plots_dir, "fig4_dual_held_out_comparison.png")
    fig4.savefig(fig4_path, dpi=300)
    plt.close(fig4)

    # 2. Figure 5: Precision vs Recall across Threshold Modes
    fig5, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="recall",
        y="precision",
        hue="model",
        style="split",
        s=120,
        ax=ax,
        edgecolor="black",
    )
    ax.set_title("Precision vs. Recall Frontier across Models & Splits", fontweight="bold")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    fig5_path = os.path.join(plots_dir, "fig5_precision_recall_frontier.png")
    fig5.savefig(fig5_path, dpi=300)
    plt.close(fig5)

    print(f"Generated dual held-out figures in {plots_dir}")


if __name__ == "__main__":
    generate_dual_held_out_figures()
