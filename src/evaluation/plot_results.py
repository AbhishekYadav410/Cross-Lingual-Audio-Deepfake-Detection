"""Generates publication-quality figures from experimental result tables."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_figures(results_dir: str = "results/tables", plots_dir: str = "results/plots"):
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    df_rq1 = pd.read_csv(os.path.join(results_dir, "table1_rq1_cross_lingual.csv"))
    df_rq2 = pd.read_csv(os.path.join(results_dir, "table2_rq2_codec_robustness.csv"))
    df_eff = pd.read_csv(os.path.join(results_dir, "table3_rq3_efficiency.csv"))

    # 1. Figure 1: Cross-Lingual AUROC Comparison across 3 splits
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    palette = {
        "In-Language Test (Seen)": "#1f77b4",
        "Held-Out Dravidian": "#2ca02c",
        "Held-Out Tibeto-Burman": "#d62728",
    }
    sns.barplot(data=df_rq1, x="model", y="auroc", hue="split", palette=palette, ax=ax1)
    ax1.set_title("RQ1: Generalization Across Language Families (Indo-Aryan -> Dravidian & Tibeto-Burman)", fontweight="bold")
    ax1.set_ylabel("AUROC Score")
    ax1.set_xlabel("Architecture")
    ax1.set_ylim(0.4, 1.05)
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Random Guess (0.50)")
    ax1.legend(loc="lower left", fontsize=10)
    ax1.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    fig1.savefig(os.path.join(plots_dir, "fig1_rq1_cross_lingual.png"), dpi=300)
    plt.close(fig1)

    # 2. Figure 2: Codec Robustness Profile
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df_rq2))
    width = 0.25

    ax2.bar(x - width, df_rq2["clean_cnn_auroc"], width, label="Clean Baseline 2D-CNN", color="#d95f02")
    ax2.bar(x, df_rq2["lcnn_mfm_auroc"], width, label="Codec-Augmented LCNN-MFM", color="#7570b3")
    ax2.bar(x + width, df_rq2["se_resnet_auroc"], width, label="Codec-Augmented SE-ResNet-18", color="#1b9e77")

    ax2.set_xticks(x)
    ax2.set_xticklabels(df_rq2["codec_condition"], rotation=15, fontsize=10)
    ax2.set_ylabel("Held-Out Dravidian AUROC")
    ax2.set_title("RQ2: Forensic Robustness Under Lossy Compression Codecs", fontweight="bold")
    ax2.set_ylim(0.3, 1.05)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    fig2.savefig(os.path.join(plots_dir, "fig2_rq2_codec_robustness.png"), dpi=300)
    plt.close(fig2)

    # 3. Figure 3: Efficiency vs Latency Tradeoff (Pareto Frontier)
    # Join with Held-Out Dravidian AUROC from df_rq1
    drav_scores = df_rq1[df_rq1["split"] == "Held-Out Dravidian"].set_index("model")["auroc"].to_dict()

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    latencies = []
    aurocs = []
    labels = []

    for _, row in df_eff.iterrows():
        m = row["model"]
        lat = row["mean_latency_ms"]
        score = drav_scores.get(m, 0.50)
        latencies.append(lat)
        aurocs.append(score)
        labels.append(m)

    scatter = ax3.scatter(latencies, aurocs, s=180, c=aurocs, cmap="viridis", edgecolors="black", zorder=3)
    cbar = fig3.colorbar(scatter, ax=ax3)
    cbar.set_label("Held-Out Dravidian AUROC")

    for i, label in enumerate(labels):
        offset_y = 0.015 if i % 2 == 0 else -0.025
        offset_x = 0.3
        ax3.annotate(
            f"{label}\n({latencies[i]:.2f} ms)",
            (latencies[i] + offset_x, aurocs[i] + offset_y),
            fontsize=9,
            weight="semibold",
        )

    ax3.set_title("RQ3: Latency vs. Accuracy Pareto Frontier (CPU)", fontweight="bold")
    ax3.set_xlabel("Mean Inference Latency per 3s Sample (ms) [Lower is Better]")
    ax3.set_ylabel("Held-Out Dravidian AUROC [Higher is Better]")
    ax3.set_xlim(-0.5, 22.0)
    ax3.set_ylim(0.40, 1.05)
    ax3.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig3.savefig(os.path.join(plots_dir, "fig3_rq3_latency_vs_accuracy.png"), dpi=300)
    plt.close(fig3)

    print(f"Generated 3 updated publication figures in: {plots_dir}")


if __name__ == "__main__":
    generate_figures()
