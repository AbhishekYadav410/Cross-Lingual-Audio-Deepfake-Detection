"""Master Model Comparison & Publication Figure Generator.

Compares:
1. Model A: Calibrated RBF-SVM (LFCC Baseline)
2. Model B1: Lightweight CNN (Log-Mel)
3. Model B2: LCNN-MFM (Max-Feature-Map)
4. Model B3: SE-ResNet-18 (Squeeze-and-Excitation)
5. Model B4: DistilHuBERT (Pre-trained Self-Supervised Speech Representation)
6. Hybrid Ensemble (LCNN + SE-ResNet + SVM)

Evaluated across:
- Seen In-Language (Indo-Aryan)
- Held-Out 1 (Dravidian: Tamil, Telugu, Kannada, Malayalam)
- Held-Out 2 (Tibeto-Burman: Bodo, Manipuri)
- External Benchmark 1: MLAAD-tiny (Out-of-Distribution European/Western Generators)
- External Benchmark 2: ASVspoof 5 (State-of-the-Art Neural Vocoders)
- Codec Robustness (Clean -> MP3 64k -> MP3 32k -> Opus 24k -> Opus 16k)
"""

from typing import Dict, Any, List
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def generate_master_comparison(
    tables_dir: str = "results/tables",
    plots_dir: str = "results/plots",
):
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Load available results
    dual_held_out_path = os.path.join(tables_dir, "dual_held_out_comprehensive_results.csv")
    distilhubert_eval_path = os.path.join(tables_dir, "distilhubert_comprehensive_evaluation.csv")
    ext_benchmarks_path = os.path.join(tables_dir, "external_benchmarks_combined.csv")
    codec_table_path = os.path.join(tables_dir, "table2_rq2_codec_robustness.csv")
    distilhubert_codec_path = os.path.join(tables_dir, "distilhubert_codec_robustness.csv")

    print("\n--- Compiling Master Model Comparison Table ---")

    records = []

    # Process dual held out (tuned threshold)
    if os.path.exists(dual_held_out_path):
        df_dho = pd.read_csv(dual_held_out_path)
        df_dho_tuned = df_dho[df_dho["threshold_mode"].str.contains("Tuned")].copy()
        for _, row in df_dho_tuned.iterrows():
            ci_str = str(row.get("auroc_ci", "")).strip("[]").split(",")
            ci_low = float(ci_str[0].strip()) if len(ci_str) == 2 else float(row["auroc"])
            ci_high = float(ci_str[1].strip()) if len(ci_str) == 2 else float(row["auroc"])
            records.append({
                "Evaluation_Domain": row["split"],
                "Model": row["model"],
                "AUROC": float(row["auroc"]),
                "AUROC_95CI_Low": ci_low,
                "AUROC_95CI_High": ci_high,
                "EER_percent": float(row["eer"]) * 100.0,
                "Accuracy_Tuned": float(row["accuracy"]) * 100.0,
                "Precision_Tuned": float(row["precision"]) * 100.0,
                "Recall_Tuned": float(row["recall"]) * 100.0,
                "F1_Tuned": float(row["macro_f1"]) * 100.0,
                "Balanced_Acc": float(row["balanced_acc"]) * 100.0,
                "Threshold": float(row["applied_threshold"]),
            })

    # Process DistilHuBERT linguistic and benchmark evaluation
    if os.path.exists(distilhubert_eval_path):
        df_dh = pd.read_csv(distilhubert_eval_path)
        for _, row in df_dh.iterrows():
            domain = row["Evaluation_Domain"]
            records.append({
                "Evaluation_Domain": domain,
                "Model": "DistilHuBERT (Self-Supervised)",
                "AUROC": row["AUROC"],
                "AUROC_95CI_Low": row["AUROC_95CI_Low"],
                "AUROC_95CI_High": row["AUROC_95CI_High"],
                "EER_percent": row["EER_percent"],
                "Accuracy_Tuned": row["Acc_Tuned"],
                "Precision_Tuned": row["Prec_Tuned"],
                "Recall_Tuned": row["Rec_Tuned"],
                "F1_Tuned": row["F1_Tuned"],
                "Balanced_Acc": (row["Rec_Tuned"] + row["Acc_Tuned"]) / 2.0,  # balanced approximation
                "Threshold": row["Optimal_Threshold"],
            })

    # Process external benchmarks for earlier models
    if os.path.exists(ext_benchmarks_path):
        df_ext = pd.read_csv(ext_benchmarks_path)
        for _, row in df_ext.iterrows():
            ci_str = str(row.get("auroc_ci", "")).strip("[]").split(",")
            ci_low = float(ci_str[0].strip()) if len(ci_str) == 2 else float(row["auroc"])
            ci_high = float(ci_str[1].strip()) if len(ci_str) == 2 else float(row["auroc"])
            records.append({
                "Evaluation_Domain": f"External Benchmark ({row['benchmark']})",
                "Model": row["model"],
                "AUROC": float(row["auroc"]),
                "AUROC_95CI_Low": ci_low,
                "AUROC_95CI_High": ci_high,
                "EER_percent": float(row["eer"]) * 100.0,
                "Accuracy_Tuned": float(row["accuracy"]) * 100.0,
                "Precision_Tuned": float(row["precision"]) * 100.0,
                "Recall_Tuned": float(row["recall"]) * 100.0,
                "F1_Tuned": float(row["macro_f1"]) * 100.0,
                "Balanced_Acc": float(row["balanced_acc"]) * 100.0,
                "Threshold": 0.5,
            })

    df_master = pd.DataFrame(records)
    master_path = os.path.join(tables_dir, "master_model_comparison.csv")
    df_master.to_csv(master_path, index=False)
    print(f"Master comparison table saved to: {master_path}")

    # Generate Publication Figures
    generate_master_figures(df_master, tables_dir, plots_dir)


def generate_master_figures(df_master: pd.DataFrame, tables_dir: str, plots_dir: str):
    """Renders high-resolution publication figures comparing all architectures."""

    # 1. Figure 6: Cross-Family Generalization: DistilHuBERT vs LCNN vs SE-ResNet vs Ensemble
    linguistic_domains = [
        "In-Language Seen (Indo-Aryan)",
        "Held-Out 1 (Dravidian)",
        "Held-Out 2 (Tibeto-Burman)",
    ]
    df_ling = df_master[df_master["Evaluation_Domain"].isin(linguistic_domains)].copy()

    if not df_ling.empty:
        fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        sns.barplot(
            data=df_ling,
            x="Evaluation_Domain",
            y="AUROC",
            hue="Model",
            ax=ax1,
            palette="deep",
        )
        ax1.set_title("Cross-Lingual AUROC Generalization", fontweight="bold")
        ax1.set_ylabel("AUROC")
        ax1.set_ylim(0.4, 1.05)
        ax1.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random Guess (0.50)")
        ax1.legend(loc="lower left", fontsize=8)
        ax1.tick_params(axis="x", rotation=10)

        sns.barplot(
            data=df_ling,
            x="Evaluation_Domain",
            y="EER_percent",
            hue="Model",
            ax=ax2,
            palette="deep",
        )
        ax2.set_title("Equal Error Rate (EER %) - Lower is Better", fontweight="bold")
        ax2.set_ylabel("EER (%)")
        ax2.set_ylim(0, 35)
        ax2.legend(loc="upper left", fontsize=8)
        ax2.tick_params(axis="x", rotation=10)

        plt.tight_layout()
        f6_path = os.path.join(plots_dir, "fig6_distilhubert_vs_baselines_cross_lingual.png")
        fig6.savefig(f6_path, dpi=300)
        plt.close(fig6)
        print(f"Saved: {f6_path}")

    # 2. Figure 7: External Benchmark Generalization (MLAAD & ASVspoof 5)
    bench_domains = [
        "External Benchmark 1 (MLAAD)",
        "External Benchmark (MLAAD)",
        "External Benchmark 2 (ASVspoof 5)",
        "External Benchmark (ASVspoof 5)",
    ]
    df_ext = df_master[df_master["Evaluation_Domain"].isin(bench_domains)].copy()
    if not df_ext.empty:
        # Standardize domain names
        df_ext["Domain_Clean"] = df_ext["Evaluation_Domain"].apply(
            lambda d: "MLAAD (OOD Generators)" if "MLAAD" in d else "ASVspoof 5 (OOD Vocoders)"
        )

        fig7, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(
            data=df_ext,
            x="Domain_Clean",
            y="AUROC",
            hue="Model",
            ax=ax,
            palette="Set2",
        )
        ax.set_title("External Benchmark Generalization (Out-of-Distribution)", fontweight="bold")
        ax.set_ylabel("AUROC Score")
        ax.set_ylim(0.3, 1.05)
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="Random (0.50)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        plt.tight_layout()
        f7_path = os.path.join(plots_dir, "fig7_external_benchmark_generalization.png")
        fig7.savefig(f7_path, dpi=300)
        plt.close(fig7)
        print(f"Saved: {f7_path}")

    # 3. Figure 8: Codec Degradation Curves across models
    codec_table_path = os.path.join(tables_dir, "table2_rq2_codec_robustness.csv")
    dh_codec_path = os.path.join(tables_dir, "distilhubert_codec_robustness.csv")

    if os.path.exists(codec_table_path) and os.path.exists(dh_codec_path):
        df_c1 = pd.read_csv(codec_table_path)
        df_c2 = pd.read_csv(dh_codec_path)

        # Map condition names to uniform formatting
        cond_map = {
            "clean": "Clean",
            "mp3_64k": "MP3 64k",
            "mp3_32k": "MP3 32k",
            "opus_24k": "Opus 24k",
            "opus_16k": "Opus 16k",
        }

        c_records = []
        for _, r in df_c1.iterrows():
            cond = cond_map.get(str(r["codec_condition"]).lower(), str(r["codec_condition"]))
            c_records.append({"Condition": cond, "Model": "Baseline CNN (No Aug)", "AUROC": float(r["clean_cnn_auroc"])})
            c_records.append({"Condition": cond, "Model": "LCNN-MFM (Augmented)", "AUROC": float(r["lcnn_mfm_auroc"])})
            c_records.append({"Condition": cond, "Model": "SE-ResNet-18 (Augmented)", "AUROC": float(r["se_resnet_auroc"])})

        for _, r in df_c2.iterrows():
            c_records.append({
                "Condition": r["Condition"],
                "Model": "DistilHuBERT (Self-Supervised)",
                "AUROC": float(r["AUROC"]),
            })

        df_all_codec = pd.DataFrame(c_records)
        condition_order = ["Clean", "MP3 64k", "MP3 32k", "Opus 24k", "Opus 16k"]

        fig8, ax = plt.subplots(figsize=(9, 5))
        sns.lineplot(
            data=df_all_codec,
            x="Condition",
            y="AUROC",
            hue="Model",
            marker="o",
            markersize=8,
            linewidth=2.2,
            ax=ax,
        )
        ax.set_title("Codec Robustness Degradation Curves (RQ2)", fontweight="bold")
        ax.set_ylabel("AUROC Score")
        ax.set_ylim(0.4, 1.02)
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random (0.50)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        plt.tight_layout()
        f8_path = os.path.join(plots_dir, "fig8_codec_robustness_all_models.png")
        fig8.savefig(f8_path, dpi=300)
        plt.close(fig8)
        print(f"Saved: {f8_path}")


if __name__ == "__main__":
    generate_master_comparison()
