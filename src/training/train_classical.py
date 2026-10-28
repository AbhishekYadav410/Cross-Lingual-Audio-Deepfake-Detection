"""Training and evaluation pipeline for Model A (Classical Feature Baselines)."""

from typing import Dict, Any, Tuple
import os
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.features.classical_features import ClassicalFeatureExtractor
from src.models.classical_baseline import ClassicalBaseline
from src.evaluation.metrics import evaluate_predictions, compute_bootstrap_ci


def extract_dataset_features(
    df: pd.DataFrame,
    extractor: ClassicalFeatureExtractor,
    desc: str = "Extracting features",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts 180-d LFCC summary vectors and labels from DataFrame."""
    X_list = []
    y_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        audio_path = row["audio_path"]
        if os.path.exists(audio_path):
            audio, _ = sf.read(audio_path, dtype="float32")
        else:
            audio = np.zeros(48000, dtype=np.float32)

        feat_vec = extractor.extract_vector(audio)
        X_list.append(feat_vec)
        y_list.append(int(row["is_tts"]))

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int)


def train_and_eval_classical(
    splits_dir: str = "data/splits",
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results/tables",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    extractor = ClassicalFeatureExtractor(sample_rate=16000, n_coefficients=30)

    # 1. Load splits
    print("Loading data splits...")
    df_train = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(splits_dir, "val.csv"))
    df_test_in = pd.read_csv(os.path.join(splits_dir, "test_in_lang.csv"))
    df_test_dravidian = pd.read_csv(os.path.join(splits_dir, "test_held_out_dravidian.csv"))

    # 2. Extract features
    X_train, y_train = extract_dataset_features(df_train, extractor, desc="Train features")
    X_val, y_val = extract_dataset_features(df_val, extractor, desc="Val features")
    X_test_in, y_test_in = extract_dataset_features(df_test_in, extractor, desc="In-Lang Test features")
    X_test_drav, y_test_drav = extract_dataset_features(df_test_dravidian, extractor, desc="Dravidian Test features")

    results_rows = []

    for model_type in ["logistic_regression", "svm_rbf"]:
        print(f"\nTraining Model A: {model_type}...")
        model = ClassicalBaseline(model_type=model_type)
        model.fit(X_train, y_train)

        # Save checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"model_a_{model_type}.joblib")
        model.save(ckpt_path)
        print(f"Saved checkpoint to: {ckpt_path}")

        # Evaluate on In-Language Test
        probs_in = model.predict_proba(X_test_in)[:, 1]
        metrics_in = evaluate_predictions(y_test_in, probs_in)
        ci_in = compute_bootstrap_ci(y_test_in, probs_in, n_resamples=500)

        results_rows.append({
            "model": f"Model A ({model_type})",
            "eval_split": "In-Language Test (Seen)",
            "auroc": metrics_in["auroc"],
            "auroc_ci_lower": ci_in["auroc"]["lower"],
            "auroc_ci_upper": ci_in["auroc"]["upper"],
            "eer": metrics_in["eer"],
            "eer_ci_lower": ci_in["eer"]["lower"],
            "eer_ci_upper": ci_in["eer"]["upper"],
            "balanced_acc": metrics_in["balanced_accuracy_tau0.5"],
            "macro_f1": metrics_in["macro_f1"],
        })

        # Evaluate on Held-out Dravidian Test
        probs_drav = model.predict_proba(X_test_drav)[:, 1]
        metrics_drav = evaluate_predictions(y_test_drav, probs_drav)
        ci_drav = compute_bootstrap_ci(y_test_drav, probs_drav, n_resamples=500)

        results_rows.append({
            "model": f"Model A ({model_type})",
            "eval_split": "Held-Out Dravidian",
            "auroc": metrics_drav["auroc"],
            "auroc_ci_lower": ci_drav["auroc"]["lower"],
            "auroc_ci_upper": ci_drav["auroc"]["upper"],
            "eer": metrics_drav["eer"],
            "eer_ci_lower": ci_drav["eer"]["lower"],
            "eer_ci_upper": ci_drav["eer"]["upper"],
            "balanced_acc": metrics_drav["balanced_accuracy_tau0.5"],
            "macro_f1": metrics_drav["macro_f1"],
        })

    results_df = pd.DataFrame(results_rows)
    out_table = os.path.join(results_dir, "classical_results.csv")
    results_df.to_csv(out_table, index=False)
    print("\n" + "=" * 60)
    print("MODEL A RESULTS SUMMARY")
    print("=" * 60)
    print(results_df[["model", "eval_split", "auroc", "eer", "balanced_acc"]].to_string(index=False))
    return results_df


if __name__ == "__main__":
    train_and_eval_classical()
