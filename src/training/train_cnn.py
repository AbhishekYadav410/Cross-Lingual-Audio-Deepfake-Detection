"""Training and evaluation pipeline for Model B (Lightweight 2D-CNN).

Trains:
1. Clean Training Regime (Clean-trained CNN)
2. Codec-Augmented Training Regime (Stochastic MP3/Opus augmentation)
"""

from typing import Dict, Any, Tuple
import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from src.models.lightweight_cnn import LightweightAudioCNN
from src.training.trainer import AudioSpectrogramDataset, CNNTrainer
from src.robustness.ffmpeg_codec import CodecAugmenter
from src.evaluation.metrics import evaluate_predictions, compute_bootstrap_ci


def train_and_eval_cnn(
    splits_dir: str = "data/splits",
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results/tables",
    epochs: int = 15,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("Loading data splits...")
    df_train = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(splits_dir, "val.csv"))
    df_test_in = pd.read_csv(os.path.join(splits_dir, "test_in_lang.csv"))
    df_test_dravidian = pd.read_csv(os.path.join(splits_dir, "test_held_out_dravidian.csv"))

    val_dataset = AudioSpectrogramDataset(df_val, feature_type="log_mel")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_in_dataset = AudioSpectrogramDataset(df_test_in, feature_type="log_mel")
    test_in_loader = DataLoader(test_in_dataset, batch_size=batch_size, shuffle=False)

    test_drav_dataset = AudioSpectrogramDataset(df_test_dravidian, feature_type="log_mel")
    test_drav_loader = DataLoader(test_drav_dataset, batch_size=batch_size, shuffle=False)

    regimes = [
        {"name": "cnn_clean", "augmenter": None},
        {"name": "cnn_augmented", "augmenter": CodecAugmenter(aug_prob=0.5)},
    ]

    results_rows = []

    for regime in regimes:
        regime_name = regime["name"]
        augmenter = regime["augmenter"]
        print(f"\n" + "=" * 60)
        print(f"TRAINING REGIME: {regime_name.upper()}")
        print("=" * 60)

        train_dataset = AudioSpectrogramDataset(
            df_train,
            feature_type="log_mel",
            augmenter=augmenter,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = LightweightAudioCNN(base_channels=16, fc_hidden=64)
        trainer = CNNTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=checkpoint_dir,
            device=device,
            patience=5,
        )

        history = trainer.fit(num_epochs=epochs, model_name=regime_name)

        # Load best checkpoint for final evaluation
        best_ckpt_path = os.path.join(checkpoint_dir, f"{regime_name}_best.pt")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # 1. In-Language Test
        metrics_in = trainer.evaluate(test_in_loader)
        results_rows.append({
            "model": f"Model B ({regime_name})",
            "eval_split": "In-Language Test (Seen)",
            "auroc": metrics_in["auroc"],
            "eer": metrics_in["eer"],
            "balanced_acc": metrics_in["balanced_accuracy_tau0.5"],
            "macro_f1": metrics_in["macro_f1"],
        })

        # 2. Held-Out Dravidian Test
        metrics_drav = trainer.evaluate(test_drav_loader)
        results_rows.append({
            "model": f"Model B ({regime_name})",
            "eval_split": "Held-Out Dravidian",
            "auroc": metrics_drav["auroc"],
            "eer": metrics_drav["eer"],
            "balanced_acc": metrics_drav["balanced_accuracy_tau0.5"],
            "macro_f1": metrics_drav["macro_f1"],
        })

    results_df = pd.DataFrame(results_rows)
    out_table = os.path.join(results_dir, "cnn_results.csv")
    results_df.to_csv(out_table, index=False)
    print("\n" + "=" * 60)
    print("MODEL B RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    train_and_eval_cnn()
