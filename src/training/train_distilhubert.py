"""Training and comprehensive evaluation pipeline for DistilHuBERT (Model B4).

Features:
- Raw waveform loading (16 kHz, 48,000 samples)
- FP16 Mixed Precision training with GradScaler (GPU accelerated)
- AdamW optimizer with differential learning rates for backbone vs classifier
- Comprehensive benchmark evaluation across:
  * In-Language Seen (Indo-Aryan)
  * Held-Out 1 (Dravidian)
  * Held-Out 2 (Tibeto-Burman)
  * External Benchmark 1 (MLAAD)
  * External Benchmark 2 (ASVspoof 5)
  * Codec Robustness (Clean, MP3 64k/32k, Opus 24k/16k)
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_curve,
)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.audio_loader import AudioStandardizer
from src.models.distilhubert_classifier import DistilHuBERTClassifier
from src.robustness.ffmpeg_codec import CodecAugmenter
from src.evaluation.threshold_optimizer import find_optimal_threshold, evaluate_comprehensive_metrics
from src.evaluation.metrics import compute_bootstrap_ci


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """Computes Equal Error Rate (EER) and threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


class RawWaveformDataset(Dataset):
    """PyTorch Dataset that loads and standardizes raw 16 kHz waveforms."""

    def __init__(
        self,
        df: pd.DataFrame,
        codec_augmenter: Optional[CodecAugmenter] = None,
        codec_prob: float = 0.0,
        is_training: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.standardizer = AudioStandardizer(target_sample_rate=16000, target_duration_sec=3.0)
        self.codec_augmenter = codec_augmenter
        self.codec_prob = codec_prob
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]
        label = float(row["is_tts"])

        if os.path.exists(audio_path):
            audio = self.standardizer.process(audio_path)
        else:
            audio = np.zeros(48000, dtype=np.float32)

        # Codec augmentation during training
        if self.is_training and self.codec_augmenter is not None:
            audio, _ = self.codec_augmenter(audio)
            audio = self.standardizer.process(audio, orig_sr=16000)

        # Subtle amplitude perturbation during training
        if self.is_training and np.random.rand() < 0.3:
            gain = np.random.uniform(0.8, 1.2)
            audio = np.clip(audio * gain, -1.0, 1.0)

        tensor_waveform = torch.tensor(audio, dtype=torch.float32)
        tensor_label = torch.tensor([label], dtype=torch.float32)
        return tensor_waveform, tensor_label


def evaluate_model_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates predictions and true labels from a DataLoader."""
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            # Mixed precision inference if on CUDA
            if device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(waveforms)
            else:
                logits = model(waveforms)
            probs = torch.sigmoid(logits).cpu().squeeze(-1).numpy()
            all_scores.extend(probs.tolist())
            all_labels.extend(labels.squeeze(-1).numpy().tolist())
    return np.array(all_labels), np.array(all_scores)


def train_distilhubert(
    train_csv: str = "data/splits/train.csv",
    val_csv: str = "data/splits/val.csv",
    checkpoint_dir: str = "checkpoints",
    epochs: int = 10,
    batch_size: int = 16,
    lr_backbone: float = 3e-5,
    lr_head: float = 1e-3,
    weight_decay: float = 1e-4,
    use_fp16: bool = True,
    device_str: Optional[str] = None,
) -> DistilHuBERTClassifier:
    """Trains DistilHuBERT anti-spoofing model with FP16 mixed precision."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, "distilhubert_best.pt")

    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"\n========================================================")
    print(f" Initializing DistilHuBERT Training on {device} (FP16={use_fp16 and device.type == 'cuda'})")
    print(f"========================================================")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    print(f"Train samples: {len(df_train)}, Validation samples: {len(df_val)}")

    codec_aug = CodecAugmenter()
    train_dataset = RawWaveformDataset(df_train, codec_augmenter=codec_aug, codec_prob=0.3, is_training=True)
    val_dataset = RawWaveformDataset(df_val, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = DistilHuBERTClassifier(freeze_feature_extractor=True, dropout_rate=0.3).to(device)
    trainable_params, total_params = model.count_parameters()
    print(f"Model Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # Optimizer with differential learning rates
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ], weight_decay=weight_decay)

    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(use_fp16 and device.type == "cuda"))

    best_val_auroc = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for waveforms, labels in train_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_fp16 and device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(waveforms)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(waveforms)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item() * len(waveforms)

        train_loss /= len(train_dataset)
        epoch_sec = time.time() - start_time

        # Validation
        y_val_true, y_val_scores = evaluate_model_on_loader(model, val_loader, device)
        val_auroc = roc_auc_score(y_val_true, y_val_scores)
        val_eer, _ = compute_eer(y_val_true, y_val_scores)

        print(
            f"Epoch {epoch:2d}/{epochs} [{epoch_sec:.1f}s] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val AUROC: {val_auroc:.4f} | "
            f"Val EER: {val_eer * 100:.2f}%"
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  -> Checkpoint saved with AUROC: {val_auroc:.4f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nTraining Complete. Best Model achieved Val AUROC: {best_val_auroc:.4f} at epoch {best_epoch}.")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    return model


def evaluate_distilhubert_benchmarks(
    model: DistilHuBERTClassifier,
    device: torch.device,
    splits_dir: str = "data/splits",
    benchmarks_dir: str = "data/benchmarks",
    results_dir: str = "results/tables",
) -> pd.DataFrame:
    """Evaluates DistilHuBERT across linguistic splits, external benchmarks, and codecs."""
    os.makedirs(results_dir, exist_ok=True)
    df_val = pd.read_csv(os.path.join(splits_dir, "val.csv"))
    val_loader = DataLoader(RawWaveformDataset(df_val, is_training=False), batch_size=16, shuffle=False)
    y_val_true, y_val_scores = evaluate_model_on_loader(model, val_loader, device)

    # Find validation-optimized threshold tau*
    tau_star = find_optimal_threshold(y_val_true, y_val_scores, metric="balanced_accuracy")
    print(f"\nDistilHuBERT Validation-Optimized Threshold: tau* = {tau_star:.4f}")

    eval_tasks = [
        ("In-Language Seen (Indo-Aryan)", os.path.join(splits_dir, "test_in_lang.csv")),
        ("Held-Out 1 (Dravidian)", os.path.join(splits_dir, "test_held_out_dravidian.csv")),
        ("Held-Out 2 (Tibeto-Burman)", os.path.join(splits_dir, "test_held_out_tibeto_burman.csv")),
        ("External Benchmark 1 (MLAAD)", os.path.join(benchmarks_dir, "mlaad/manifest.csv")),
        ("External Benchmark 2 (ASVspoof 5)", os.path.join(benchmarks_dir, "asvspoof5/manifest.csv")),
    ]

    all_results = []

    for task_name, path in eval_tasks:
        if not os.path.exists(path):
            print(f"Skipping {task_name}: {path} not found.")
            continue

        df_task = pd.read_csv(path)
        loader = DataLoader(RawWaveformDataset(df_task, is_training=False), batch_size=16, shuffle=False)
        y_true, y_scores = evaluate_model_on_loader(model, loader, device)

        # Standard metrics
        auroc = roc_auc_score(y_true, y_scores)
        eer, _ = compute_eer(y_true, y_scores)
        ci_res = compute_bootstrap_ci(y_true, y_scores, n_resamples=500)
        ci_low = ci_res["auroc"]["lower"]
        ci_high = ci_res["auroc"]["upper"]

        # At default threshold 0.5
        acc_05 = accuracy_score(y_true, (y_scores >= 0.5).astype(int))
        prec_05 = precision_score(y_true, (y_scores >= 0.5).astype(int), zero_division=0)
        rec_05 = recall_score(y_true, (y_scores >= 0.5).astype(int), zero_division=0)
        f1_05 = f1_score(y_true, (y_scores >= 0.5).astype(int), zero_division=0)

        # At tuned threshold tau*
        acc_tau = accuracy_score(y_true, (y_scores >= tau_star).astype(int))
        prec_tau = precision_score(y_true, (y_scores >= tau_star).astype(int), zero_division=0)
        rec_tau = recall_score(y_true, (y_scores >= tau_star).astype(int), zero_division=0)
        f1_tau = f1_score(y_true, (y_scores >= tau_star).astype(int), zero_division=0)

        print(f"\nTask: {task_name}")
        print(f"  AUROC: {auroc:.4f} [95% CI: {ci_low:.4f} - {ci_high:.4f}]")
        print(f"  EER: {eer * 100:.2f}%")
        print(f"  Default (0.50): Acc={acc_05*100:.2f}%, Prec={prec_05*100:.2f}%, Rec={rec_05*100:.2f}%, F1={f1_05*100:.2f}%")
        print(f"  Tuned ({tau_star:.2f}): Acc={acc_tau*100:.2f}%, Prec={prec_tau*100:.2f}%, Rec={rec_tau*100:.2f}%, F1={f1_tau*100:.2f}%")

        all_results.append({
            "Evaluation_Domain": task_name,
            "Model": "DistilHuBERT (Self-Supervised)",
            "AUROC": auroc,
            "AUROC_95CI_Low": ci_low,
            "AUROC_95CI_High": ci_high,
            "EER_percent": eer * 100.0,
            "Acc_Default": acc_05 * 100.0,
            "Prec_Default": prec_05 * 100.0,
            "Rec_Default": rec_05 * 100.0,
            "F1_Default": f1_05 * 100.0,
            "Acc_Tuned": acc_tau * 100.0,
            "Prec_Tuned": prec_tau * 100.0,
            "Rec_Tuned": rec_tau * 100.0,
            "F1_Tuned": f1_tau * 100.0,
            "Optimal_Threshold": tau_star,
            "Num_Samples": len(df_task),
        })

    df_out = pd.DataFrame(all_results)
    out_path = os.path.join(results_dir, "distilhubert_comprehensive_evaluation.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nComprehensive results exported to: {out_path}")

    # Evaluate Codec Robustness
    evaluate_distilhubert_codecs(model, device, os.path.join(splits_dir, "test_in_lang.csv"), results_dir, tau_star)

    return df_out


def evaluate_distilhubert_codecs(
    model: DistilHuBERTClassifier,
    device: torch.device,
    test_in_csv: str,
    results_dir: str,
    tau_star: float = 0.5,
) -> pd.DataFrame:
    """Evaluates DistilHuBERT robustness under progressive lossy compression with parallel simulation."""
    import concurrent.futures
    from src.robustness.ffmpeg_codec import AudioCodecSimulator

    print("\n--- Evaluating DistilHuBERT Codec Robustness ---")
    df_test = pd.read_csv(test_in_csv)
    codec_conditions = [
        ("Clean", None),
        ("MP3 64k", "mp3_64k"),
        ("MP3 32k", "mp3_32k"),
        ("Opus 24k", "opus_24k"),
        ("Opus 16k", "opus_16k"),
    ]

    standardizer = AudioStandardizer(target_sample_rate=16000, target_duration_sec=3.0)
    simulator = AudioCodecSimulator()

    # Preload clean audio
    clean_audios, labels = [], []
    for _, row in df_test.iterrows():
        path = row["audio_path"]
        if os.path.exists(path):
            audio = standardizer.process(path)
        else:
            audio = np.zeros(48000, dtype=np.float32)
        clean_audios.append(audio)
        labels.append(float(row["is_tts"]))
    y_true = np.array(labels)

    def compress_one(args):
        audio, prof = args
        if prof is None:
            return audio
        return simulator.compress_decompress(audio, codec_profile=prof)

    codec_results = []
    model.eval()

    for cond_name, profile in codec_conditions:
        tasks = [(a, profile) for a in clean_audios]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            processed_audios = list(executor.map(compress_one, tasks))

        t_audios = torch.tensor(np.array(processed_audios), dtype=torch.float32).to(device)
        scores = []
        batch_sz = 32
        with torch.no_grad():
            for i in range(0, len(t_audios), batch_sz):
                batch = t_audios[i : i + batch_sz]
                if device.type == "cuda":
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        logits = model(batch)
                else:
                    logits = model(batch)
                scores.extend(torch.sigmoid(logits).cpu().squeeze(-1).numpy().tolist())

        y_scores = np.array(scores)
        auroc = roc_auc_score(y_true, y_scores)
        eer, _ = compute_eer(y_true, y_scores)
        acc_tau = accuracy_score(y_true, (y_scores >= tau_star).astype(int))

        print(f"Condition: {cond_name:8s} | AUROC: {auroc:.4f} | EER: {eer*100:.2f}% | Acc: {acc_tau*100:.2f}%")
        codec_results.append({
            "Condition": cond_name,
            "Model": "DistilHuBERT (Self-Supervised)",
            "AUROC": auroc,
            "EER_percent": eer * 100.0,
            "Accuracy_Tuned": acc_tau * 100.0,
            "Threshold": tau_star,
        })

    df_codec = pd.DataFrame(codec_results)
    codec_out_path = os.path.join(results_dir, "distilhubert_codec_robustness.csv")
    df_codec.to_csv(codec_out_path, index=False)
    print(f"Codec robustness results exported to: {codec_out_path}")
    return df_codec


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "checkpoints/distilhubert_best.pt"
    if os.path.exists(ckpt_path):
        print(f"\nLoading existing trained checkpoint from {ckpt_path}...")
        model = DistilHuBERTClassifier().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        model = train_distilhubert(epochs=10, batch_size=16, use_fp16=True)
    evaluate_distilhubert_benchmarks(model, device)
