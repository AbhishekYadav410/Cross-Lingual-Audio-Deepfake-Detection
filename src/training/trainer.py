"""PyTorch training pipeline for Model B (Lightweight CNN).

Includes:
1. AudioDataset with on-the-fly feature extraction and stochastic codec augmentation.
2. CNNTrainer with EarlyStopping and validation metric tracking (AUROC and EER).
"""

from typing import Dict, Any, Optional, List, Callable
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

from ..features.extractors import FeatureExtractor
from ..features.spec_augment import SpecAugment
from ..robustness.ffmpeg_codec import CodecAugmenter
from ..evaluation.metrics import evaluate_predictions
from .losses import LabelSmoothedBCEWithLogitsLoss


class AudioSpectrogramDataset(Dataset):
    """PyTorch Dataset loading audio samples, applying codec augmentation, and extracting features."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_type: str = "log_mel",
        augmenter: Optional[CodecAugmenter] = None,
        spec_augmenter: Optional[SpecAugment] = None,
        sample_rate: int = 16000,
        n_features: int = 80,
    ):
        self.df = df.reset_index(drop=True)
        self.feature_type = feature_type.lower()
        self.augmenter = augmenter
        self.spec_augmenter = spec_augmenter
        self.sample_rate = sample_rate
        self.extractor = FeatureExtractor(
            sample_rate=sample_rate,
            n_mels=n_features,
            n_lfcc=n_features,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        label = int(row["is_tts"])

        # Check if audio array/bytes or path is present
        if "audio_array" in row and isinstance(row["audio_array"], np.ndarray):
            audio = row["audio_array"]
        elif "audio_path" in row and os.path.exists(str(row["audio_path"])):
            import soundfile as sf
            audio, _ = sf.read(str(row["audio_path"]), dtype="float32")
        else:
            # Fallback for synthetic / dummy testing
            audio = np.zeros(self.sample_rate * 3, dtype=np.float32)

        # Apply stochastic codec augmentation if enabled
        if self.augmenter is not None:
            audio, _ = self.augmenter(audio)

        # Extract 2D spectral feature
        if self.feature_type == "log_mel":
            feature = self.extractor.extract_log_mel(audio)
        elif self.feature_type == "lfcc":
            feature = self.extractor.extract_lfcc(audio)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

        # Apply SpecAugment if enabled
        if self.spec_augmenter is not None:
            feature = self.spec_augmenter(feature)

        # Shape: (1, n_features, time_frames)
        feature_tensor = torch.from_numpy(feature).float().unsqueeze(0)
        label_tensor = torch.tensor([label], dtype=torch.float32)

        return {
            "feature": feature_tensor,
            "label": label_tensor,
            "language": str(row.get("language", "unknown")),
        }


class CNNTrainer:
    """Manages training loop, validation tracking, and model checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str = "checkpoints",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        patience: int = 7,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.criterion = LabelSmoothedBCEWithLogitsLoss(label_smoothing=0.05)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=25, eta_min=1e-5
        )
        self.patience = patience

    def train_epoch(self) -> float:
        """Executes one training epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            features = batch["feature"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluates model and computes AUROC, EER, and Balanced Accuracy."""
        target_loader = loader or self.val_loader
        self.model.eval()

        all_preds = []
        all_targets = []

        for batch in target_loader:
            features = batch["feature"].to(self.device)
            labels = batch["label"].cpu().numpy().ravel()

            probs = self.model.predict_proba(features).cpu().numpy().ravel()
            all_preds.extend(probs)
            all_targets.extend(labels)

        return evaluate_predictions(np.array(all_targets), np.array(all_preds))

    def fit(self, num_epochs: int = 25, model_name: str = "lightweight_cnn") -> Dict[str, List[float]]:
        """Executes full training with early stopping on validation AUROC."""
        history = {"train_loss": [], "val_auroc": [], "val_eer": []}
        best_val_auroc = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}_best.pt")

        print(f"Starting training on device: {self.device} for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            loss = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            self.scheduler.step()

            val_auroc = val_metrics["auroc"]
            val_eer = val_metrics["eer"]

            history["train_loss"].append(loss)
            history["val_auroc"].append(val_auroc)
            history["val_eer"].append(val_eer)

            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] "
                f"Loss: {loss:.4f} | Val AUROC: {val_auroc:.4f} | Val EER: {val_eer:.4f} | Time: {elapsed:.1f}s"
            )

            # Checkpoint on best validation AUROC
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_auroc": val_auroc,
                    "val_eer": val_eer,
                }, checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch} (AUROC: {best_val_auroc:.4f})")
                    break

        print(f"Training completed. Best model checkpoint saved to: {checkpoint_path}")
        return history
