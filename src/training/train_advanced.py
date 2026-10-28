"""Advanced training and dual held-out evaluation pipeline.

Trains:
1. LCNN (Light CNN with Max-Feature-Map MFM)
2. SE-ResNet (Squeeze-and-Excitation ResNet with Channel Attention)
With SpecAugment and Multi-Codec data augmentation.

Evaluates:
- Seen In-Language (Indo-Aryan)
- Held-Out 1: Dravidian (Tamil, Telugu, Kannada, Malayalam)
- Held-Out 2: Tibeto-Burman (Bodo, Manipuri)
At both default threshold (0.5) and validation-optimized threshold (tau*).
"""

from typing import Dict, Any, List, Tuple
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from src.models.lcnn import LCNN
from src.models.se_resnet import SEResNet
from src.models.classical_baseline import ClassicalBaseline
from src.models.ensemble import ForensicEnsemble
from src.features.classical_features import ClassicalFeatureExtractor
from src.features.spec_augment import SpecAugment
from src.robustness.ffmpeg_codec import CodecAugmenter
from src.training.trainer import AudioSpectrogramDataset, CNNTrainer
from src.evaluation.threshold_optimizer import find_optimal_threshold, evaluate_comprehensive_metrics
from src.evaluation.metrics import compute_bootstrap_ci


def extract_features_df(df: pd.DataFrame, extractor: ClassicalFeatureExtractor) -> Tuple[np.ndarray, np.ndarray]:
    import soundfile as sf
    X, y = [], []
    for _, row in df.iterrows():
        path = row["audio_path"]
        if os.path.exists(path):
            audio, _ = sf.read(path, dtype="float32")
        else:
            audio = np.zeros(48000, dtype=np.float32)
        X.append(extractor.extract_vector(audio))
        y.append(int(row["is_tts"]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


def train_and_evaluate_all_models(
    splits_dir: str = "data/splits",
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results/tables",
    epochs: int = 15,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    device_obj = torch.device(device)

    # 1. Load all 5 splits
    print("\nLoading linguistic splits...")
    df_train = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(splits_dir, "val.csv"))
    df_test_in = pd.read_csv(os.path.join(splits_dir, "test_in_lang.csv"))
    df_test_drav = pd.read_csv(os.path.join(splits_dir, "test_held_out_dravidian.csv"))
    df_test_tb = pd.read_csv(os.path.join(splits_dir, "test_held_out_tibeto_burman.csv"))

    print(f"Train samples (Indo-Aryan): {len(df_train)}")
    print(f"Validation samples (Indo-Aryan): {len(df_val)}")
    print(f"In-Language Test samples (Indo-Aryan): {len(df_test_in)}")
    print(f"Held-Out Target 1 (Dravidian): {len(df_test_drav)}")
    print(f"Held-Out Target 2 (Tibeto-Burman): {len(df_test_tb)}")

    # DataLoaders for evaluation
    val_dataset = AudioSpectrogramDataset(df_val, feature_type="log_mel")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_in_dataset = AudioSpectrogramDataset(df_test_in, feature_type="log_mel")
    test_in_loader = DataLoader(test_in_dataset, batch_size=batch_size, shuffle=False)

    test_drav_dataset = AudioSpectrogramDataset(df_test_drav, feature_type="log_mel")
    test_drav_loader = DataLoader(test_drav_dataset, batch_size=batch_size, shuffle=False)

    test_tb_dataset = AudioSpectrogramDataset(df_test_tb, feature_type="log_mel")
    test_tb_loader = DataLoader(test_tb_dataset, batch_size=batch_size, shuffle=False)

    eval_splits = [
        ("In-Language Seen (Indo-Aryan)", test_in_loader, np.array(df_test_in["is_tts"])),
        ("Held-Out 1 (Dravidian)", test_drav_loader, np.array(df_test_drav["is_tts"])),
        ("Held-Out 2 (Tibeto-Burman)", test_tb_loader, np.array(df_test_tb["is_tts"])),
    ]

    # Models configuration
    model_configs = [
        {
            "name": "lcnn_mfm_augmented",
            "model_fn": lambda: LCNN(in_channels=1, dropout_rate=0.3),
            "spec_aug": SpecAugment(freq_mask_param=12, time_mask_param=16, prob=0.5),
            "codec_aug": CodecAugmenter(aug_prob=0.5),
            "display_name": "Model B2 (LCNN-MFM)",
        },
        {
            "name": "se_resnet_augmented",
            "model_fn": lambda: SEResNet(in_channels=1, base_channels=24, dropout_rate=0.3),
            "spec_aug": SpecAugment(freq_mask_param=12, time_mask_param=16, prob=0.5),
            "codec_aug": CodecAugmenter(aug_prob=0.5),
            "display_name": "Model B3 (SE-ResNet-18)",
        },
    ]

    all_results = []
    trained_models = {}
    val_predictions = {}

    # Train advanced neural models
    for cfg in model_configs:
        model_name = cfg["name"]
        display_name = cfg["display_name"]
        print(f"\n" + "=" * 60)
        print(f"TRAINING: {display_name} ({model_name})")
        print("=" * 60)

        train_dataset = AudioSpectrogramDataset(
            df_train,
            feature_type="log_mel",
            augmenter=cfg["codec_aug"],
            spec_augmenter=cfg["spec_aug"],
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = cfg["model_fn"]()
        trainer = CNNTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=checkpoint_dir,
            device=device,
            patience=5,
        )

        history = trainer.fit(num_epochs=epochs, model_name=model_name)

        # Load best weights
        best_ckpt = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
        ckpt = torch.load(best_ckpt, map_location=device_obj)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        trained_models[model_name] = model

        # Get validation predictions for threshold optimization
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                feats = batch["feature"].to(device_obj)
                probs = model.predict_proba(feats).cpu().numpy().ravel()
                val_preds.extend(probs)
                val_targets.extend(batch["label"].numpy().ravel())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_predictions[model_name] = val_preds

        # Find optimal validation threshold tau*
        tau_star = find_optimal_threshold(val_targets, val_preds, metric="balanced_accuracy")
        print(f"Optimal Validation Threshold tau* for {display_name}: {tau_star:.4f}")

        # Evaluate across the 3 evaluation splits
        for split_label, loader, y_true in eval_splits:
            test_preds = []
            with torch.no_grad():
                for batch in loader:
                    feats = batch["feature"].to(device_obj)
                    probs = model.predict_proba(feats).cpu().numpy().ravel()
                    test_preds.extend(probs)
            test_preds = np.array(test_preds)

            # Evaluate at default threshold (0.5)
            m_def = evaluate_comprehensive_metrics(y_true, test_preds, threshold=0.5)
            # Evaluate at optimal threshold (tau*)
            m_opt = evaluate_comprehensive_metrics(y_true, test_preds, threshold=tau_star)
            # Bootstrap CI
            ci = compute_bootstrap_ci(y_true, test_preds, n_resamples=500)

            all_results.append({
                "model": display_name,
                "split": split_label,
                "threshold_mode": "Default (0.50)",
                "applied_threshold": 0.50,
                "accuracy": m_def["accuracy"],
                "balanced_acc": m_def["balanced_accuracy"],
                "precision": m_def["precision"],
                "recall": m_def["recall"],
                "specificity": m_def["specificity"],
                "macro_f1": m_def["macro_f1"],
                "auroc": m_def["auroc"],
                "auroc_ci": f"[{ci['auroc']['lower']:.3f}, {ci['auroc']['upper']:.3f}]",
                "eer": m_def["eer"],
                "eer_ci": f"[{ci['eer']['lower']:.3f}, {ci['eer']['upper']:.3f}]",
            })

            all_results.append({
                "model": display_name,
                "split": split_label,
                "threshold_mode": f"Tuned (tau*={tau_star:.2f})",
                "applied_threshold": tau_star,
                "accuracy": m_opt["accuracy"],
                "balanced_acc": m_opt["balanced_accuracy"],
                "precision": m_opt["precision"],
                "recall": m_opt["recall"],
                "specificity": m_opt["specificity"],
                "macro_f1": m_opt["macro_f1"],
                "auroc": m_opt["auroc"],
                "auroc_ci": f"[{ci['auroc']['lower']:.3f}, {ci['auroc']['upper']:.3f}]",
                "eer": m_opt["eer"],
                "eer_ci": f"[{ci['eer']['lower']:.3f}, {ci['eer']['upper']:.3f}]",
            })

    # ==========================================
    # Train Model A (LFCC-SVM) & Hybrid Ensemble
    # ==========================================
    print("\n" + "=" * 60)
    print("TRAINING MODEL A & EVALUATING HYBRID FORENSIC ENSEMBLE")
    print("=" * 60)

    extractor_c = ClassicalFeatureExtractor(sample_rate=16000, n_coefficients=30)
    X_train_c, y_train_c = extract_features_df(df_train, extractor_c)
    X_val_c, y_val_c = extract_features_df(df_val, extractor_c)

    svm_model = ClassicalBaseline(model_type="svm_rbf")
    svm_model.fit(X_train_c, y_train_c)
    svm_model.save(os.path.join(checkpoint_dir, "model_a_svm_rbf.joblib"))

    val_probs_svm = svm_model.predict_proba(X_val_c)[:, 1]
    tau_star_svm = find_optimal_threshold(y_val_c, val_probs_svm, metric="balanced_accuracy")

    # Ensemble setup: Blending LFCC-SVM + LCNN + SE-ResNet
    ensemble = ForensicEnsemble(weights={"svm": 0.34, "lcnn": 0.33, "se_resnet": 0.33})

    val_ensemble_probs = ensemble.predict_proba({
        "svm": val_probs_svm,
        "lcnn": val_predictions["lcnn_mfm_augmented"],
        "se_resnet": val_predictions["se_resnet_augmented"],
    })
    tau_star_ens = find_optimal_threshold(y_val_c, val_ensemble_probs, metric="balanced_accuracy")

    # Evaluate Model A and Hybrid Ensemble on the 3 splits
    for split_label, df_split, loader in [
        ("In-Language Seen (Indo-Aryan)", df_test_in, test_in_loader),
        ("Held-Out 1 (Dravidian)", df_test_drav, test_drav_loader),
        ("Held-Out 2 (Tibeto-Burman)", df_test_tb, test_tb_loader),
    ]:
        X_sp_c, y_sp = extract_features_df(df_split, extractor_c)
        probs_svm = svm_model.predict_proba(X_sp_c)[:, 1]

        # Model A results
        m_svm = evaluate_comprehensive_metrics(y_sp, probs_svm, threshold=tau_star_svm)
        ci_svm = compute_bootstrap_ci(y_sp, probs_svm, n_resamples=500)

        all_results.append({
            "model": "Model A (RBF-SVM)",
            "split": split_label,
            "threshold_mode": f"Tuned (tau*={tau_star_svm:.2f})",
            "applied_threshold": tau_star_svm,
            "accuracy": m_svm["accuracy"],
            "balanced_acc": m_svm["balanced_accuracy"],
            "precision": m_svm["precision"],
            "recall": m_svm["recall"],
            "specificity": m_svm["specificity"],
            "macro_f1": m_svm["macro_f1"],
            "auroc": m_svm["auroc"],
            "auroc_ci": f"[{ci_svm['auroc']['lower']:.3f}, {ci_svm['auroc']['upper']:.3f}]",
            "eer": m_svm["eer"],
            "eer_ci": f"[{ci_svm['eer']['lower']:.3f}, {ci_svm['eer']['upper']:.3f}]",
        })

        # Get neural predictions
        with torch.no_grad():
            preds_lcnn, preds_se = [], []
            for batch in loader:
                feats = batch["feature"].to(device_obj)
                preds_lcnn.extend(trained_models["lcnn_mfm_augmented"].predict_proba(feats).cpu().numpy().ravel())
                preds_se.extend(trained_models["se_resnet_augmented"].predict_proba(feats).cpu().numpy().ravel())

        # Ensemble predictions
        probs_ens = ensemble.predict_proba({
            "svm": probs_svm,
            "lcnn": np.array(preds_lcnn),
            "se_resnet": np.array(preds_se),
        })

        m_ens = evaluate_comprehensive_metrics(y_sp, probs_ens, threshold=tau_star_ens)
        ci_ens = compute_bootstrap_ci(y_sp, probs_ens, n_resamples=500)

        all_results.append({
            "model": "Hybrid Forensic Ensemble",
            "split": split_label,
            "threshold_mode": f"Tuned (tau*={tau_star_ens:.2f})",
            "applied_threshold": tau_star_ens,
            "accuracy": m_ens["accuracy"],
            "balanced_acc": m_ens["balanced_accuracy"],
            "precision": m_ens["precision"],
            "recall": m_ens["recall"],
            "specificity": m_ens["specificity"],
            "macro_f1": m_ens["macro_f1"],
            "auroc": m_ens["auroc"],
            "auroc_ci": f"[{ci_ens['auroc']['lower']:.3f}, {ci_ens['auroc']['upper']:.3f}]",
            "eer": m_ens["eer"],
            "eer_ci": f"[{ci_ens['eer']['lower']:.3f}, {ci_ens['eer']['upper']:.3f}]",
        })

    # Save master results table
    results_df = pd.DataFrame(all_results)
    out_csv = os.path.join(results_dir, "dual_held_out_comprehensive_results.csv")
    results_df.to_csv(out_csv, index=False)
    print("\n" + "=" * 80)
    print("DUAL HELD-OUT COMPREHENSIVE BENCHMARK RESULTS")
    print("=" * 80)
    print(results_df[["model", "split", "threshold_mode", "accuracy", "precision", "recall", "macro_f1", "auroc", "eer"]].to_string(index=False))

    return results_df


if __name__ == "__main__":
    train_and_evaluate_all_models()
