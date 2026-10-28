"""Acquires standardized evaluation sets for MLAAD and ASVspoof 5,
and evaluates trained models (Model A, B1, B2, B3, and Hybrid Ensemble).
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import time
import json
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import concurrent.futures
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
import fsspec
import pyarrow.parquet as pq

from src.preprocessing.audio_loader import AudioStandardizer
from src.features.extractors import FeatureExtractor
from src.features.classical_features import ClassicalFeatureExtractor
from src.models.lightweight_cnn import LightweightAudioCNN
from src.models.lcnn import LCNN
from src.models.se_resnet import SEResNet
from src.models.classical_baseline import ClassicalBaseline
from src.models.ensemble import ForensicEnsemble
from src.evaluation.metrics import evaluate_predictions, compute_bootstrap_ci


def download_and_process_mlaad_file(args):
    repo_id, f, is_tts, out_dir, idx = args
    try:
        local_path = hf_hub_download(repo_id, f, repo_type="dataset")
        audio, sr = sf.read(local_path, dtype="float32")
        standardizer = AudioStandardizer(target_sample_rate=16000, target_duration_sec=3.0)
        std_audio = standardizer.process(audio, orig_sr=sr)
        prefix = "fake" if is_tts == 1 else "real"
        out_name = f"mlaad_{prefix}_{idx:04d}.wav"
        out_wav = os.path.join(out_dir, out_name)
        sf.write(out_wav, std_audio, 16000, subtype="PCM_16")
        return {
            "sample_id": out_name,
            "audio_path": out_wav,
            "is_tts": is_tts,
            "source_file": f,
            "benchmark": "MLAAD",
        }
    except Exception as e:
        print(f"Error on {f}: {e}")
        return None


def prepare_mlaad_benchmark(out_dir: str, n_samples_per_class: int = 150) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    manifest_file = os.path.join(out_dir, "manifest.csv")
    if os.path.exists(manifest_file):
        df_cached = pd.read_csv(manifest_file)
        if len(df_cached) >= n_samples_per_class * 2:
            print(f"Loading cached MLAAD manifest from {manifest_file}...")
            return df_cached

    print(f"Preparing MLAAD benchmark ({n_samples_per_class * 2} samples, 16 parallel threads)...", flush=True)
    api = HfApi()
    repo_id = "mueller91/MLAAD-tiny"
    all_files = api.list_repo_files(repo_id, repo_type="dataset")

    orig_files = [f for f in all_files if f.startswith("original/") and f.endswith(".wav")]
    fake_files = [f for f in all_files if f.startswith("fake/") and f.endswith(".wav")]

    np.random.seed(42)
    selected_orig = list(np.random.choice(orig_files, size=min(n_samples_per_class, len(orig_files)), replace=False))
    selected_fake = list(np.random.choice(fake_files, size=min(n_samples_per_class, len(fake_files)), replace=False))

    tasks = []
    idx = 0
    for f in selected_orig:
        tasks.append((repo_id, f, 0, out_dir, idx))
        idx += 1
    for f in selected_fake:
        tasks.append((repo_id, f, 1, out_dir, idx))
        idx += 1

    records = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(download_and_process_mlaad_file, tasks))
        for r in results:
            if r is not None:
                records.append(r)

    print(f"Downloaded and standardized {len(records)} MLAAD clips in {time.time() - t0:.2f}s", flush=True)
    df = pd.DataFrame(records)
    df.to_csv(manifest_file, index=False)
    print(f"Saved MLAAD benchmark manifest with {len(df)} samples to {manifest_file}", flush=True)
    return df


def prepare_asvspoof5_benchmark(out_dir: str, n_samples_per_class: int = 150) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    manifest_file = os.path.join(out_dir, "manifest.csv")
    if os.path.exists(manifest_file):
        df_cached = pd.read_csv(manifest_file)
        if len(df_cached) >= n_samples_per_class * 2:
            print(f"Loading cached ASVspoof 5 manifest from {manifest_file}...")
            return df_cached

    print(f"Preparing ASVspoof 5 benchmark ({n_samples_per_class * 2} samples)...", flush=True)
    repo_id = "SpeechAntiSpoofingBenchmarks/ASVspoof5"
    url = hf_hub_url(repo_id, "data/test-00000-of-00200.parquet", repo_type="dataset")

    standardizer = AudioStandardizer(target_sample_rate=16000, target_duration_sec=3.0)
    records = []
    bonafide_needed = n_samples_per_class
    spoof_needed = n_samples_per_class

    fs, path = fsspec.core.url_to_fs(url)
    with fs.open(path, "rb") as f:
        pf = pq.ParquetFile(f)
        for rg_idx in range(pf.num_row_groups):
            if bonafide_needed <= 0 and spoof_needed <= 0:
                break
            table = pf.read_row_group(rg_idx)
            df_rg = table.to_pandas()
            for i, row in df_rg.iterrows():
                label = int(row["label"])
                if label == 0 and bonafide_needed > 0:
                    raw_bytes = row["audio"]["bytes"]
                    audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
                    std_audio = standardizer.process(audio, orig_sr=sr)
                    out_name = f"asv5_real_{len(records):04d}.wav"
                    out_wav = os.path.join(out_dir, out_name)
                    sf.write(out_wav, std_audio, 16000, subtype="PCM_16")
                    records.append({
                        "sample_id": out_name,
                        "audio_path": out_wav,
                        "is_tts": 0,
                        "source_file": row["path"],
                        "benchmark": "ASVspoof5",
                    })
                    bonafide_needed -= 1
                elif label == 1 and spoof_needed > 0:
                    raw_bytes = row["audio"]["bytes"]
                    audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
                    std_audio = standardizer.process(audio, orig_sr=sr)
                    out_name = f"asv5_spoof_{len(records):04d}.wav"
                    out_wav = os.path.join(out_dir, out_name)
                    sf.write(out_wav, std_audio, 16000, subtype="PCM_16")
                    records.append({
                        "sample_id": out_name,
                        "audio_path": out_wav,
                        "is_tts": 1,
                        "source_file": row["path"],
                        "benchmark": "ASVspoof5",
                    })
                    spoof_needed -= 1

    df = pd.DataFrame(records)
    df.to_csv(manifest_file, index=False)
    print(f"Saved ASVspoof 5 benchmark manifest with {len(df)} samples to {manifest_file}", flush=True)
    return df


def evaluate_models_on_benchmark(df: pd.DataFrame, benchmark_name: str, checkpoint_dir: str = "checkpoints") -> pd.DataFrame:
    print(f"\n" + "=" * 60)
    print(f"EVALUATING TRAINED MODELS ON: {benchmark_name.upper()}")
    print("=" * 60)

    device = torch.device("cpu")
    extractor = FeatureExtractor(sample_rate=16000, n_mels=80)
    classical_extractor = ClassicalFeatureExtractor(sample_rate=16000, n_coefficients=30)

    # 1. Load Audio
    audios, labels = [], []
    for _, row in df.iterrows():
        a, _ = sf.read(row["audio_path"], dtype="float32")
        audios.append(a)
        labels.append(int(row["is_tts"]))
    y_true = np.array(labels)

    # 2. Extract features
    specs = np.array([extractor.extract_log_mel(a) for a in audios])
    t_specs = torch.from_numpy(specs).unsqueeze(1).float().to(device)
    X_classical = np.array([classical_extractor.extract_vector(a) for a in audios])

    # 3. Load Models
    svm_model = ClassicalBaseline(model_type="svm_rbf")
    svm_model.load(os.path.join(checkpoint_dir, "model_a_svm_rbf.joblib"))

    cnn_clean = LightweightAudioCNN(base_channels=16, fc_hidden=64).to(device)
    cnn_clean.load_state_dict(torch.load(os.path.join(checkpoint_dir, "cnn_clean_best.pt"), map_location=device)["model_state_dict"])
    cnn_clean.eval()

    lcnn_model = LCNN(in_channels=1, dropout_rate=0.3).to(device)
    lcnn_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "lcnn_mfm_augmented_best.pt"), map_location=device)["model_state_dict"])
    lcnn_model.eval()

    se_resnet = SEResNet(in_channels=1, base_channels=24, dropout_rate=0.3).to(device)
    se_resnet.load_state_dict(torch.load(os.path.join(checkpoint_dir, "se_resnet_augmented_best.pt"), map_location=device)["model_state_dict"])
    se_resnet.eval()

    ensemble = ForensicEnsemble(weights={"svm": 0.34, "lcnn": 0.33, "se_resnet": 0.33})

    # 4. Predictions
    with torch.no_grad():
        p_svm = svm_model.predict_proba(X_classical)[:, 1]
        p_cnn = cnn_clean.predict_proba(t_specs).cpu().numpy().ravel()
        p_lcnn = lcnn_model.predict_proba(t_specs).cpu().numpy().ravel()
        p_se = se_resnet.predict_proba(t_specs).cpu().numpy().ravel()
        p_ens = ensemble.predict_proba({"svm": p_svm, "lcnn": p_lcnn, "se_resnet": p_se})

    models = [
        ("Model A (RBF-SVM)", p_svm),
        ("Model B1 (Baseline 2D-CNN)", p_cnn),
        ("Model B2 (LCNN-MFM)", p_lcnn),
        ("Model B3 (SE-ResNet-18)", p_se),
        ("Hybrid Forensic Ensemble", p_ens),
    ]

    results = []
    for name, p_score in models:
        m = evaluate_predictions(y_true, p_score)
        ci = compute_bootstrap_ci(y_true, p_score, n_resamples=500)
        y_pred = (p_score >= 0.50).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        accuracy = (tp + tn) / max(1, len(y_true))

        results.append({
            "benchmark": benchmark_name,
            "model": name,
            "accuracy": round(accuracy, 4),
            "balanced_acc": round(m["balanced_accuracy_tau0.5"], 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "macro_f1": round(m["macro_f1"], 4),
            "auroc": round(m["auroc"], 4),
            "auroc_ci": f"[{ci['auroc']['lower']:.3f}, {ci['auroc']['upper']:.3f}]",
            "eer": round(m["eer"], 4),
            "eer_ci": f"[{ci['eer']['lower']:.3f}, {ci['eer']['upper']:.3f}]",
        })

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    return res_df


def main():
    mlaad_dir = "data/benchmarks/mlaad"
    asv5_dir = "data/benchmarks/asvspoof5"
    tables_dir = "results/tables"
    os.makedirs(tables_dir, exist_ok=True)

    # 1. Prepare benchmarks
    df_mlaad = prepare_mlaad_benchmark(mlaad_dir, n_samples_per_class=150)
    df_asv5 = prepare_asvspoof5_benchmark(asv5_dir, n_samples_per_class=150)

    # 2. Evaluate on MLAAD
    res_mlaad = evaluate_models_on_benchmark(df_mlaad, benchmark_name="MLAAD")
    res_mlaad.to_csv(os.path.join(tables_dir, "benchmark_mlaad_results.csv"), index=False)

    # 3. Evaluate on ASVspoof 5
    res_asv5 = evaluate_models_on_benchmark(df_asv5, benchmark_name="ASVspoof5")
    res_asv5.to_csv(os.path.join(tables_dir, "benchmark_asvspoof5_results.csv"), index=False)

    # 4. Combined benchmark table
    combined_df = pd.concat([res_mlaad, res_asv5], ignore_index=True)
    combined_csv = os.path.join(tables_dir, "external_benchmarks_combined.csv")
    combined_df.to_csv(combined_csv, index=False)
    print("\n" + "=" * 70)
    print("EXTERNAL BENCHMARKS COMBINED RESULTS")
    print("=" * 70)
    print(combined_df[["benchmark", "model", "accuracy", "precision", "recall", "macro_f1", "auroc", "eer"]].to_string(index=False))


if __name__ == "__main__":
    main()
