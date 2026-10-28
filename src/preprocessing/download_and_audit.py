"""Dataset acquisition, data quality auditing, and leak-free split generation.

Loads `SherryT997/IndicTTS-Deepfake-Challenge-Data` from Hugging Face,
audits language distributions and class balances, saves standardized 16kHz
mono WAV files, and executes prompt-disjoint splitting.
"""

from typing import Dict, Any, Optional
import os
import json
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from .audio_loader import AudioStandardizer
from .split_generator import create_dual_linguistic_splits, DRAVIDIAN_LANGUAGES, TIBETO_BURMAN_LANGUAGES


def generate_synthetic_dataset(n_samples_per_lang: int = 40) -> pd.DataFrame:
    """Generates synthetic audio dataset for deterministic dry-run verification."""
    languages = [
        "Hindi", "Marathi", "Bengali", "Gujarati", "Odia", "Assamese",
        "Dogri", "Nepali", "Sanskrit", "Bodo", "Manipuri", "English",
        "Tamil", "Telugu", "Kannada", "Malayalam"
    ]
    records = []
    sr = 16000
    duration_samples = sr * 3

    print(f"Generating synthetic verification dataset ({len(languages)} languages, {n_samples_per_lang} samples/lang)...")

    for lang in languages:
        for i in range(n_samples_per_lang):
            is_tts = i % 2
            # Deterministic synthetic audio
            t = np.linspace(0, 3.0, duration_samples, endpoint=False)
            freq = 200 + (hash(lang) % 400) + (100 if is_tts else 0)
            audio = (0.3 * np.sin(2 * np.pi * freq * t) + 0.05 * np.random.randn(duration_samples)).astype(np.float32)

            records.append({
                "id": f"{lang.lower()}_{'tts' if is_tts else 'real'}_{i:04d}",
                "language": lang,
                "is_tts": is_tts,
                "text": f"Sample transcription text for {lang} utterance number {i // 2}",
                "audio_array": audio,
                "sample_rate": sr,
            })

    return pd.DataFrame(records)


def run_audit(
    df: pd.DataFrame,
    processed_dir: str,
    manifest_dir: str,
    splits_dir: str,
) -> Dict[str, Any]:
    """Runs data quality checks, saves standardized audio, and creates splits."""
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    standardizer = AudioStandardizer(target_sample_rate=16000, target_duration_sec=3.0)

    print("\n" + "=" * 60)
    print("DATASET QUALITY & INTEGRITY AUDIT")
    print("=" * 60)

    total_samples = len(df)
    lang_counts = df["language"].value_counts().to_dict()
    tts_counts = df["is_tts"].value_counts().to_dict()

    print(f"Total labeled samples: {total_samples}")
    print(f"Real speech (is_tts=0): {tts_counts.get(0, 0)} ({tts_counts.get(0, 0) / total_samples * 100:.1f}%)")
    print(f"Synthetic speech (is_tts=1): {tts_counts.get(1, 0)} ({tts_counts.get(1, 0) / total_samples * 100:.1f}%)")
    print(f"Number of languages: {len(lang_counts)}")

    audio_paths = []
    print("\nStandardizing and caching audio waveforms to disk...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio"):
        sample_id = row["id"]
        out_wav = os.path.join(processed_dir, f"{sample_id}.wav")

        if os.path.exists(out_wav):
            audio_paths.append(out_wav)
            continue

        if "audio_array" in row and isinstance(row["audio_array"], np.ndarray):
            raw_audio = row["audio_array"]
            sr = row.get("sample_rate", 16000)
            standardized = standardizer.process(raw_audio, orig_sr=sr)
        elif "audio" in row and isinstance(row["audio"], dict):
            # Hugging Face Audio feature format
            raw_audio = np.array(row["audio"]["array"], dtype=np.float32)
            sr = row["audio"]["sampling_rate"]
            standardized = standardizer.process(raw_audio, orig_sr=sr)
        else:
            standardized = np.zeros(48000, dtype=np.float32)

        sf.write(out_wav, standardized, 16000, subtype="PCM_16")
        audio_paths.append(out_wav)

    df["audio_path"] = audio_paths

    # Generate master manifest
    master_manifest_path = os.path.join(manifest_dir, "master_manifest.jsonl")
    manifest_records = []
    for _, row in df.iterrows():
        manifest_records.append({
            "sample_id": row["id"],
            "audio_path": row["audio_path"],
            "language": row["language"],
            "is_dravidian": str(row["language"]).lower() in DRAVIDIAN_LANGUAGES,
            "is_tibeto_burman": str(row["language"]).lower() in TIBETO_BURMAN_LANGUAGES,
            "label_is_tts": int(row["is_tts"]),
            "text": str(row.get("text", "")),
        })

    with open(master_manifest_path, "w", encoding="utf-8") as f:
        for r in manifest_records:
            f.write(json.dumps(r) + "\n")

    print(f"Saved master manifest to: {master_manifest_path}")

    # Generate linguistic splits
    print("\nGenerating text-disjoint linguistic splits...")
    splits = create_dual_linguistic_splits(df, output_dir=splits_dir)

    audit_summary = {
        "total_samples": total_samples,
        "languages": lang_counts,
        "class_balance": tts_counts,
        "split_counts": {k: len(v) for k, v in splits.items()},
    }

    summary_path = os.path.join(manifest_dir, "audit_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"Saved audit summary to: {summary_path}")
    return audit_summary


def main():
    parser = argparse.ArgumentParser(description="Audit and partition IndicTTS Deepfake Dataset")
    parser.add_argument("--dry_run", action="store_true", help="Generate synthetic test dataset for fast dry run")
    parser.add_argument("--samples_per_lang", type=int, default=50, help="Samples per language in dry run mode")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--manifest_dir", type=str, default="data/manifests")
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    args = parser.parse_args()

    if args.dry_run:
        df = generate_synthetic_dataset(n_samples_per_lang=args.samples_per_lang)
    else:
        print("Loading SherryT997/IndicTTS-Deepfake-Challenge-Data from Hugging Face...")
        from datasets import load_dataset
        hf_dataset = load_dataset("SherryT997/IndicTTS-Deepfake-Challenge-Data", split="train")
        df = hf_dataset.to_pandas()

    run_audit(
        df=df,
        processed_dir=args.processed_dir,
        manifest_dir=args.manifest_dir,
        splits_dir=args.splits_dir,
    )


if __name__ == "__main__":
    main()
