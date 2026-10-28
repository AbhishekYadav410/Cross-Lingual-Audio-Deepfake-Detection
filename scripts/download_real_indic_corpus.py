"""Downloads and standardizes real audio clips across all 16 Indian languages from Hugging Face.

Streams row groups 0 and 1 for 16 target shards (200 samples per language = 3,200 samples total)
using HTTP range requests, avoiding downloading 16GB of parquet files.
Standardizes audio to 16kHz mono 3.0s PCM WAV files and creates text-disjoint dual held-out splits.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import json
import time
import hashlib
import concurrent.futures
import soundfile as sf
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import fsspec
from huggingface_hub import hf_hub_url

from src.preprocessing.audio_loader import AudioStandardizer
from src.preprocessing.split_generator import (
    create_dual_linguistic_splits,
    DRAVIDIAN_LANGUAGES,
    TIBETO_BURMAN_LANGUAGES,
)

# 16 Target Shards corresponding to each language
TARGET_SHARDS = [
    {"shard_idx": 0, "primary_lang": "Assamese", "family": "Indo-Aryan"},
    {"shard_idx": 3, "primary_lang": "Bengali", "family": "Indo-Aryan"},
    {"shard_idx": 5, "primary_lang": "Bodo", "family": "Tibeto-Burman"},
    {"shard_idx": 7, "primary_lang": "Dogri", "family": "Indo-Aryan"},
    {"shard_idx": 9, "primary_lang": "Kannada", "family": "Dravidian"},
    {"shard_idx": 12, "primary_lang": "Malayalam", "family": "Dravidian"},
    {"shard_idx": 14, "primary_lang": "Marathi", "family": "Indo-Aryan"},
    {"shard_idx": 16, "primary_lang": "Sanskrit", "family": "Indo-Aryan"},
    {"shard_idx": 18, "primary_lang": "Nepali", "family": "Indo-Aryan"},
    {"shard_idx": 21, "primary_lang": "English", "family": "Indo-Aryan"},
    {"shard_idx": 23, "primary_lang": "Telugu", "family": "Dravidian"},
    {"shard_idx": 25, "primary_lang": "Hindi", "family": "Indo-Aryan"},
    {"shard_idx": 27, "primary_lang": "Odia", "family": "Indo-Aryan"},
    {"shard_idx": 29, "primary_lang": "Manipuri", "family": "Tibeto-Burman"},
    {"shard_idx": 31, "primary_lang": "Gujarati", "family": "Indo-Aryan"},
    {"shard_idx": 33, "primary_lang": "Tamil", "family": "Dravidian"},
]


def process_shard(shard_info: dict, processed_dir: str, num_row_groups: int = 2) -> list:
    shard_idx = shard_info["shard_idx"]
    primary_lang = shard_info["primary_lang"]
    repo_id = "SherryT997/IndicTTS-Deepfake-Challenge-Data"
    filename = f"data/train-{shard_idx:05d}-of-00035.parquet"
    url = hf_hub_url(repo_id, filename, repo_type="dataset")

    standardizer = AudioStandardizer(target_sample_rate=16000, target_duration_sec=3.0)
    records = []

    print(f"[{primary_lang}] Connecting to Shard {shard_idx:02d}...", flush=True)
    fs, path = fsspec.core.url_to_fs(url)
    with fs.open(path, "rb") as f:
        pf = pq.ParquetFile(f)
        for rg_idx in range(min(num_row_groups, pf.num_row_groups)):
            t0 = time.time()
            table = pf.read_row_group(rg_idx)
            df = table.select(["id", "language", "is_tts", "text"]).to_pandas()
            audio_col = table.column("audio")

            saved_count = 0
            for i in range(len(table)):
                row_id = str(df.iloc[i]["id"])
                lang = str(df.iloc[i]["language"])
                is_tts = int(df.iloc[i]["is_tts"])
                text = str(df.iloc[i]["text"])
                out_wav = os.path.join(processed_dir, f"{row_id}.wav")

                if not os.path.exists(out_wav):
                    audio_dict = audio_col[i].as_py()
                    raw_bytes = audio_dict["bytes"]
                    audio_data, sr = sf.read(io.BytesIO(raw_bytes))
                    processed_audio = standardizer.process(audio_data, orig_sr=sr)
                    sf.write(out_wav, processed_audio, 16000, subtype="PCM_16")

                records.append({
                    "id": row_id,
                    "audio_path": out_wav,
                    "language": lang,
                    "is_tts": is_tts,
                    "text": text,
                })
                saved_count += 1

            print(f"[{primary_lang}] Row group {rg_idx} ({saved_count} samples) processed in {time.time()-t0:.2f}s", flush=True)

    return records


def main():
    processed_dir = "d:/MAJOR PROJECT IV YEAR/audio-deepfake-research/data/processed"
    manifest_dir = "d:/MAJOR PROJECT IV YEAR/audio-deepfake-research/data/manifests"
    splits_dir = "d:/MAJOR PROJECT IV YEAR/audio-deepfake-research/data/splits"

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    print("=" * 70, flush=True)
    print("DOWNLOADING & STANDARDIZING REAL INDICTTS CORPUS (16 LANGUAGES)", flush=True)
    print("=" * 70, flush=True)

    all_records = []
    # Use 4 parallel threads to stream shards concurrently without overwhelming network
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_shard, s, processed_dir, 2): s["primary_lang"] for s in TARGET_SHARDS}
        for future in concurrent.futures.as_completed(futures):
            lang_name = futures[future]
            try:
                records = future.result()
                all_records.extend(records)
                print(f"--> Finished {lang_name}: {len(records)} samples.", flush=True)
            except Exception as e:
                print(f"--> Error on {lang_name}: {e}", flush=True)

    df_all = pd.DataFrame(all_records)
    print("\n" + "=" * 70, flush=True)
    print("DATASET ACQUISITION COMPLETE", flush=True)
    print(f"Total downloaded and processed samples: {len(df_all)}", flush=True)
    print("Class Balance:\n", df_all["is_tts"].value_counts().to_dict(), flush=True)
    print("Languages:\n", df_all["language"].value_counts().to_dict(), flush=True)

    # Master manifest
    master_manifest_path = os.path.join(manifest_dir, "master_manifest.jsonl")
    with open(master_manifest_path, "w", encoding="utf-8") as f:
        for r in all_records:
            lang_lower = str(r["language"]).lower()
            manifest_row = {
                "sample_id": r["id"],
                "audio_path": r["audio_path"],
                "language": r["language"],
                "is_dravidian": lang_lower in DRAVIDIAN_LANGUAGES,
                "is_tibeto_burman": lang_lower in TIBETO_BURMAN_LANGUAGES,
                "label_is_tts": int(r["is_tts"]),
                "text": r["text"],
            }
            f.write(json.dumps(manifest_row) + "\n")
    print(f"Saved master manifest to: {master_manifest_path}", flush=True)

    # Create linguistic splits
    print("\nGenerating text-disjoint linguistic splits...", flush=True)
    splits = create_dual_linguistic_splits(df_all, output_dir=splits_dir)

    # Audit summary
    audit_summary = {
        "total_samples": len(df_all),
        "languages": df_all["language"].value_counts().to_dict(),
        "class_balance": df_all["is_tts"].value_counts().to_dict(),
        "split_counts": {k: len(v) for k, v in splits.items()},
    }
    summary_path = os.path.join(manifest_dir, "audit_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    # Also sync splits to root data directory
    root_splits_dir = "d:/MAJOR PROJECT IV YEAR/data/splits"
    os.makedirs(root_splits_dir, exist_ok=True)
    for k, v in splits.items():
        v.to_csv(os.path.join(root_splits_dir, f"{k}.csv"), index=False)
        print(f"Split {k}: {len(v)} samples", flush=True)

    print(f"\nAudit summary successfully saved to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
