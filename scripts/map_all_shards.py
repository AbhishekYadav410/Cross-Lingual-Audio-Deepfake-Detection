import concurrent.futures
import fsspec
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_url
import json
import os

def inspect_single_shard(shard_idx: int):
    repo_id = "SherryT997/IndicTTS-Deepfake-Challenge-Data"
    filename = f"data/train-{shard_idx:05d}-of-00035.parquet"
    url = hf_hub_url(repo_id, filename, repo_type="dataset")
    fs, path = fsspec.core.url_to_fs(url)
    try:
        with fs.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            table = pf.read(columns=["language", "is_tts"])
            df = table.to_pandas()
            lang_counts = df["language"].value_counts().to_dict()
            tts_counts = df["is_tts"].value_counts().to_dict()
            res = {
                "shard_idx": shard_idx,
                "filename": filename,
                "total_rows": len(df),
                "languages": lang_counts,
                "class_balance": tts_counts,
            }
            print(f"Shard {shard_idx:02d}: {len(df)} rows | {lang_counts} | TTS: {tts_counts}", flush=True)
            return res
    except Exception as e:
        print(f"Error reading shard {shard_idx}: {e}", flush=True)
        return {"shard_idx": shard_idx, "error": str(e)}

def main():
    total_shards = 35
    print(f"Scanning all {total_shards} training shards in parallel (8 threads)...", flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(inspect_single_shard, i): i for i in range(total_shards)}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)

    results.sort(key=lambda x: x.get("shard_idx", 0))
    out_path = "d:/MAJOR PROJECT IV YEAR/audio-deepfake-research/data/manifests/hf_shard_map.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved complete shard map to {out_path}", flush=True)

if __name__ == "__main__":
    main()
