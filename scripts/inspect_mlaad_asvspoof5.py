from huggingface_hub import HfApi, hf_hub_url
import fsspec
import pyarrow.parquet as pq
import json

def inspect_mlaad():
    api = HfApi()
    print("Listing files in mueller91/MLAAD-tiny...")
    files = api.list_repo_files("mueller91/MLAAD-tiny", repo_type="dataset")
    wav_files = [f for f in files if f.endswith(".wav")]
    print(f"Total WAV files in MLAAD-tiny: {len(wav_files)}")
    real_wavs = [f for f in wav_files if f.startswith("real/")]
    fake_wavs = [f for f in wav_files if f.startswith("fake/")]
    print(f"Real WAVs: {len(real_wavs)}, Fake WAVs: {len(fake_wavs)}")
    print("Sample Real WAVs:", real_wavs[:5])
    print("Sample Fake WAVs:", fake_wavs[:5])

def inspect_asvspoof5():
    api = HfApi()
    print("\nInspecting SpeechAntiSpoofingBenchmarks/ASVspoof5...")
    url = hf_hub_url("SpeechAntiSpoofingBenchmarks/ASVspoof5", "data/labels.parquet", repo_type="dataset")
    fs, path = fsspec.core.url_to_fs(url)
    with fs.open(path, "rb") as f:
        pf = pq.ParquetFile(f)
        table = pf.read()
        df = table.to_pandas()
        print("ASVspoof5 Labels schema / head:")
        print(df.head(5))
        print("Columns:", df.columns.tolist())
        if "label" in df.columns:
            print("Label distribution:", df["label"].value_counts().to_dict())

def main():
    inspect_mlaad()
    inspect_asvspoof5()

if __name__ == "__main__":
    main()
