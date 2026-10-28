import fsspec
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_url
import soundfile as sf
import io
import time

def main():
    repo_id = "SherryT997/IndicTTS-Deepfake-Challenge-Data"
    filename = "data/train-00005-of-00035.parquet"  # Bodo
    url = hf_hub_url(repo_id, filename, repo_type="dataset")
    print("Opening remote parquet file for Bodo...")
    t0 = time.time()
    fs, path = fsspec.core.url_to_fs(url)
    with fs.open(path, "rb") as f:
        pf = pq.ParquetFile(f)
        print(f"Header read in {time.time() - t0:.2f}s. Num row groups: {pf.num_row_groups}")
        t1 = time.time()
        print("Reading row group 0 (100 audio samples)...")
        table = pf.read_row_group(0)
        print(f"Row group 0 read in {time.time() - t1:.2f}s! Total rows: {len(table)}")
        df = table.select(["id", "language", "is_tts"]).to_pandas()
        print("Language & class counts:")
        print(df["language"].value_counts().to_dict())
        print(df["is_tts"].value_counts().to_dict())
        
        # Test decoding first audio sample
        first_audio = table.column("audio")[0].as_py()
        audio_data, sr = sf.read(io.BytesIO(first_audio["bytes"]))
        print(f"Audio sample 0 decoded: shape={audio_data.shape}, sr={sr}")

if __name__ == "__main__":
    main()
