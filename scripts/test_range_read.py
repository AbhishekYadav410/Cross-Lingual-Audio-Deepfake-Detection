import fsspec
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_url

def main():
    url = hf_hub_url("SherryT997/IndicTTS-Deepfake-Challenge-Data", "data/train-00001-of-00035.parquet", repo_type="dataset")
    print(f"URL: {url}")
    fs, path = fsspec.core.url_to_fs(url)
    with fs.open(path, "rb") as f:
        parquet_file = pq.ParquetFile(f)
        print("Schema columns:", parquet_file.schema.names)
        # Read ONLY metadata/text columns, NOT audio!
        table = parquet_file.read(columns=["id", "language", "is_tts"])
        df = table.to_pandas()
        print("Shard 1 language counts:", df["language"].value_counts().to_dict())

if __name__ == "__main__":
    main()
