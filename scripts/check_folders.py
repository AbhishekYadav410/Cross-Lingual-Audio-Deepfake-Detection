from huggingface_hub import HfApi, hf_hub_url
import fsspec
import pyarrow.parquet as pq

def check_mlaad_folders():
    api = HfApi()
    files = api.list_repo_files("mueller91/MLAAD-tiny", repo_type="dataset")
    prefixes = set()
    for f in files:
        parts = f.split("/")
        if len(parts) > 1:
            prefixes.add(parts[0])
    print("MLAAD-tiny top-level folders:", prefixes)
    non_fake = [f for f in files if not f.startswith("fake/")]
    print("Non-fake files in MLAAD-tiny (first 10):", non_fake[:10])

def check_asvspoof5_shard():
    url = hf_hub_url("SpeechAntiSpoofingBenchmarks/ASVspoof5", "data/test-00000-of-00200.parquet", repo_type="dataset")
    fs, path = fsspec.core.url_to_fs(url)
    with fs.open(path, "rb") as f:
        pf = pq.ParquetFile(f)
        print("\nASVspoof5 Shard 0 num row groups:", pf.num_row_groups)
        print("Schema columns:", pf.schema.names)
        table = pf.read_row_group(0)
        df = table.to_pandas()
        print("Row group 0 rows:", len(df))
        print("Columns in df:", df.columns.tolist())
        print(df.head(2))

if __name__ == "__main__":
    check_mlaad_folders()
    check_asvspoof5_shard()
