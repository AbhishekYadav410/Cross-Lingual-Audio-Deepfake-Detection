from huggingface_hub import HfApi

def main():
    api = HfApi()
    files = api.list_repo_files("SherryT997/IndicTTS-Deepfake-Challenge-Data", repo_type="dataset")
    parquet_files = [f for f in files if f.endswith(".parquet")]
    print(f"Total parquet files: {len(parquet_files)}")
    for f in parquet_files:
        print(f)

if __name__ == "__main__":
    main()
