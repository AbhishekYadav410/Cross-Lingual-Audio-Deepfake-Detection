from huggingface_hub import HfApi

def main():
    api = HfApi()
    repo_info = api.repo_info("SherryT997/IndicTTS-Deepfake-Challenge-Data", repo_type="dataset", files_metadata=True)
    for sibling in repo_info.siblings:
        if sibling.rfilename.startswith("data/train"):
            size_mb = (sibling.size or 0) / (1024 * 1024)
            print(f"{sibling.rfilename}: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
