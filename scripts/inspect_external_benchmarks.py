from huggingface_hub import HfApi

def inspect_repo(repo_id: str, repo_type: str = "dataset"):
    api = HfApi()
    print("=" * 60)
    print(f"INSPECTING: {repo_id}")
    print("=" * 60)
    try:
        info = api.repo_info(repo_id, repo_type=repo_type, files_metadata=True)
        files = [(s.rfilename, (s.size or 0) / (1024 * 1024)) for s in info.siblings]
        print(f"Total files: {len(files)}")
        for fn, size_mb in files[:15]:
            print(f"  {fn} ({size_mb:.2f} MB)")
        if len(files) > 15:
            print(f"  ... and {len(files) - 15} more files")
    except Exception as e:
        print(f"Error inspecting {repo_id}: {e}")

def main():
    inspect_repo("mueller91/MLAAD-tiny")
    inspect_repo("mueller91/MLAAD")
    inspect_repo("jungjee/asvspoof5")
    inspect_repo("SpeechAntiSpoofingBenchmarks/ASVspoof5")

if __name__ == "__main__":
    main()
