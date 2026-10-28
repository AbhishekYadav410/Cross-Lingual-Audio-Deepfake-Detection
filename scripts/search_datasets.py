from huggingface_hub import HfApi

def main():
    api = HfApi()
    print("Searching for MLAAD datasets...")
    mlaad_datasets = list(api.list_datasets(search="MLAAD", limit=20))
    for d in mlaad_datasets:
        print(f"MLAAD: {d.id} | likes: {d.likes} | downloads: {d.downloads}")

    print("\nSearching for ASVspoof5 datasets...")
    asv5_datasets = list(api.list_datasets(search="asvspoof5", limit=20))
    for d in asv5_datasets:
        print(f"ASVspoof5: {d.id} | likes: {d.likes} | downloads: {d.downloads}")

    if not asv5_datasets:
        print("\nSearching for ASVspoof (general)...")
        asv_datasets = list(api.list_datasets(search="asvspoof", limit=15))
        for d in asv_datasets:
            print(f"ASVspoof: {d.id} | likes: {d.likes} | downloads: {d.downloads}")

if __name__ == "__main__":
    main()
