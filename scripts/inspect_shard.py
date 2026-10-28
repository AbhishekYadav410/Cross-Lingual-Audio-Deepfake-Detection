from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import io
import soundfile as sf

def main():
    print("Downloading shard 0...")
    local_file = hf_hub_download(
        repo_id="SherryT997/IndicTTS-Deepfake-Challenge-Data",
        repo_type="dataset",
        filename="data/train-00000-of-00035.parquet"
    )
    print(f"Downloaded to {local_file}")
    table = pq.read_table(local_file)
    print("Schema:")
    print(table.schema)
    print(f"Num rows: {len(table)}")
    df = table.select(["id", "language", "is_tts", "text"]).to_pandas()
    # Avoid printing Indic text to Windows cp1252 console
    print("Columns:", df.columns.tolist())
    print("Language counts in shard 0:")
    print(df["language"].value_counts().to_dict())
    print("Class balance in shard 0:")
    print(df["is_tts"].value_counts().to_dict())

    # Check audio column format
    audio_col = table.column("audio")[0].as_py()
    print("Audio col type:", type(audio_col))
    print("Audio keys/bytes:", audio_col.keys() if isinstance(audio_col, dict) else type(audio_col))
    if isinstance(audio_col, dict) and "bytes" in audio_col:
        raw_bytes = audio_col["bytes"]
        print(f"Raw bytes length: {len(raw_bytes)}")
        data, sr = sf.read(io.BytesIO(raw_bytes))
        print(f"Audio read successfully! Shape: {data.shape}, Sample rate: {sr}")

if __name__ == "__main__":
    main()
