import pyarrow.parquet as pq

def main():
    local_file = "C:/Users/Abhishek_Yadav/.cache/huggingface/hub/datasets--SherryT997--IndicTTS-Deepfake-Challenge-Data/snapshots/57347517658ae989597d8cef303cffb647ed2434/data/train-00000-of-00035.parquet"
    pf = pq.ParquetFile(local_file)
    print("Num row groups:", pf.num_row_groups)
    for i in range(pf.num_row_groups):
        rg = pf.metadata.row_group(i)
        print(f"Row group {i}: num_rows={rg.num_rows}, total_byte_size={rg.total_byte_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
