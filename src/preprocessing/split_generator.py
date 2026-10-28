"""Linguistic-aware and prompt-disjoint split generator for dual held-out evaluations.

Partitions dataset into:
1. Seen / Training Languages: Indo-Aryan (Hindi, Marathi, Bengali, Gujarati, Odia,
   Assamese, Dogri, Nepali, Sanskrit) + Indian English.
   Partitioned into Train (70%), Val (15%), and In-Language Test (15%) with zero text overlap.
2. Held-Out Evaluation Target 1: Dravidian Languages (Tamil, Telugu, Kannada, Malayalam).
3. Held-Out Evaluation Target 2: Tibeto-Burman Languages (Bodo, Manipuri).
"""

from typing import Dict, List, Tuple
import hashlib
import os
import re
import pandas as pd
import numpy as np


DRAVIDIAN_LANGUAGES = {"tamil", "telugu", "kannada", "malayalam"}
TIBETO_BURMAN_LANGUAGES = {"bodo", "manipuri"}


def normalize_text(text: str) -> str:
    """Normalizes transcript text to detect semantic duplicates."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join(text.split())


def compute_text_hash(text: str) -> str:
    """Computes SHA-256 hash of normalized transcription."""
    norm_text = normalize_text(text)
    return hashlib.sha256(norm_text.encode("utf-8")).hexdigest()


def normalize_language_name(lang: str) -> str:
    """Standardizes language string formatting."""
    return str(lang).strip().lower()


def create_dual_linguistic_splits(
    df: pd.DataFrame,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Partitions the dataset into Seen Indo-Aryan, Held-Out Dravidian, and Held-Out Tibeto-Burman."""
    os.makedirs(output_dir, exist_ok=True)
    df = df.copy()

    df["lang_norm"] = df["language"].apply(normalize_language_name)
    df["text_hash"] = df["text"].apply(compute_text_hash)

    # 1. Isolate the two held-out target groups
    is_dravidian = df["lang_norm"].isin(DRAVIDIAN_LANGUAGES)
    is_tibeto_burman = df["lang_norm"].isin(TIBETO_BURMAN_LANGUAGES)

    df_held_out_dravidian = df[is_dravidian].copy()
    df_held_out_tibeto_burman = df[is_tibeto_burman].copy()

    # Seen Indo-Aryan & contact languages
    df_seen = df[~is_dravidian & ~is_tibeto_burman].copy()

    # 2. Grouped partitioning by text_hash at language level for Seen group
    np.random.seed(random_seed)

    train_indices = []
    val_indices = []
    test_in_lang_indices = []

    for lang, group in df_seen.groupby("lang_norm"):
        unique_hashes = list(group["text_hash"].unique())
        np.random.shuffle(unique_hashes)

        n_total = len(unique_hashes)
        n_train = max(1, int(round(n_total * train_ratio)))
        n_val = max(1, int(round(n_total * val_ratio)))
        if n_train + n_val >= n_total and n_total >= 3:
            n_train = n_total - 2
            n_val = 1

        train_hashes = set(unique_hashes[:n_train])
        val_hashes = set(unique_hashes[n_train : n_train + n_val])
        test_hashes = set(unique_hashes[n_train + n_val :])

        train_idx = group[group["text_hash"].isin(train_hashes)].index
        val_idx = group[group["text_hash"].isin(val_hashes)].index
        test_idx = group[group["text_hash"].isin(test_hashes)].index

        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_in_lang_indices.extend(test_idx)

    df_train = df_seen.loc[train_indices].sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    df_val = df_seen.loc[val_indices].sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    df_test_in_lang = df_seen.loc[test_in_lang_indices].sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    df_held_out_dravidian = df_held_out_dravidian.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    df_held_out_tibeto_burman = df_held_out_tibeto_burman.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Prompt leakage assertion
    train_texts = set(df_train["text_hash"].unique())
    val_texts = set(df_val["text_hash"].unique())
    test_texts = set(df_test_in_lang["text_hash"].unique())

    overlap_train_val = train_texts.intersection(val_texts)
    overlap_train_test = train_texts.intersection(test_texts)

    assert len(overlap_train_val) == 0, f"Prompt leakage: {len(overlap_train_val)} texts overlap between Train and Val!"
    assert len(overlap_train_test) == 0, f"Prompt leakage: {len(overlap_train_test)} texts overlap between Train and Test!"

    splits = {
        "train": df_train,
        "val": df_val,
        "test_in_lang": df_test_in_lang,
        "test_held_out_dravidian": df_held_out_dravidian,
        "test_held_out_tibeto_burman": df_held_out_tibeto_burman,
    }

    # Save to disk
    for name, split_df in splits.items():
        out_path = os.path.join(output_dir, f"{name}.csv")
        split_df.to_csv(out_path, index=False)
        print(f"Saved {name}: {len(split_df)} samples ({out_path})")

    return splits


# Alias for backward compatibility
create_linguistic_splits = create_dual_linguistic_splits
