import json
from collections import defaultdict

DRAVIDIAN_LANGUAGES = {"tamil", "telugu", "kannada", "malayalam"}
TIBETO_BURMAN_LANGUAGES = {"bodo", "manipuri"}

def main():
    with open("data/manifests/hf_shard_map.json") as f:
        data = json.load(f)

    lang_totals = defaultdict(int)
    lang_tts = defaultdict(lambda: defaultdict(int))
    family_totals = defaultdict(int)

    for item in data:
        for lang, count in item["languages"].items():
            lang_totals[lang] += count
            l_lower = lang.lower()
            if l_lower in DRAVIDIAN_LANGUAGES:
                family_totals["Dravidian (Held-Out 1)"] += count
            elif l_lower in TIBETO_BURMAN_LANGUAGES:
                family_totals["Tibeto-Burman (Held-Out 2)"] += count
            else:
                family_totals["Indo-Aryan & Contact (Seen)"] += count

    print("=" * 60)
    print("TOTAL DATASET DISTRIBUTION ACROSS ALL 35 SHARDS")
    print("=" * 60)
    for fam, cnt in family_totals.items():
        print(f"  {fam:<35}: {cnt:>6} samples ({cnt/31102*100:.1f}%)")
    print("-" * 60)
    print("LANGUAGE BREAKDOWN:")
    for lang, cnt in sorted(lang_totals.items()):
        l_lower = lang.lower()
        fam = "Dravidian" if l_lower in DRAVIDIAN_LANGUAGES else ("Tibeto-Burman" if l_lower in TIBETO_BURMAN_LANGUAGES else "Indo-Aryan")
        print(f"  {lang:<15} [{fam:<14}]: {cnt:>5} samples")
    print("=" * 60)
    print(f"Total samples: {sum(lang_totals.values())}")

if __name__ == "__main__":
    main()
