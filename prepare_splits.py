import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# =====================================================================
# CONFIGURATION VARIABLES
# =====================================================================
SEED = 77
ROOT_DIR = "./MMRDR/MMRDR-CFP"   # Directory containing img/ and FP.csv
SPLITS_DIR = "./data_splits"          # Output directory for train/val/test CSVs
VAL_RATIO = 0.15                      # 85% train / 15% validation split from 'tr*' images
# =====================================================================


def prepare_splits() -> None:
    csv_path = os.path.join(ROOT_DIR, "FP.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}. Please check ROOT_DIR at the top of prepare_splits.py."
        )

    df = pd.read_csv(csv_path)
    df["filename"] = df["image"].apply(lambda x: os.path.basename(str(x)))

    # Isolate predefined training ('tr*') and testing ('ts*') partitions
    train_val_df = df[df["filename"].str.startswith("tr")].copy().reset_index(drop=True)
    test_df = df[df["filename"].str.startswith("ts")].copy().reset_index(drop=True)

    if len(train_val_df) == 0 or len(test_df) == 0:
        raise ValueError("Failed to detect 'tr' and 'ts' image prefixes in FP.csv.")

    # Stratified split across the 5 DR severity grades
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
    train_idx, val_idx = next(splitter.split(train_val_df, train_val_df["grade"]))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    os.makedirs(SPLITS_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(SPLITS_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(SPLITS_DIR, "test.csv"), index=False)

    print(f"Deterministic splits saved to: {SPLITS_DIR}/")
    print("-" * 65)
    print(f"{'Grade':<10} | {'Train':<12} | {'Val':<12} | {'Test (Locked)':<15} | {'Total':<10}")
    print("-" * 65)
    for grade in sorted(df["grade"].unique()):
        n_tr = int((train_df["grade"] == grade).sum())
        n_val = int((val_df["grade"] == grade).sum())
        n_ts = int((test_df["grade"] == grade).sum())
        total = n_tr + n_val + n_ts
        print(f"Grade {grade:<4} | {n_tr:<12} | {n_val:<12} | {n_ts:<15} | {total:<10}")
    print("-" * 65)
    print(f"{'Total':<10} | {len(train_df):<12} | {len(val_df):<12} | {len(test_df):<15} | {len(df):<10}")


if __name__ == "__main__":
    prepare_splits()