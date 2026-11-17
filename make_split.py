"""Create the untouched reporting set (20%) and the training set (80%).

Run once from the repository root:
    python make_split.py

Source data: A. Hannousse and S. Yahiouche, "Web page phishing detection,"
Mendeley Data, v3, 2021, doi: 10.17632/c2gw7fy2j4.3 (file dataset_B_05_2020.csv).

The split is stratified on the outcome ("status") and uses a fixed random seed, so
re-running this script reproduces exactly the same two files.
"""

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
REPORTING_FRACTION = 0.20
OUTCOME = "status"

DATA_DIR = Path(__file__).resolve().parent / "Data"
RAW_FILE = DATA_DIR / "raw" / "dataset_B_05_2020.csv"
RAW_URL = (
    "https://data.mendeley.com/public-files/datasets/c2gw7fy2j4/files/"
    "575316f4-ee1d-453e-a04f-7b950915b61b/file_downloaded"
)
# SHA-256 published by Mendeley Data for dataset_B_05_2020.csv
RAW_SHA256 = "21093e2902e5441c86a6daf95e86e7c332046e477fdf109a579d7bd81e586d6c"

TRAIN_FILE = DATA_DIR / "phishing_train.csv"
REPORTING_FILE = DATA_DIR / "phishing_reporting.csv"


def get_raw_data() -> pd.DataFrame:
    """Download the original file if needed and check it is the published version."""
    if not RAW_FILE.exists():
        RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading original data to {RAW_FILE} ...")
        urllib.request.urlretrieve(RAW_URL, RAW_FILE)

    digest = hashlib.sha256(RAW_FILE.read_bytes()).hexdigest()
    if digest != RAW_SHA256:
        raise ValueError("The raw file does not match the published version (SHA-256 mismatch).")
    return pd.read_csv(RAW_FILE)


def main() -> None:
    data = get_raw_data()
    print(f"Original data: {data.shape[0]:,} rows, {data.shape[1] - 2} features "
          f"(plus the 'url' index and the '{OUTCOME}' outcome)")

    train, reporting = train_test_split(
        data,
        test_size=REPORTING_FRACTION,
        stratify=data[OUTCOME],
        random_state=RANDOM_STATE,
    )

    # Sanity checks: no row is in both sets, and nothing was lost
    assert set(train.index).isdisjoint(reporting.index)
    assert len(train) + len(reporting) == len(data)

    train.to_csv(TRAIN_FILE, index=False)
    reporting.to_csv(REPORTING_FILE, index=False)

    for name, part in [("Training", train), ("Reporting", reporting)]:
        counts = part[OUTCOME].value_counts().to_dict()
        print(f"{name:>9} set: {len(part):,} rows  {counts}")
    print(f"\nSaved {TRAIN_FILE.name} and {REPORTING_FILE.name} in {DATA_DIR}")


if __name__ == "__main__":
    main()
