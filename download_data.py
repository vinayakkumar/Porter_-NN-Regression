"""Download the Porter delivery dataset to data/porter_delivery.csv.

    python download_data.py                     # skips the download if the file is already there
    python download_data.py --force             # download again and replace the file
    python download_data.py --output some/path.csv

The file is the Google Drive file shared in the problem brief (175,777 rows, 14 columns).
Needs the `gdown` package (it is in requirements.txt).
"""
import argparse
import os
import sys

FILE_ID = "1kkgBGldeswHBgVKEJiqqG7VZvqZNMc-5"      # file id from the Google Drive link in the brief
EXPECTED_ROWS = 175_777
HERE = os.path.dirname(os.path.abspath(__file__))


def count_rows(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1                # minus the header line


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=os.path.join(HERE, "data", "porter_delivery.csv"))
    ap.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = ap.parse_args()

    if os.path.exists(args.output) and not args.force:
        print(f"Already there: {args.output} ({count_rows(args.output):,} rows). Use --force to download again.")
        return 0

    try:
        import gdown
    except ImportError:
        print("gdown is not installed. Run: pip install gdown", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result = gdown.download(id=FILE_ID, output=args.output, quiet=False)
    if result is None or not os.path.exists(args.output):
        print("Download failed. Open the Google Drive link from the brief in a browser and save the file as "
              f"{args.output}.", file=sys.stderr)
        return 1

    rows = count_rows(args.output)
    print(f"Saved {args.output} with {rows:,} rows.")
    if rows != EXPECTED_ROWS:
        print(f"Warning: expected {EXPECTED_ROWS:,} rows.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
