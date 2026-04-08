import csv
from urllib.parse import urlparse

INPUT_FILE = "../../datasets/combined_urls.csv"  # change this
OUTPUT_FILE = "../feature_extraction/dataset_benign_with_path.csv"


def has_valid_path(url):
    try:
        parsed = urlparse(url)
        path = parsed.path.strip()

        # Exclude empty or root-only paths
        return path not in ("", "/")
    except:
        return False


def main():
    count = 0

    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as infile, \
         open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            try:
                if len(row) < 2:
                    continue

                url = row[0].strip()
                result = row[1].strip()

                # Keep only benign
                if result != "0":
                    continue

                # Keep only URLs with path
                if not has_valid_path(url):
                    continue

                count += 1
                writer.writerow([count, url])

            except Exception:
                continue  # skip bad rows

    print(f"Saved {count} benign URLs with path to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()