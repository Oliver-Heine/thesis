import csv
from collections import defaultdict, Counter
import tldextract

INPUT_FILE = "../../datasets/combined_urls.csv"  # Change to your dataset path
TOP_N = 10  # Number of TLDs to keep explicitly


def extract_suffix(url: str) -> str:
    """Extract the full suffix (multi-level) from a URL."""
    url = url.lower().strip()
    ext = tldextract.extract(url)
    if ext.suffix:
        return "." + ext.suffix  # keep multi-level like co.uk
    else:
        return ".unknown"


def main():
    seen_urls = set()

    # Stats dictionaries
    tld_counts = defaultdict(int)
    tld_benign = defaultdict(int)
    tld_malicious = defaultdict(int)

    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            url = row[0].strip()
            result = row[1].strip()

            if url in seen_urls:
                continue
            seen_urls.add(url)

            tld = extract_suffix(url)
            tld_counts[tld] += 1
            if result == "0":
                tld_benign[tld] += 1
            elif result == "1":
                tld_malicious[tld] += 1

    # Identify top N TLDs
    top_tlds = [tld for tld, _ in Counter(tld_counts).most_common(TOP_N)]

    # Prepare summary with 'Other' aggregated
    other_total = other_benign = other_malicious = 0

    print("===== TLD DISTRIBUTION =====")
    for tld in top_tlds:
        total = tld_counts[tld]
        benign = tld_benign[tld]
        malicious = tld_malicious[tld]
        print(f"{tld:10} | Total: {total:6} | Benign: {benign:6} | Malicious: {malicious:6}")

    # Aggregate the rest into 'Other'
    for tld, count in tld_counts.items():
        if tld not in top_tlds:
            other_total += count
            other_benign += tld_benign[tld]
            other_malicious += tld_malicious[tld]

    print(f"{'Other':10} | Total: {other_total:6} | Benign: {other_benign:6} | Malicious: {other_malicious:6}")


if __name__ == "__main__":
    main()