import csv
import pickle

BENIGN_FILE = "dataset_benign.csv"
MALICIOUS_FILE = "dataset_malicious.csv"

BENIGN_UNIQUE_FILE = "dataset_benign.csv"
MALICIOUS_UNIQUE_FILE = "dataset_malicious.csv"


OUTPUT_FILE = "seen_urls.pkl"

def load_deduplicate_and_save(csv_file, output_file):
    seen = set()
    ordered_rows = []

    with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
        cleaned = (line.replace("\x00", "") for line in f)
        reader = csv.reader(cleaned)

        header = next(reader, None)
        if header:
            ordered_rows.append(header)

        for row in reader:
            try:
                if not row:
                    continue

                domain = row[0].strip()

                if domain and domain not in seen:
                    seen.add(domain)
                    ordered_rows.append(row)

            except Exception:
                continue  # skip bad rows

    # Write cleaned CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(ordered_rows)

    print(f"{csv_file}: {len(seen)} unique domains → saved to {output_file}")

    return seen

def load_unique_domains(csv_file):
    seen = set()
    ordered_unique = []

    with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
        # Remove NULL bytes on the fly
        cleaned = (line.replace("\x00", "") for line in f)

        reader = csv.reader(cleaned)
        header = next(reader, None)  # skip header

        for row in reader:
            if not row:
                continue

            domain = row[0].strip()

            if domain and domain not in seen:
                seen.add(domain)
                ordered_unique.append(domain)

    print(f"{csv_file}: {len(ordered_unique)} unique domains")
    return ordered_unique


def main():
    print("Loading and deduplicating CSV files...")

    benign_domains = load_deduplicate_and_save(BENIGN_FILE, BENIGN_UNIQUE_FILE)
    malicious_domains = load_deduplicate_and_save(MALICIOUS_FILE, MALICIOUS_UNIQUE_FILE)

    # # Combine both (still ensure uniqueness across both files)
    # combined_set = set(benign_domains)
    # combined_set.update(malicious_domains)
    #
    # print(f"Total unique domains combined: {len(combined_set)}")
    #
    # # Save to pickle
    # with open(OUTPUT_FILE, "wb") as f:
    #     pickle.dump(combined_set, f)
    #
    # print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()