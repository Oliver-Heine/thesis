import re
import csv
import os

CORRUPTED_FILE = "seen_urls.pkl"
RESCUED_CSV = "seen_urls_rescued.csv"

def rescue_corrupted_pickle():
    if not os.path.exists(CORRUPTED_FILE):
        print(f"Could not find {CORRUPTED_FILE}")
        return

    print(f"Reading raw bytes from {CORRUPTED_FILE}...")
    
    with open(CORRUPTED_FILE, "rb") as f:
        raw_data = f.read()

    # Convert the raw binary to text, ignoring the binary gibberish/pickle control characters
    text_data = raw_data.decode("utf-8", errors="ignore")

    # Regex to find anything that looks like a URL (http or https)
    # This will pull the URLs right out of the surrounding binary junk
    print("Extracting URLs...")
    recovered_urls = set(re.findall(r'https?://[a-zA-Z0-9.\-_~:/?#[\]@!$&\'()*+,;=%]+', text_data))

    # NOTE: If your crawler only saves raw domains (like 'example.com') instead of full URLs, 
    # comment out the line above and uncomment the line below:
    # recovered_urls = set(re.findall(r'[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', text_data))

    if not recovered_urls:
        print("No URLs could be extracted. The file might be corrupted beyond text recovery.")
        return

    print(f"Success! Rescued {len(recovered_urls)} unique URLs.")
    
    # Save the rescued data to a safe CSV
    print(f"Saving to {RESCUED_CSV}...")
    with open(RESCUED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for url in recovered_urls:
            writer.writerow([url])
            
    print("Rescue complete! You can now use the rescued CSV file.")

if __name__ == "__main__":
    rescue_corrupted_pickle()
