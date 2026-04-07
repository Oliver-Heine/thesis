import pandas as pd
import json
import os
from pathlib import Path

# ---------------------------
# Config
# ---------------------------
DATA_PATH = "../../datasets/backend_training.csv"  # your dataset
OUTPUT_BUCKET_FILE = "../../datasets/feature_buckets.json"
NUM_QUANTILES = 10  # can adjust: e.g., quartiles (4), deciles (10)

# Features to bucket
COUNT_FEATURES = [
    "redirect_count",
    "third_party_domains",
    "num_inputs",
    "num_iframes",
    "external_scripts",
    "domain_age"
]

CONTINUOUS_FEATURES = [
    "page_size",
    "tld_entropy"
]

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv(DATA_PATH)

# ---------------------------
# Function to generate buckets
# ---------------------------
def generate_buckets(series, num_quantiles):
    """
    Returns list of bucket boundaries from quantiles.
    """
    # Remove NaN first
    series_clean = series.dropna()
    # Quantile boundaries
    quantiles = series_clean.quantile([i / num_quantiles for i in range(1, num_quantiles)]).tolist()
    # Round small floats for readability
    quantiles = [round(q, 6) for q in quantiles]
    return quantiles

# ---------------------------
# Generate all buckets
# ---------------------------
bucket_dict = {}

# Count-based features
for feature in COUNT_FEATURES:
    # For counts, we can round to integers and remove duplicates
    boundaries = generate_buckets(df[feature].astype(int), NUM_QUANTILES)
    bucket_dict[feature] = boundaries

# Continuous features
for feature in CONTINUOUS_FEATURES:
    boundaries = generate_buckets(df[feature], NUM_QUANTILES)
    bucket_dict[feature] = boundaries

# ---------------------------
# Save buckets
# ---------------------------
with open(OUTPUT_BUCKET_FILE, "w") as f:
    json.dump(bucket_dict, f, indent=4)

print(f"Buckets saved to {OUTPUT_BUCKET_FILE}")
print(json.dumps(bucket_dict, indent=4))