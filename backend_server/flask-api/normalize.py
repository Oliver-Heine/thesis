import pandas as pd
import re
from urllib.parse import urlparse
import tldextract

NUMERICAL_BINS = {
    "redirect_count": [0, 1, 2, 3, 4, 5, 6, 7, 8, 231],
    "server_redirect_count": [0, 1, 2, 3, 4, 5, 6, 7, 8, 488],
    "third_party_domains": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 193],
    "num_inputs": [0, 1, 2, 3, 4, 5, 10, 15, 20, 3775],
    "num_iframes": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 102.0],
    "external_scripts": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 407],
    "page_size": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 24635.0810546875]
}

BOOLEAN_FEATURES = [
    "login_form",
    "password_input",
    "uses_eval",
    "canvas_fingerprint",
    "cert_valid"
]

def bucketize(features: dict):
    tokens = []

    for feature, bins in NUMERICAL_BINS.items():
        value = float(features.get(feature, 0))

        bucket = pd.cut(
            [value],
            bins=bins,
            labels=False,
            include_lowest=True
        )[0]

        tokens.append(f"<{feature.upper()}_{int(bucket)}>")

    for feature in BOOLEAN_FEATURES:
        val = features.get(feature, 0)
        tokens.append(
            f"<{feature.upper()}_{'YES' if val == 1 else 'NO'}>"
        )

    return tokens


def normalize_url(url: str):
    url = url.lower().strip()

    # remove protocol and www for normalization
    url_clean = re.sub(r"^https?://", "", url)
    url_clean = re.sub(r"^www\.", "", url_clean)

    # Ensure urlparse sees a scheme, otherwise domain may be misparsed
    parsed = urlparse("http://" + url_clean)
    ext = tldextract.extract(url_clean)

    tokens = []

    # subdomain
    if ext.subdomain:
        tokens.append("<subdomain>")
        tokens.extend(ext.subdomain.split("."))

    # domain
    tokens.append("<domain>")
    tokens.append(ext.domain)

    # domain extension
    tokens.append("<suffix>")
    tokens.extend(ext.suffix.split("."))

    # path
    if parsed.path and parsed.path != "/":
        tokens.append("<path>")
        tokens.extend(re.split(r"[/\-_.?=&]", parsed.path))

    # query
    if parsed.query:
        tokens.append("<query>")
        tokens.extend(re.split(r"[=&]", parsed.query))

    return " ".join([t for t in tokens if t])

def build_output(url: str, features: dict):
    url_tokens = normalize_url(url)
    feature_tokens = bucketize(features)

    return f"{url_tokens} {' '.join(feature_tokens)}"