"""
test_data.py — pytest tests for model_training/src/data.py
"""

from __future__ import annotations

import sys
import pathlib

import pandas as pd
import pytest

# Make src importable when running pytest from model_training/
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from data import preprocess_url, balance_dataset, load_raw_data


# ---------------------------------------------------------------------------
# preprocess_url
# ---------------------------------------------------------------------------

class TestPreprocessUrl:
    def test_lowercase(self):
        assert preprocess_url("HTTP://EXAMPLE.COM/PATH") == "http://example.com/path"

    def test_strip_whitespace(self):
        assert preprocess_url("  https://example.com  ") == "https://example.com"

    def test_valid_https(self):
        assert preprocess_url("https://example.com") == "https://example.com"

    def test_valid_http(self):
        assert preprocess_url("http://example.com") == "http://example.com"

    def test_none_input(self):
        assert preprocess_url(None) is None  # type: ignore[arg-type]

    def test_empty_string(self):
        assert preprocess_url("") is None

    def test_no_scheme_rejected(self):
        assert preprocess_url("example.com/path") is None

    def test_ftp_scheme_rejected(self):
        assert preprocess_url("ftp://example.com") is None

    def test_too_long_rejected(self):
        long_url = "https://example.com/" + "a" * 2048
        assert preprocess_url(long_url) is None

    def test_non_string_rejected(self):
        assert preprocess_url(12345) is None  # type: ignore[arg-type]

    def test_url_with_query_string(self):
        url = "https://example.com/page?id=1&ref=2"
        result = preprocess_url(url)
        assert result == url

    def test_url_with_fragment(self):
        url = "https://example.com/#section"
        assert preprocess_url(url) == url


# ---------------------------------------------------------------------------
# balance_dataset
# ---------------------------------------------------------------------------

class TestBalanceDataset:
    def _make_df(self, n_benign: int, n_malicious: int) -> pd.DataFrame:
        rows = [{"url": f"https://b{i}.com", "label": 0} for i in range(n_benign)]
        rows += [{"url": f"https://m{i}.com", "label": 1} for i in range(n_malicious)]
        return pd.DataFrame(rows)

    def test_equal_classes_unchanged(self):
        df = self._make_df(100, 100)
        balanced = balance_dataset(df, seed=0)
        assert (balanced["label"] == 0).sum() == 100
        assert (balanced["label"] == 1).sum() == 100

    def test_majority_undersampled(self):
        df = self._make_df(300, 100)
        balanced = balance_dataset(df, seed=0)
        assert (balanced["label"] == 0).sum() == 100
        assert (balanced["label"] == 1).sum() == 100

    def test_minority_not_grown(self):
        df = self._make_df(50, 200)
        balanced = balance_dataset(df, seed=0)
        assert (balanced["label"] == 0).sum() == 50
        assert (balanced["label"] == 1).sum() == 50

    def test_total_length(self):
        df = self._make_df(1000, 400)
        balanced = balance_dataset(df)
        assert len(balanced) == 800  # 400 * 2

    def test_reproducible_with_seed(self):
        df = self._make_df(500, 200)
        b1 = balance_dataset(df, seed=7)
        b2 = balance_dataset(df, seed=7)
        pd.testing.assert_frame_equal(b1.reset_index(drop=True), b2.reset_index(drop=True))

    def test_different_seeds_different_rows(self):
        df = self._make_df(500, 200)
        b1 = balance_dataset(df, seed=1)
        b2 = balance_dataset(df, seed=2)
        # Same shape but different row order / selection
        assert len(b1) == len(b2)
        # Very unlikely to be identical
        assert not b1["url"].tolist() == b2["url"].tolist()


# ---------------------------------------------------------------------------
# load_raw_data
# ---------------------------------------------------------------------------

class TestLoadRawData:
    def _write_csv(self, tmp_path: pathlib.Path, data: list[dict], filename="data.csv") -> pathlib.Path:
        df = pd.DataFrame(data)
        p = tmp_path / filename
        df.to_csv(p, index=False)
        return p

    def test_basic_load(self, tmp_path):
        rows = [
            {"url": "https://good.com", "result": 0},
            {"url": "https://bad.com", "result": 1},
        ]
        p = self._write_csv(tmp_path, rows)
        df = load_raw_data(str(p), "url", "result")
        assert len(df) == 2
        assert set(df.columns) == {"url", "label"}
        assert list(df["label"]) == [0, 1]

    def test_invalid_labels_dropped(self, tmp_path):
        rows = [
            {"url": "https://good.com", "result": 0},
            {"url": "https://bad.com", "result": "not_a_number"},
            {"url": "https://other.com", "result": 1},
        ]
        p = self._write_csv(tmp_path, rows)
        df = load_raw_data(str(p), "url", "result")
        assert len(df) == 2

    def test_non_binary_labels_dropped(self, tmp_path):
        rows = [
            {"url": "https://good.com", "result": 0},
            {"url": "https://bad.com", "result": 2},  # not 0 or 1
        ]
        p = self._write_csv(tmp_path, rows)
        df = load_raw_data(str(p), "url", "result")
        assert len(df) == 1

    def test_invalid_urls_dropped(self, tmp_path):
        rows = [
            {"url": "not-a-url", "result": 0},
            {"url": "https://valid.com", "result": 1},
        ]
        p = self._write_csv(tmp_path, rows)
        df = load_raw_data(str(p), "url", "result")
        assert len(df) == 1
        assert df.iloc[0]["url"] == "https://valid.com"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_raw_data(str(tmp_path / "nonexistent.csv"), "url", "result")

    def test_missing_column_raises(self, tmp_path):
        rows = [{"link": "https://example.com", "result": 0}]
        p = self._write_csv(tmp_path, rows)
        with pytest.raises((ValueError, KeyError)):
            load_raw_data(str(p), "url", "result")

    def test_empty_after_cleaning_raises(self, tmp_path):
        rows = [
            {"url": "not-a-url", "result": 5},
        ]
        p = self._write_csv(tmp_path, rows)
        with pytest.raises(ValueError):
            load_raw_data(str(p), "url", "result")
