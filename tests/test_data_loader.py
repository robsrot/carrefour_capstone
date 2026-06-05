"""Tests for src/data_loader.py — checksum utilities and loader helpers."""
from __future__ import annotations

import hashlib
import json

import polars as pl
import pytest

from src.data_loader import _sha256, verify_csv_checksums


# ── _sha256 ───────────────────────────────────────────────────────────────────

def test_sha256_consistent(tmp_path):
    f = tmp_path / "file.csv"
    f.write_bytes(b"idarticu;desc\n001;milk\n002;bread\n")
    assert _sha256(f) == _sha256(f)


def test_sha256_differs_for_different_content(tmp_path):
    f1 = tmp_path / "a.csv"
    f2 = tmp_path / "b.csv"
    f1.write_bytes(b"content_a")
    f2.write_bytes(b"content_b")
    assert _sha256(f1) != _sha256(f2)


def test_sha256_matches_stdlib(tmp_path):
    content = b"hello carrefour\n" * 1000
    f = tmp_path / "data.csv"
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert _sha256(f) == expected


def test_sha256_large_file_chunked(tmp_path):
    """Verify digest is correct even when content spans multiple read chunks."""
    content = b"x" * (9 * 1024 * 1024)   # 9 MB — crosses the 8 MB chunk boundary
    f = tmp_path / "large.bin"
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert _sha256(f) == expected


# ── verify_csv_checksums ──────────────────────────────────────────────────────

def _patch_loader(monkeypatch, tmp_path, csv_filename: str, content: bytes):
    """Helper: write a fake CSV, patch _CSV_DIR and _SOURCES to point at it."""
    (tmp_path / csv_filename).write_bytes(content)
    monkeypatch.setattr("src.data_loader._CSV_DIR", tmp_path)
    monkeypatch.setattr("src.data_loader._CHECKSUMS_FILE", tmp_path / "checksums.json")
    monkeypatch.setattr("src.data_loader._SOURCES", {
        "test_file": {"csv": csv_filename, "parquet": "test.parquet"},
    })


def test_verify_record_writes_checksums_json(tmp_path, monkeypatch):
    content = b"idarticu;desc\n001;milk\n"
    _patch_loader(monkeypatch, tmp_path, "test.csv", content)

    verify_csv_checksums(record=True)

    stored = json.loads((tmp_path / "checksums.json").read_text())
    assert "test.csv" in stored
    assert stored["test.csv"] == hashlib.sha256(content).hexdigest()


def test_verify_record_digest_matches_file(tmp_path, monkeypatch):
    content = b"a;b\n1;2\n3;4\n"
    _patch_loader(monkeypatch, tmp_path, "data.csv", content)
    verify_csv_checksums(record=True)

    stored_digest = json.loads((tmp_path / "checksums.json").read_text())["data.csv"]
    assert stored_digest == hashlib.sha256(content).hexdigest()


def test_verify_matching_digest_does_not_raise(tmp_path, monkeypatch):
    content = b"matching content"
    digest = hashlib.sha256(content).hexdigest()
    _patch_loader(monkeypatch, tmp_path, "test.csv", content)
    monkeypatch.setattr("src.data_loader._CSV_CHECKSUMS", {"test.csv": digest})

    verify_csv_checksums()   # must not raise


def test_verify_mismatched_digest_raises(tmp_path, monkeypatch):
    _patch_loader(monkeypatch, tmp_path, "test.csv", b"actual content")
    monkeypatch.setattr(
        "src.data_loader._CSV_CHECKSUMS",
        {"test.csv": "deadbeef" * 8},   # wrong digest
    )

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_csv_checksums()


def test_verify_missing_file_prints_skip_not_raises(tmp_path, monkeypatch, capsys):
    """A CSV that doesn't exist should be skipped gracefully, not crash."""
    monkeypatch.setattr("src.data_loader._CSV_DIR", tmp_path)
    monkeypatch.setattr("src.data_loader._CHECKSUMS_FILE", tmp_path / "checksums.json")
    monkeypatch.setattr("src.data_loader._SOURCES", {
        "missing": {"csv": "no_such_file.csv", "parquet": "x.parquet"},
    })

    verify_csv_checksums()   # must not raise

    out = capsys.readouterr().out
    assert "not found" in out or "skipping" in out
