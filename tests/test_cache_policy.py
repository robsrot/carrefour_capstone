from src.utils import cache_status, should_use_cache, write_artifact_metadata


def test_existing_artifact_is_cache_hit_without_metadata_manifest(tmp_path):
    artifact = tmp_path / "cached.parquet"
    artifact.write_text("cached", encoding="utf-8")

    status = cache_status(artifact, metadata={"stage": "current"})

    assert status["cache_hit"] is True
    assert status["metadata_status"] == "missing_manifest"
    assert status["reason"] == "artifact exists; metadata manifest missing ignored"


def test_existing_artifact_is_cache_hit_with_metadata_mismatch(tmp_path):
    artifact = tmp_path / "cached.parquet"
    artifact.write_text("cached", encoding="utf-8")
    write_artifact_metadata(artifact, {"stage": "old"})

    status = cache_status(artifact, metadata={"stage": "current"})

    assert status["cache_hit"] is True
    assert status["metadata_status"] == "mismatch"
    assert status["reason"] == "artifact exists; metadata hash mismatch ignored"


def test_force_and_disabled_cache_still_override_existing_artifact(tmp_path):
    artifact = tmp_path / "cached.parquet"
    artifact.write_text("cached", encoding="utf-8")

    assert should_use_cache(artifact, force=True, metadata={"stage": "current"}) is False
    assert should_use_cache(artifact, use_cached=False, metadata={"stage": "current"}) is False


def test_missing_artifact_is_not_cache_hit(tmp_path):
    artifact = tmp_path / "missing.parquet"

    status = cache_status(artifact, metadata={"stage": "current"})

    assert status["cache_hit"] is False
    assert status["reason"] == "artifact missing"
