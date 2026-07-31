"""Fast unit tests for pins.py's revision-pin lookup/storage (LOG.md, Day 4 —
three of four A100 models had no recorded revision because the pin file lived
only on a Colab VM that got deleted). No network calls: `_resolve()` is
monkeypatched wherever a pin might need resolving, and `PINS_PATH` is
redirected to a tmp_path file so the real configs/model_pins.yaml is never
touched."""

import pytest

from retrieval import pins


@pytest.fixture(autouse=True)
def isolated_pins(tmp_path, monkeypatch):
    monkeypatch.setattr(pins, "PINS_PATH", tmp_path / "model_pins.yaml")


def _forbid_network(monkeypatch):
    def _fail(model_id):
        raise AssertionError(f"_resolve() called for {model_id!r} — should have hit the pin file")

    monkeypatch.setattr(pins, "_resolve", _fail)


def test_revision_for_legacy_key_no_network_call(monkeypatch):
    pins.PINS_PATH.write_text("small: abc123\n")
    _forbid_network(monkeypatch)
    assert pins.revision_for("BAAI/bge-small-en-v1.5") == "abc123"


def test_revision_for_direct_model_id_key_no_network_call(monkeypatch):
    pins.PINS_PATH.write_text("cross-encoder/ms-marco-MiniLM-L6-v2: def456\n")
    _forbid_network(monkeypatch)
    assert pins.revision_for("cross-encoder/ms-marco-MiniLM-L6-v2") == "def456"


def test_load_pins_never_rewrites_the_file():
    content = "small: abc123\nBAAI/bge-base-en-v1.5: def456\n"
    pins.PINS_PATH.write_text(content)
    before = pins.PINS_PATH.read_bytes()

    pins.load_pins()

    assert pins.PINS_PATH.read_bytes() == before


def test_revision_for_unpinned_id_resolves_once_then_reads_from_file(monkeypatch):
    calls = []
    monkeypatch.setattr(pins, "_resolve", lambda model_id: calls.append(model_id) or "resolved-sha")

    first = pins.revision_for("some/new-model")
    second = pins.revision_for("some/new-model")

    assert first == second == "resolved-sha"
    assert calls == ["some/new-model"]  # second call read the file, didn't re-resolve
    assert pins.load_pins()["some/new-model"] == "resolved-sha"


def test_stamp_model_revisions_covers_all_four_model_ids(monkeypatch):
    from retrieval import embed, pipeline, provenance

    monkeypatch.setattr(pins, "_resolve", lambda model_id: f"sha-for-{model_id}")
    monkeypatch.setattr(provenance, "_git_sha", lambda: "test-sha")
    monkeypatch.setattr(provenance, "_nvidia_smi_field", lambda field: None)

    prov = provenance.stamp()

    expected_ids = {*embed.MODEL_IDS.values(), pipeline.RERANKER_MODEL_ID, pipeline.GENERATOR_MODEL_ID}
    assert set(prov["model_revisions"]) == expected_ids
    assert all(prov["model_revisions"][mid] == f"sha-for-{mid}" for mid in expected_ids)
