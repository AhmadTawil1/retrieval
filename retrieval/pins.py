"""Model revision pinning, shared by every model the pipeline loads.

Why this module exists (LOG.md, Day 4): pins used to live only in embed.py and
covered only the two embedding models. The reranker and the generator were
loaded with no revision at all, so three of the four models in the A100 sweep
have no recorded revision anywhere. `configs/model_pins.yaml` is tracked, but it
is *written at runtime* on whichever machine loads a model first — and that
machine was a Colab VM that has since been deleted.

Two rules follow, and they are the point of this file:

1. Every model load goes through `revision_for()`. No bare `from_pretrained(id)`.
2. Every pinned revision is stamped into each result record's `prov` block by
   `provenance.stamp()`, not left in a file that may never be committed back.

Keys are the Hugging Face model IDs themselves, so a pin is self-describing and
cannot drift from the model it pins.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PINS_PATH = Path("configs/model_pins.yaml")

# Legacy keys from the original embed.py-only scheme, kept so the tracked pin
# for bge-small (frozen at git 04caf49, the A100 sweep's commit) still resolves.
LEGACY_KEYS = {
    "BAAI/bge-small-en-v1.5": "small",
    "BAAI/bge-base-en-v1.5": "base",
}


def load_pins() -> dict[str, str]:
    if PINS_PATH.exists():
        return yaml.safe_load(PINS_PATH.read_text()) or {}
    return {}


def _save_pin(key: str, revision: str) -> None:
    pins = load_pins()
    pins[key] = revision
    PINS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PINS_PATH.write_text(yaml.safe_dump(pins, sort_keys=True))


def _resolve(model_id: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().model_info(model_id).sha


def revision_for(model_id: str) -> str:
    """The frozen revision for `model_id`, resolving and pinning on first use.

    Checks the model ID first, then the legacy short key, so an existing
    `small:` line keeps working without being rewritten.
    """
    pins = load_pins()
    if model_id in pins:
        return pins[model_id]

    legacy = LEGACY_KEYS.get(model_id)
    if legacy and legacy in pins:
        return pins[legacy]

    revision = _resolve(model_id)
    _save_pin(model_id, revision)
    return revision


def all_pinned_revisions(model_ids: list[str]) -> dict[str, str]:
    """{model_id: revision} for every model the pipeline uses — goes into `prov`."""
    return {mid: revision_for(mid) for mid in model_ids}
