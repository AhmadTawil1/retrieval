"""Provenance stamping. REFERENCE.md ground rule 5: a result without
provenance is not a result — every record's `prov` block comes from here."""

from __future__ import annotations

import subprocess
from pathlib import Path

import faiss
import sentence_transformers
import torch

import chunker
import eval as eval_module


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _nvidia_smi_field(field: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip().splitlines()[0]
    except Exception:
        return None


def stamp(corpus_dir: Path = Path("corpus"), gold_path: Path = Path("data/gold.jsonl")) -> dict:
    if torch.cuda.is_available():
        gpu = _nvidia_smi_field("name") or torch.cuda.get_device_name(0)
        driver = _nvidia_smi_field("driver_version")
    else:
        gpu, driver = "none (CPU)", None

    return {
        "gpu": gpu,
        "driver": driver,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "sbert": sentence_transformers.__version__,
        "faiss": faiss.__version__,
        "git_sha": _git_sha(),
        "corpus_sha": chunker.corpus_sha(corpus_dir),
        "gold_sha": eval_module.gold_sha(gold_path),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(stamp(), indent=2))
