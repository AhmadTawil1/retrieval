"""PDF extraction and recursive character chunking for the retrieval corpus.

Chunk size and overlap are expressed in tokens (per the grid), but splitting itself
walks a list of character separators from coarse to fine, only descending into a
piece when it is still over budget. Every chunk keeps {doc_id, char_start, char_end}
relative to that document's full extracted text, so gold spans (see relevance.py,
day 2) stay meaningful across every chunk-size level.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pypdf

EXTRACTION_LIBRARY = "pypdf"
EXTRACTION_LIBRARY_VERSION = pypdf.__version__

# Coarse -> fine. "" is the hard-cut fallback for a piece with no separator at all.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Token counting piggybacks on the base embedding model's tokenizer (bge-base-en-v1.5),
# so "256 tokens" here means the same thing it will mean in embed.py.
_TOKENIZER = None


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    char_start: int
    char_end: int
    text: str


def _token_length(text: str) -> int:
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    return len(_TOKENIZER.encode(text, add_special_tokens=False))


def extract_text(pdf_path: Path) -> tuple[str, int]:
    """Returns (full_text, page_count). Pages are joined with a blank line."""
    reader = pypdf.PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages), len(reader.pages)


def _hard_slice(text: str, offset: int, size: int) -> list[tuple[str, int, int]]:
    """Last-resort split with no separator: binary-search each boundary rather than
    growing one character at a time. A linear grow-and-retokenize loop is O(n^2)
    tokenizer calls on a long unbroken run (e.g. PDF-extracted code or table text
    with no spaces), which is slow enough to matter on real documents."""
    slices: list[tuple[str, int, int]] = []
    n = len(text)
    start = 0
    while start < n:
        lo, hi, best = start + 1, n, start + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if _token_length(text[start:mid]) <= size:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        slices.append((text[start:best], offset + start, offset + best))
        start = best
    return slices


def _atomize(text: str, offset: int, size: int, separators: list[str]) -> list[tuple[str, int, int]]:
    """Recursively split `text` into pieces each <= `size` tokens, tracking each
    piece's (start, end) offset relative to the original full document text."""
    if _token_length(text) <= size:
        return [(text, offset, offset + len(text))]
    if not separators:
        return _hard_slice(text, offset, size)
    sep, rest = separators[0], separators[1:]
    if sep == "":
        return _hard_slice(text, offset, size)
    if sep not in text:
        return _atomize(text, offset, size, rest)

    parts = text.split(sep)
    reattached = [p + sep for p in parts[:-1]] + [parts[-1]]

    atoms: list[tuple[str, int, int]] = []
    local_offset = offset
    for part in reattached:
        if part:
            atoms.extend(_atomize(part, local_offset, size, rest))
        local_offset += len(part)
    return atoms


def _merge(atoms: list[tuple[str, int, int]], size: int, overlap_tokens: int) -> list[tuple[str, int, int]]:
    """Greedily pack atoms into chunks up to `size` tokens, carrying trailing atoms
    forward into the next chunk so consecutive chunks overlap by ~overlap_tokens."""
    chunks: list[tuple[str, int, int]] = []
    current: list[tuple[str, int, int]] = []
    current_len = 0
    i = 0
    while i < len(atoms):
        atom_text, _, _ = atoms[i]
        atom_len = _token_length(atom_text)
        if current and current_len + atom_len > size:
            start, end = current[0][1], current[-1][2]
            chunks.append(("".join(a[0] for a in current), start, end))

            overlap_atoms: list[tuple[str, int, int]] = []
            overlap_len = 0
            for a in reversed(current):
                a_len = _token_length(a[0])
                if overlap_len + a_len > overlap_tokens:
                    break
                overlap_atoms.insert(0, a)
                overlap_len += a_len

            if len(overlap_atoms) == len(current):
                # Trimming dropped nothing (the whole `current` already fit inside
                # the overlap budget). If the new atom still doesn't fit alongside
                # it, keeping `current` unchanged would retry this exact state
                # forever. Drop overlap for this one boundary instead.
                overlap_atoms, overlap_len = [], 0
            current, current_len = overlap_atoms, overlap_len
            continue
        current.append(atoms[i])
        current_len += atom_len
        i += 1

    if current:
        start, end = current[0][1], current[-1][2]
        chunks.append(("".join(a[0] for a in current), start, end))
    return chunks


def chunk_document(text: str, doc_id: str, size: int, overlap: float) -> list[Chunk]:
    """`size` in tokens, `overlap` as a fraction of `size` (e.g. 0.15)."""
    atoms = _atomize(text, 0, size, SEPARATORS)
    merged = _merge(atoms, size, overlap_tokens=round(size * overlap))
    return [Chunk(doc_id=doc_id, char_start=s, char_end=e, text=t) for t, s, e in merged]


def chunk_corpus(corpus_dir: Path, size: int, overlap: float) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pdf_path in sorted(corpus_dir.glob("*.pdf")):
        text, _ = extract_text(pdf_path)
        chunks.extend(chunk_document(text, doc_id=pdf_path.stem, size=size, overlap=overlap))
    return chunks


def corpus_stats(corpus_dir: Path) -> dict:
    per_doc: dict[str, dict] = {}
    total_pages = total_tokens = 0
    for pdf_path in sorted(corpus_dir.glob("*.pdf")):
        text, pages = extract_text(pdf_path)
        tokens = _token_length(text)
        per_doc[pdf_path.stem] = {"pages": pages, "tokens": tokens}
        total_pages += pages
        total_tokens += tokens
    return {
        "doc_count": len(per_doc),
        "total_pages": total_pages,
        "total_tokens": total_tokens,
        "per_doc": per_doc,
        "extraction_library": EXTRACTION_LIBRARY,
        "extraction_library_version": EXTRACTION_LIBRARY_VERSION,
    }


def corpus_sha(corpus_dir: Path) -> str:
    """SHA1 over sorted (filename, bytes) pairs — stable regardless of mtimes."""
    h = hashlib.sha1()
    for pdf_path in sorted(corpus_dir.glob("*.pdf")):
        h.update(pdf_path.name.encode("utf-8"))
        h.update(pdf_path.read_bytes())
    return h.hexdigest()


if __name__ == "__main__":
    import json

    stats = corpus_stats(Path("corpus"))
    stats["corpus_sha"] = corpus_sha(Path("corpus"))
    print(json.dumps(stats, indent=2))
