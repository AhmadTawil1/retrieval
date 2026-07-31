"""Hand-labelling aid for data/gold.jsonl — finds exact character spans so
they don't have to be counted by hand. Gold spans must be measured against
the same extracted text the pipeline actually indexes (pypdf output, not the
PDF's visual layout), which is exactly what this reads.

Usage:
  uv run python scripts/label_helper.py dump RAG                  # read a doc with offset markers
  uv run python scripts/label_helper.py find RAG "cross-encoder"   # locate a phrase's exact span
  uv run python scripts/label_helper.py list                       # available doc_ids
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retrieval import chunker

CORPUS_DIR = Path("corpus")
MARKER_EVERY = 500  # characters, for `dump`'s ruler


def _text_for(doc_id: str) -> str:
    pdf_path = CORPUS_DIR / f"{doc_id}.pdf"
    if not pdf_path.exists():
        available = ", ".join(sorted(p.stem for p in CORPUS_DIR.glob("*.pdf")))
        raise SystemExit(f"no {pdf_path} — available doc_ids: {available}")
    text, _ = chunker.extract_text(pdf_path)
    return text


def cmd_list(_args: argparse.Namespace) -> None:
    for p in sorted(CORPUS_DIR.glob("*.pdf")):
        print(p.stem)


def cmd_dump(args: argparse.Namespace) -> None:
    text = _text_for(args.doc_id)
    print(f"# {args.doc_id} — {len(text)} chars total\n")
    for start in range(0, len(text), MARKER_EVERY):
        end = min(start + MARKER_EVERY, len(text))
        print(f"--- char {start} ---")
        print(text[start:end])
    print(f"--- char {len(text)} (end) ---")


def cmd_find(args: argparse.Namespace) -> None:
    text = _text_for(args.doc_id)
    phrase = args.phrase
    matches = []
    cursor = 0
    while True:
        idx = text.find(phrase, cursor)
        if idx == -1:
            break
        matches.append(idx)
        cursor = idx + 1

    if not matches:
        print(f"no match for {phrase!r} in {args.doc_id} ({len(text)} chars)")
        return

    for idx in matches:
        start, end = idx, idx + len(phrase)
        ctx_start, ctx_end = max(0, start - 60), min(len(text), end + 60)
        print(f'"doc_id": "{args.doc_id}", "char_start": {start}, "char_end": {end}')
        print(f"  ...{text[ctx_start:start]}[[{text[start:end]}]]{text[end:ctx_end]}...")
    if len(matches) > 1:
        print(f"\n{len(matches)} matches — pick the right one, or narrow the phrase")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)

    p_list = sub.add_parser("list", help="list available doc_ids")
    p_list.set_defaults(func=cmd_list)

    p_dump = sub.add_parser("dump", help="print a doc's extracted text with char-offset markers")
    p_dump.add_argument("doc_id")
    p_dump.set_defaults(func=cmd_dump)

    p_find = sub.add_parser("find", help="find the exact char_start/char_end of a phrase")
    p_find.add_argument("doc_id")
    p_find.add_argument("phrase")
    p_find.set_defaults(func=cmd_find)

    args = parser.parse_args()
    args.func(args)
