"""
ingest.py
---------
Load every PDF/txt/md file from a folder, chunk it, embed the chunks with a
local Ollama model, and store them in a persistent Chroma collection.

Incremental: a manifest (ingest_manifest.json) tracks each file's content
hash, so re-running this script only re-embeds files that actually changed
— re-embedding unchanged files on every run is wasted GPU/CPU time and, in
a cloud-embedding setup, wasted money.

Usage:
    python ingest.py --folder data
    python ingest.py --folder data --reset      # wipe the collection first
"""

import argparse
import json
import sys
from pathlib import Path

from config import EMBED_MODEL, MANIFEST_PATH
from errors import RAGError
from rag_core import VectorStore, check_ollama_ready, chunk_document, load_documents


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2))


def main() -> None:
    # --- CLI arguments -----------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="data", help="Folder of PDFs/text files to ingest")
    parser.add_argument("--reset", action="store_true", help="Delete existing collection first")
    parser.add_argument("--force", action="store_true", help="Re-embed all files, ignoring manifest")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    try:
        # fail fast with a clear message rather than a confusing error mid-run
        check_ollama_ready(EMBED_MODEL)

        # --- Set up (or reset) the vector store -----------------------------
        store = VectorStore()
        manifest_path = Path(MANIFEST_PATH)
        manifest = {} if args.reset or args.force else load_manifest(manifest_path)

        if args.reset:
            # drop the whole collection, then recreate an empty one to write into
            store.client.delete_collection(store.collection.name)
            store = VectorStore()
            print("Existing collection cleared.")

        # --- Load every supported file in the folder ------------------------
        docs = load_documents(folder)
        if not docs:
            raise SystemExit(f"No .pdf/.txt/.md files found in {folder}")

        # --- Chunk + embed + store, skipping unchanged files -----------------
        total_chunks = 0
        skipped = 0
        for doc in docs:
            if manifest.get(doc.source) == doc.content_hash:
                skipped += 1
                continue   # file unchanged since last ingest — nothing to do

            chunks = chunk_document(doc)     # split this document's text into overlapping chunks
            store.add_chunks(chunks)          # embed the chunks and upsert into Chroma
            manifest[doc.source] = doc.content_hash
            total_chunks += len(chunks)
            print(f"  {doc.source}: {len(chunks)} chunks")

        save_manifest(manifest_path, manifest)

        print(f"\nIngested {len(docs) - skipped} document(s), {total_chunks} chunk(s) total "
              f"({skipped} unchanged file(s) skipped).")
        print(f"Collection now holds {store.count()} chunks.")

    except RAGError as e:
        # our own well-formed errors get a clean message, no stack trace noise
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
