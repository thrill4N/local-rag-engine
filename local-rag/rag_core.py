"""
rag_core.py
-----------
Core building blocks for the local RAG pipeline:

  1. Document loading (PDF + plain text)
  2. Chunking (recursive, overlap-aware)
  3. Embedding (via a local Ollama model)
  4. Vector storage / retrieval (via ChromaDB, persisted to disk)

Everything here is deliberately framework-free (no LangChain/LlamaIndex) so
you can see exactly what happens to your text between "file on disk" and
"context fed to the LLM".
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
import ollama

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from config import CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, DB_DIR, EMBED_MODEL
from errors import DocumentLoadError, ModelNotPulledError, OllamaUnavailableError

# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

@dataclass
class RawDocument:
    """One ingested file, before any chunking has happened."""
    source: str       # filename, used later for citation
    text: str          # full extracted text of the file
    content_hash: str  # sha1 of the raw file bytes, used for incremental ingestion


def _hash_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def load_pdf(path: Path) -> RawDocument:
    """Extract text page-by-page from a PDF, tagging each page for context."""
    try:
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""   # extract_text() can return None
            if page_text.strip():                    # skip blank/image-only pages
                pages.append(f"[page {i + 1}]\n{page_text}")
        return RawDocument(source=path.name, text="\n\n".join(pages), content_hash=_hash_file(path))
    except PdfReadError as e:
        # corrupt/encrypted/malformed PDFs shouldn't kill the whole ingest run —
        # surface a clear error so the caller can decide to skip and continue
        raise DocumentLoadError(path.name, f"unreadable PDF ({e})") from e


def load_text(path: Path) -> RawDocument:
    """Read a plain .txt or .md file as-is."""
    try:
        # errors="ignore" so a stray non-UTF-8 byte doesn't crash the whole ingest
        text = path.read_text(encoding="utf-8", errors="ignore")
        return RawDocument(source=path.name, text=text, content_hash=_hash_file(path))
    except OSError as e:
        raise DocumentLoadError(path.name, str(e)) from e


def load_documents(folder: Path, skip_errors: bool = True) -> list[RawDocument]:
    """
    Walk a folder recursively and load every supported file type in it.

    If skip_errors is True (default), a single bad file (corrupt PDF, unreadable
    encoding) is logged and skipped rather than aborting the whole ingest run.
    """
    docs = []
    for path in sorted(folder.rglob("*")):        # rglob("*") = recursive, all files
        try:
            if path.suffix.lower() == ".pdf":
                docs.append(load_pdf(path))
            elif path.suffix.lower() in {".txt", ".md"}:
                docs.append(load_text(path))
            # anything else (images, .docx, etc.) is silently skipped for now
        except DocumentLoadError as e:
            if skip_errors:
                print(f"  [skipped] {e}")
                continue
            raise
    return docs


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """One retrievable unit of text, ready to be embedded and stored."""
    id: str            # stable unique id, used as the Chroma primary key
    text: str          # the chunk's actual text
    source: str        # which file this chunk came from (for citations)
    chunk_index: int   # position of this chunk within its source document


def _split_into_sentencesish(text: str) -> list[str]:
    """Cheap sentence/paragraph splitter — good enough for chunk boundaries."""
    text = re.sub(r"\s+", " ", text).strip()   # collapse all whitespace/newlines to single spaces
    # Split after ., !, or ? when followed by a space + capital letter/digit.
    # This is a heuristic, not a real NLP sentence tokenizer — it's fine for
    # deciding *where a chunk boundary is allowed*, not for grammar analysis.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p for p in parts if p]   # drop any empty strings from the split


def chunk_document(doc: RawDocument, chunk_size: int = CHUNK_SIZE,
                    overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """
    Recursive/greedy chunking: pack sentences into a window of ~chunk_size
    characters, then start the next window `overlap` characters back so
    context isn't lost at chunk boundaries.
    """
    sentences = _split_into_sentencesish(doc.text)
    chunks: list[Chunk] = []
    current = ""    # the chunk currently being built up
    idx = 0         # running count of chunks emitted so far for this document

    for sentence in sentences:
        # if adding this sentence would overflow the window, close out the
        # current chunk first (unless it's still empty)
        if len(current) + len(sentence) + 1 > chunk_size and current:
            chunks.append(_make_chunk(current, doc.source, idx))
            idx += 1
            # seed the new chunk with the tail of the previous one, so
            # a fact split across the boundary still appears in one chunk
            current = current[-overlap:] + " " + sentence
        else:
            current = (current + " " + sentence).strip()

    if current.strip():             # flush whatever's left after the loop ends
        chunks.append(_make_chunk(current, doc.source, idx))

    return chunks


def _make_chunk(text: str, source: str, idx: int) -> Chunk:
    # hash source+index+text-prefix so the id is stable across re-runs
    # (re-ingesting the same file produces the same ids -> upsert, not duplicate)
    chunk_id = hashlib.sha1(f"{source}-{idx}-{text[:50]}".encode()).hexdigest()[:16]
    return Chunk(id=chunk_id, text=text.strip(), source=source, chunk_index=idx)


# ---------------------------------------------------------------------------
# 3. Embedding
# ---------------------------------------------------------------------------

def check_ollama_ready(model_name: str) -> None:
    """
    Fail fast with a clear message if Ollama isn't running or the model
    hasn't been pulled, instead of letting a confusing connection error
    surface three layers down inside the ollama client.
    """
    try:
        available = {m["model"].split(":")[0] for m in ollama.list()["models"]}
    except Exception as e:
        raise OllamaUnavailableError() from e

    if model_name.split(":")[0] not in available:
        raise ModelNotPulledError(model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using a local Ollama embedding model."""
    try:
        # Ollama's Python client embeds one string per call, so we loop here.
        # For large corpora you could parallelize this with a thread pool.
        return [ollama.embeddings(model=EMBED_MODEL, prompt=t)["embedding"] for t in texts]
    except Exception as e:
        # connection errors surface here too (not just at startup check),
        # e.g. if Ollama crashes mid-run
        raise OllamaUnavailableError() from e


# ---------------------------------------------------------------------------
# 4. Vector store (Chroma)
# ---------------------------------------------------------------------------

class VectorStore:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(self, persist_dir: str = DB_DIR, collection_name: str = COLLECTION_NAME):
        # PersistentClient writes the index to disk so it survives between runs
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},   # use cosine similarity for search
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed a batch of chunks and upsert them into the collection."""
        if not chunks:
            return
        embeddings = embed_texts([c.text for c in chunks])
        self.collection.upsert(
            ids=[c.id for c in chunks],                      # primary keys
            embeddings=embeddings,                            # vectors for similarity search
            documents=[c.text for c in chunks],               # raw text, returned on query
            metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
        )

    def query(self, question: str, top_k: int = 4) -> list[dict]:
        """Embed the question and return the top_k most similar chunks."""
        query_embedding = embed_texts([question])[0]   # embed the question the same way as chunks
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        hits = []
        # Chroma returns lists-of-lists (one outer list per query embedding);
        # since we only sent one query, we index [0] to unwrap it.
        for text, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            # Chroma returns cosine *distance*; convert to a similarity score (1 = identical)
            hits.append({"text": text, "source": meta["source"], "score": 1 - dist})
        return hits

    def count(self) -> int:
        """Number of chunks currently stored — used to check if ingest has run."""
        return self.collection.count()
