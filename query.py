"""
query.py
--------
Ask questions against the ingested documents. Retrieves the most relevant
chunks from Chroma, stuffs them into a prompt, and streams a grounded answer
from a local Ollama chat model.

Every query is logged to logs/queries.jsonl with a per-stage latency
breakdown (embed / search / generate) and the retrieved sources — see
logger.py.

Usage:
    python query.py "What does the document say about X?"
    python query.py --top-k 6 "..."
    python query.py            # interactive mode, no argument
"""

import argparse
import sys

import ollama

from config import CHAT_MODEL, TOP_K
from errors import EmptyCorpusError, OllamaUnavailableError, RAGError
from logger import StageTimer, log_query_event
from rag_core import VectorStore, check_ollama_ready

# System prompt is what keeps the model "grounded" — it's told to only use
# the retrieved context and to admit when the answer isn't in it, rather
# than falling back on its own (unverifiable, un-cited) general knowledge.
SYSTEM_PROMPT = """You are a careful research assistant. Answer the user's \
question using ONLY the provided context. If the context doesn't contain \
the answer, say so plainly instead of guessing. Cite sources by filename \
in square brackets, e.g. [handbook.pdf]."""


def build_prompt(question: str, hits: list[dict]) -> str:
    """Assemble the retrieved chunks + the question into a single prompt string."""
    # Each retrieved chunk is labeled with its source file and similarity
    # score, so the model (and you, reading the prompt) can see where each
    # piece of context came from and how confident the retrieval was.
    context_blocks = "\n\n".join(
        f"[{h['source']}] (relevance {h['score']:.2f})\n{h['text']}" for h in hits
    )
    return (
        f"Context:\n{context_blocks}\n\n"
        f"Question: {question}\n\n"
        "Answer, citing sources in square brackets:"
    )


def answer(question: str, top_k: int = TOP_K, quiet: bool = False) -> str:
    store = VectorStore()
    if store.count() == 0:
        raise EmptyCorpusError()

    timer = StageTimer()

    # --- Retrieval: find the top_k chunks most similar to the question ------
    with timer.stage("search"):
        hits = store.query(question, top_k=top_k)
    prompt = build_prompt(question, hits)

    # --- Generation: stream the answer token-by-token from the local LLM ---
    full_answer = ""
    with timer.stage("generate"):
        stream = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=True,   # yields incremental chunks instead of waiting for the full reply
        )
        if not quiet:
            print()
        for chunk in stream:
            token = chunk["message"]["content"]
            full_answer += token
            if not quiet:
                # print each streamed token immediately, without a trailing newline,
                # so the answer appears to "type itself out" in the terminal
                print(token, end="", flush=True)
        if not quiet:
            print("\n")

    if not quiet:
        sources = sorted({h["source"] for h in hits})
        print(f"(retrieved from: {', '.join(sources)} | "
              f"search: {timer.durations_ms['search']}ms, "
              f"generate: {timer.durations_ms['generate']}ms)")

    log_query_event(question, hits, full_answer, timer.durations_ms, top_k)
    return full_answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="Question to ask")   # optional positional arg
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of chunks to retrieve")
    args = parser.parse_args()

    try:
        check_ollama_ready(CHAT_MODEL)   # fail fast with a clear message if not ready

        if args.question:
            # single-shot mode: answer one question and exit
            answer(args.question, top_k=args.top_k)
            return

        # no question given on the command line -> drop into a REPL loop instead
        print("Interactive mode — type a question, or 'exit' to quit.\n")
        while True:
            question = input("> ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if question:
                try:
                    answer(question, top_k=args.top_k)
                except RAGError as e:
                    # in interactive mode, one bad turn shouldn't kill the session
                    print(f"Error: {e}\n")

    except RAGError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
