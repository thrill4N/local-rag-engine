"""
query.py
--------
Ask questions against the ingested documents. Retrieves the most relevant
chunks from Chroma, stuffs them into a prompt, and asks the model to return
a structured RAGAnswer (not a raw string) — this lets us validate that any
cited source was actually retrieved, and lets the model self-report whether
the context was sufficient, rather than inferring "groundedness" from
whether the text happens to contain a bracket character.

Note: structured output means this is no longer token-streamed to the
terminal — the model must complete a syntactically valid JSON object before
it can be parsed, so we wait for the full response rather than printing
tokens as they arrive. This is a deliberate tradeoff (correctness of the
grounding signal over perceived responsiveness); see README.

Usage:
    python query.py "What does the document say about X?"
    python query.py --top-k 6 "..."
    python query.py            # interactive mode
"""

import argparse
import sys

import ollama

from config import CHAT_MODEL, TOP_K
from errors import AnswerParsingError, EmptyCorpusError, OllamaUnavailableError, RAGError
from logger import StageTimer, log_query_event
from rag_core import VectorStore, check_ollama_ready
from schema import RAGAnswer

# System prompt is what keeps the model "grounded" — it's told to only use
# the retrieved context and to set context_sufficient=False rather than
# guessing when the context doesn't cover the question.
SYSTEM_PROMPT = """You are a careful research assistant. Answer the user's \
question using ONLY the provided context. Only list a filename in \
cited_sources if you actually used content from that source to answer — \
never list a source you didn't rely on. List filenames exactly as they \
appear in the context labels, WITHOUT surrounding brackets (e.g. use \
"handbook.pdf", not "[handbook.pdf]"). If the context doesn't contain the \
answer, say so plainly in `answer`, leave `cited_sources` empty, and set \
`context_sufficient` to false rather than guessing."""


def build_prompt(question: str, hits: list[dict]) -> str:
    """Assemble the retrieved chunks + the question into a single prompt string."""
    # Each retrieved chunk is labeled with its source file and similarity
    # score, so the model (and you, reading the prompt) can see where each
    # piece of context came from and how confident the retrieval was.
    context_blocks = "\n\n".join(
        f"[{h['source']}] (relevance {h['score']:.2f})\n{h['text']}" for h in hits
    )
    return f"Context:\n{context_blocks}\n\nQuestion: {question}"


def generate_answer(question: str, hits: list[dict]) -> RAGAnswer:
    """Call the model with a JSON-schema-constrained response and validate it."""
    prompt = build_prompt(question, hits)
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        format=RAGAnswer.model_json_schema(),   # constrains output to valid JSON matching the schema
    )
    raw = response["message"]["content"]
    try:
        return RAGAnswer.model_validate_json(raw)
    except Exception as e:
        raise AnswerParsingError(raw, str(e)) from e


def _normalize_citation(raw: str) -> str:
    """
    The model sometimes echoes the bracketed format it saw in the context
    labels (e.g. "[handbook.pdf]") instead of the bare filename the schema
    asks for. Strip brackets/whitespace so a correctly-cited real source
    isn't flagged as fabricated purely due to formatting.
    """
    return raw.strip().strip("[]").strip()


def validate_citations(parsed: RAGAnswer, hits: list[dict]) -> tuple[RAGAnswer, list[str]]:
    """
    Cross-check the model's claimed citations against what was actually
    retrieved. A citation to a file that was never in `hits` is a concrete
    hallucination signal, not a guess — this is the fix for the old
    bracket-presence heuristic, which couldn't tell a real citation from a
    fabricated one.
    """
    retrieved_sources = {h["source"] for h in hits}
    fabricated = [
        s for s in parsed.cited_sources
        if _normalize_citation(s) not in retrieved_sources
    ]
    return parsed, fabricated


def answer(question: str, top_k: int = TOP_K, quiet: bool = False) -> RAGAnswer:
    store = VectorStore()
    if store.count() == 0:
        raise EmptyCorpusError()

    timer = StageTimer()

    # --- Retrieval: find the top_k chunks most similar to the question ------
    with timer.stage("search"):
        hits = store.query(question, top_k=top_k)

    # --- Generation: structured, validated answer ---------------------------
    with timer.stage("generate"):
        parsed = generate_answer(question, hits)
        parsed, fabricated_citations = validate_citations(parsed, hits)

    # groundedness is now derived, not guessed: sufficient context claimed
    # AND no citation pointed at a source that was never retrieved
    grounded = parsed.context_sufficient and not fabricated_citations

    if not quiet:
        print(f"\n{parsed.answer}\n")
        if parsed.cited_sources:
            print(f"Cited: {', '.join(parsed.cited_sources)}")
        if fabricated_citations:
            print(f"⚠ Fabricated citation(s) — not in retrieved context: "
                  f"{', '.join(fabricated_citations)}")
        print(f"Confidence: {parsed.confidence:.2f} | "
              f"Context sufficient: {parsed.context_sufficient} | "
              f"Grounded: {grounded}")
        print(f"(search: {timer.durations_ms['search']}ms, "
              f"generate: {timer.durations_ms['generate']}ms)")

    log_query_event(question, hits, parsed, fabricated_citations, grounded,
                     timer.durations_ms, top_k)
    return parsed


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