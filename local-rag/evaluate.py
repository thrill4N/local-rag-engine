"""
evaluate.py
-----------
Retrieval + answer-quality evaluation harness.

Given a labeled eval set (question -> expected source file -> reference
answer), this computes:

  - Recall@k   : was a chunk from the expected source actually in the top-k
                 retrieved results?
  - MRR        : Mean Reciprocal Rank — how *highly* did the expected source
                 rank, on average, across all questions (1.0 = always #1)
  - Judge score: runs the full RAG pipeline (retrieve + generate) and uses
                 the LLM itself as a grader, scoring the generated answer
                 against the reference answer on a 1-5 scale
  - Calibration: compares the judge's score against the model's own
                 self-reported confidence — a model that's consistently
                 overconfident on answers the judge scores poorly is a
                 concrete, trackable miscalibration signal, not just a
                 vague "hallucinates sometimes."

Usage:
    python evaluate.py --eval-set eval/eval_set.json
    python evaluate.py --eval-set eval/eval_set.json --top-k 8 --skip-judge
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import ollama

from config import CHAT_MODEL, EMBED_MODEL, TOP_K
from errors import RAGError
from query import generate_answer, validate_citations
from rag_core import VectorStore, check_ollama_ready

JUDGE_PROMPT = """You are grading a RAG system's answer against a reference answer.

Question: {question}

Reference answer (ground truth): {reference}

System's answer: {generated}

Score the system's answer from 1-5 for factual correctness relative to the
reference answer (5 = fully correct and consistent, 1 = wrong or contradicts
the reference). Respond in EXACTLY this format, nothing else:
SCORE: <number>
REASON: <one sentence>"""


def load_eval_set(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["questions"]


def evaluate_retrieval(store: VectorStore, item: dict, top_k: int) -> dict:
    """Recall@k and reciprocal rank for a single eval question."""
    hits = store.query(item["question"], top_k=top_k)
    sources = [h["source"] for h in hits]

    hit_at_k = item["expected_source"] in sources
    reciprocal_rank = 1 / (sources.index(item["expected_source"]) + 1) if hit_at_k else 0.0

    return {"hit_at_k": hit_at_k, "reciprocal_rank": reciprocal_rank, "hits": hits}


def judge_answer(question: str, reference: str, generated: str) -> dict:
    """Ask the LLM to grade the generated answer against the reference."""
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, generated=generated)
    response = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    text = response["message"]["content"]

    score_match = re.search(r"SCORE:\s*(\d)", text)
    reason_match = re.search(r"REASON:\s*(.+)", text)
    return {
        "score": int(score_match.group(1)) if score_match else None,
        "reason": reason_match.group(1).strip() if reason_match else text.strip(),
    }


def run_evaluation(eval_set_path: Path, top_k: int, skip_judge: bool) -> None:
    check_ollama_ready(EMBED_MODEL)
    if not skip_judge:
        check_ollama_ready(CHAT_MODEL)

    store = VectorStore()
    questions = load_eval_set(eval_set_path)
    results = []

    for i, item in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {item['question'][:60]}...")

        retrieval = evaluate_retrieval(store, item, top_k)
        result = {
            "question": item["question"],
            "expected_source": item["expected_source"],
            "hit_at_k": retrieval["hit_at_k"],
            "reciprocal_rank": retrieval["reciprocal_rank"],
        }

        if not skip_judge:
            # run the actual (structured) generation step too, so we're
            # grading the full pipeline's output, not just the retriever
            parsed = generate_answer(item["question"], retrieval["hits"])
            parsed, fabricated = validate_citations(parsed, retrieval["hits"])
            judged = judge_answer(item["question"], item["reference_answer"], parsed.answer)

            result["generated_answer"] = parsed.answer
            result["model_confidence"] = parsed.confidence
            result["model_context_sufficient"] = parsed.context_sufficient
            result["fabricated_citations"] = fabricated
            result["judge_score"] = judged["score"]
            result["judge_reason"] = judged["reason"]
            # miscalibration: model was confident (>=0.7) but judge disagreed (<=2/5)
            result["overconfident"] = (
                judged["score"] is not None and judged["score"] <= 2 and parsed.confidence >= 0.7
            )

        results.append(result)

    print_report(results, top_k, skip_judge)
    save_results(results)


def print_report(results: list[dict], top_k: int, skip_judge: bool) -> None:
    n = len(results)
    recall_at_k = sum(r["hit_at_k"] for r in results) / n
    mrr = sum(r["reciprocal_rank"] for r in results) / n

    print("\n" + "=" * 60)
    print(f"RETRIEVAL  (top_k={top_k}, n={n} questions)")
    print("=" * 60)
    print(f"  Recall@{top_k}: {recall_at_k:.2%}   (expected source appeared in top-{top_k})")
    print(f"  MRR:       {mrr:.3f}     (1.0 = expected source always ranked #1)")

    misses = [r for r in results if not r["hit_at_k"]]
    if misses:
        print(f"\n  Missed retrievals ({len(misses)}):")
        for m in misses:
            print(f"    - \"{m['question'][:55]}\" (expected: {m['expected_source']})")

    if not skip_judge:
        scored = [r for r in results if r.get("judge_score") is not None]
        avg_score = sum(r["judge_score"] for r in scored) / len(scored) if scored else 0
        avg_conf = sum(r["model_confidence"] for r in scored) / len(scored) if scored else 0
        overconfident = [r for r in scored if r["overconfident"]]
        fabricated = [r for r in scored if r["fabricated_citations"]]

        print("\n" + "=" * 60)
        print("ANSWER QUALITY (LLM-as-judge, 1-5 scale)")
        print("=" * 60)
        print(f"  Average judge score:      {avg_score:.2f} / 5")
        print(f"  Average self-confidence:  {avg_conf:.2f} / 1.0")

        low_scores = [r for r in scored if r["judge_score"] <= 2]
        if low_scores:
            print(f"\n  Low-scoring answers ({len(low_scores)}):")
            for r in low_scores:
                print(f"    - \"{r['question'][:55]}\" -> {r['judge_score']}/5: {r['judge_reason']}")

        if overconfident:
            print(f"\n  ⚠ Overconfident answers ({len(overconfident)}) — model reported "
                  f"high confidence but judge scored poorly:")
            for r in overconfident:
                print(f"    - \"{r['question'][:55]}\" -> confidence "
                      f"{r['model_confidence']:.2f}, judge {r['judge_score']}/5")

        if fabricated:
            print(f"\n  ⚠ Fabricated citations ({len(fabricated)}) — cited a source "
                  f"never retrieved:")
            for r in fabricated:
                print(f"    - \"{r['question'][:55]}\" -> {r['fabricated_citations']}")

    print()


def save_results(results: list[dict]) -> None:
    out_path = Path("eval") / f"eval_results_{int(time.time())}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Full results saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="eval/eval_set.json", help="Path to eval set JSON")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--skip-judge", action="store_true",
                         help="Only evaluate retrieval, skip LLM-as-judge answer grading (faster)")
    args = parser.parse_args()

    eval_path = Path(args.eval_set)
    if not eval_path.exists():
        raise SystemExit(
            f"Eval set not found at {eval_path}. Copy eval/eval_set.example.json to "
            f"{eval_path} and fill it in with real questions from your documents."
        )

    try:
        run_evaluation(eval_path, args.top_k, args.skip_judge)
    except RAGError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
