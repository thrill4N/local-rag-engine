"""
logger.py
---------
Structured observability for the query path. Every query is logged as one
JSON line with: the question, retrieved sources + scores, a per-stage
latency breakdown, and the model's structured self-report (confidence,
context_sufficient) plus a derived `grounded` flag based on validating
citations against what was actually retrieved — not a string heuristic.

This is what turns "it feels slow sometimes" into "embedding takes 40ms,
vector search 8ms, generation 1900ms — generation is the bottleneck," and
turns "the model seems to hallucinate sometimes" into a trackable rate.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from config import LOG_DIR, QUERY_LOG_FILE
from schema import RAGAnswer


@dataclass
class StageTimer:
    """Collects named stage durations across one query's lifecycle."""
    durations_ms: dict = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.durations_ms[name] = round((time.perf_counter() - start) * 1000, 1)


def log_query_event(
    question: str,
    hits: list[dict],
    parsed: RAGAnswer,
    fabricated_citations: list[str],
    grounded: bool,
    timings: dict,
    top_k: int,
) -> None:
    """Append one structured record describing a completed query."""
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(exist_ok=True)

    record = {
        "timestamp": time.time(),
        "question": question,
        "top_k": top_k,
        "retrieved": [{"source": h["source"], "score": round(h["score"], 4)} for h in hits],
        "top_score": round(hits[0]["score"], 4) if hits else None,
        "cited_sources": parsed.cited_sources,
        "fabricated_citations": fabricated_citations,   # non-empty = model cited an unretrieved source
        "context_sufficient": parsed.context_sufficient,   # model's own self-report
        "confidence": parsed.confidence,
        "grounded": grounded,   # derived: context_sufficient AND no fabricated citations
        "timings_ms": timings,
        "answer_length_chars": len(parsed.answer),
    }

    with open(log_dir / QUERY_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
