"""
logger.py
---------
Structured observability for the query path. Every query is logged as one
JSON line with: the question, retrieved sources + scores, a per-stage
latency breakdown, and whether the answer appears "grounded" (cites a
source) vs. a fallback ("not in the context").

This is what turns "it feels slow sometimes" into "embedding takes 40ms,
vector search 8ms, generation 1900ms — generation is the bottleneck."
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from config import LOG_DIR, QUERY_LOG_FILE


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
    answer: str,
    timings: dict,
    top_k: int,
) -> None:
    """Append one structured record describing a completed query."""
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(exist_ok=True)

    # crude "did this answer actually use retrieved context" heuristic —
    # good enough to flag a spike in ungrounded answers, not a rigorous metric
    grounded = "[" in answer and "]" in answer

    record = {
        "timestamp": time.time(),
        "question": question,
        "top_k": top_k,
        "retrieved": [{"source": h["source"], "score": round(h["score"], 4)} for h in hits],
        "top_score": round(hits[0]["score"], 4) if hits else None,
        "grounded": grounded,
        "timings_ms": timings,
        "answer_length_chars": len(answer),
    }

    with open(log_dir / QUERY_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
