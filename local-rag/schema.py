"""
schema.py
---------
Typed generation output, replacing a raw answer string with a structured
object the model fills in directly (via Ollama's JSON-schema-constrained
output). This exists specifically to fix a known blind spot: a plain
string answer gives no reliable way to tell whether a citation is real
(matches an actually-retrieved source) or hallucinated, and no way to
distinguish "confidently grounded" from "context was thin, best guess."

See: GitHub issue "Replace string-based grounding heuristic with typed,
structured answer output" for the full rationale.
"""

from pydantic import BaseModel, Field


class RAGAnswer(BaseModel):
    answer: str = Field(
        description="The answer to the question, based only on the provided context."
    )
    cited_sources: list[str] = Field(
        default_factory=list,
        description="Filenames of sources actually used to answer, exactly as shown "
                    "in the context labels (e.g. 'handbook.pdf'). Empty if the answer "
                    "wasn't found in the context.",
    )
    context_sufficient: bool = Field(
        description="True if the retrieved context contained enough information to "
                    "answer the question confidently. False if the answer is a guess, "
                    "partial, or the context didn't cover the question."
    )
    confidence: float = Field(
        ge=0, le=1,
        description="Model's self-reported confidence in the answer, from 0 to 1.",
    )
