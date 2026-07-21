# Local RAG Engine

A fully local retrieval-augmented generation pipeline — documents in, grounded
and cited answers out, nothing ever leaves the machine. Built as a backend
retrieval-engineering artifact: the focus is correctness, evaluability, and
observability of the retrieval/generation core, not a polished end-user
product. See [Non-goals](#non-goals-and-why) for what's deliberately not here yet.

## Why this exists

Two things drove this project: getting hands-on with vector search
fundamentals (embeddings, similarity metrics, ANN indexing) beyond a
conceptual understanding of them, and building something that demonstrates
how I actually reason about retrieval system design — not just that I can
call an embedding API.

The target scenario I designed around: **regulated industries — legal,
medical — where documents cannot leave the device.** That constraint shaped
almost every decision below, and it's worth stating up front because it
explains choices (local inference, no cloud vector DB, CLI-first) that
would look wrong in a different context.

## Architecture

```
data/*.pdf, *.txt              (user-supplied documents)
      │
      ▼
  load_documents()              extract text; corrupt files logged & skipped,
      │                         don't abort the whole ingest run
      ▼
  chunk_document()               sentence-aware packing into ~800-char
      │                         windows with ~150-char overlap
      ▼
  embed_texts()                  Ollama (nomic-embed-text), local inference
      │
      ▼
  VectorStore (Chroma)           persisted to disk; cosine similarity space
  ingest_manifest.json           content-hash per file → skip unchanged
      │
      ▼  ── at query time ──
  VectorStore.query()            embed question → cosine top-k search
      │
      ▼
  build_prompt()                 stuff top-k chunks into a grounding prompt
      │
      ▼
  ollama.chat (llama3.2)         structured RAGAnswer: answer, cited
                                  sources, confidence, context_sufficient
  validate_citations()           cross-check cited_sources against hits
      │
      ▼
  log_query_event()              JSONL: sources, scores, per-stage latency
```

Three logical layers, deliberately kept separate:

- **`rag_core.py`** — the engine: loading, chunking, embedding, vector ops.
  No CLI concerns, no logging concerns — this is the part that would
  survive a rewrite of the interface layer untouched.
- **`ingest.py` / `query.py` / `evaluate.py`** — thin CLI entry points that
  compose the core into workflows.
- **`config.py` / `errors.py` / `logger.py`** — cross-cutting concerns,
  intentionally not baked into `rag_core.py` so they can change
  independently (e.g. swapping JSONL logging for structured logging to a
  real log aggregator later doesn't touch retrieval logic at all).

## Design decisions and reasoning

Each of these follows the same shape: **what I chose → why → what it costs
→ what I'd do with more time.** That's deliberate — a decision without a
named tradeoff usually means the tradeoff wasn't actually considered.

### Local inference over cloud APIs (Ollama)
**Why:** the target use case (legal/medical) treats "documents never leave
the device" as a hard requirement, not a preference — this isn't a cost
optimization, it's what makes the system usable at all in that context.
**Cost:** slower inference than a frontier hosted model, and the user is
responsible for their own hardware/GPU.
**Next step:** support a pluggable backend so the same interface can target
either Ollama or a hosted API, letting the deployment context decide —
without changing `rag_core.py`.

### No LangChain / no retrieval framework
**Why:** frameworks are right for shipping fast; they're wrong for an
artifact meant to demonstrate understanding of the mechanics. I wanted every
step — chunk boundary logic, what a similarity score actually means, what
the assembled prompt looks like — to be inspectable, not hidden behind an
abstraction I'd have to explain by reference to someone else's source code.
**Cost:** more code to write and maintain than `from langchain import ...`.
**Next step:** if this became a team project with multiple retrieval
strategies to swap between, a framework's abstractions would start paying
for themselves — the calculus changes with team size and velocity needs.

### Sentence-aware chunking with overlap, not fixed-size windows
**Why:** naive character-slicing can cut a fact in half mid-sentence.
Packing whole sentences into a size budget, then carrying the tail of one
chunk into the next, keeps facts intact and protects against losing context
right at a chunk boundary.
**Cost:** still content-agnostic — it doesn't know it's mid-table or
mid-list, which naive sentence-splitting handles no better than fixed-size
windows would for structured content.
**Next step:** structure-aware chunking (respecting markdown headers / PDF
layout) is the highest-leverage fix here — see the eval harness section for
how I'd actually detect when this is the bottleneck rather than guessing.

### Cosine similarity, via Chroma's HNSW index
**Why:** cosine measures directional similarity in embedding space
independent of magnitude — two chunks about the same topic should point the
same way even if one is more verbose. HNSW trades a small amount of recall
for large query-speed gains over brute-force search, which is the right
trade at any corpus size beyond trivial.
**Cost:** approximate search means occasionally missing the true nearest
neighbor — acceptable here, but worth knowing the failure mode exists.

### An eval harness, not just "it looks right"
**Why:** the fundamental discipline of retrieval engineering is that
quality claims need to be measurable, not asserted. `evaluate.py` computes
Recall@k and MRR for retrieval, and a separate LLM-as-judge pass for
end-to-end answer quality — deliberately two numbers, not one, because
retrieval can be perfect while generation still hallucinates, and the two
failure modes need to be diagnosable independently.
**Cost:** requires a hand-labeled eval set (question → correct source →
reference answer) — there's no way around this being manual work if the
ground truth is going to mean anything.
**Update:** the model now returns a typed `RAGAnswer` (see `schema.py`)
instead of a raw string — `cited_sources` is validated against the actual
retrieved set in `query.validate_citations()`, and the model self-reports
`context_sufficient`/`confidence`. `grounded` is now derived (sufficient
context claimed AND no citation to an unretrieved source), not guessed
from string characters. This replaced the earlier bracket-presence
heuristic, which couldn't distinguish a real citation from a hallucinated
one. Still not a full NLI-style faithfulness check (see Roadmap) — this
catches *fabricated attribution*, not subtler cases of the model
technically citing a real source while misreading its content.

### Structured JSONL logging, per-stage latency
**Why:** "it feels slow sometimes" isn't a diagnosis. Logging per-stage
timings (embed/search/generate) as one JSON line per query means a
bottleneck is visible in the data (generation dominates by 10-100x over
search, in practice), not guessed at.
**Cost:** log growth is unbounded as-is — fine for local single-user use,
not fine for a long-running deployed service.
Each log line now also carries the model's structured self-report
(`confidence`, `context_sufficient`) and any `fabricated_citations` —
concrete signals, not inferred from string content.
**Next step:** rotation/retention policy, and (per the faithfulness note
above) a check for citations that are real but misapplied, once logs are
being used to make real quality-monitoring decisions.

### CLI over a web UI
**Why:** the CLI was the right surface for building and proving the engine
— fast iteration, no UI work competing for time against the parts an
AI/backend engineering interview actually probes.
**This is a real gap, not a defensible end state:** "local" (where data
lives) and "CLI" (how a human interacts with it) are different axes. A
lawyer or clinician shouldn't need a terminal — a `localhost`-bound web
app satisfies the same privacy constraint with a usable interface. I'm
naming this directly rather than rationalizing it, because the honest
answer to "would you ship this to a non-technical user as-is" is no.

### No Docker (yet)
**Why:** Docker's value is proportional to how many environments something
needs to run consistently across. Right now this is a single-user local
tool — one person, one machine, one Ollama instance. Containerizing adds
real friction here specifically: Ollama typically wants direct host GPU
access, so either the Python app reaches out of its container to a host
Ollama process (cross-container networking for no real benefit at this
scale), or Ollama itself gets containerized too and now needs GPU
passthrough configured — meaningful complexity for zero users beyond
myself.
**When this flips:** the moment this needs to run on a teammate's machine,
a shared server, or in CI (e.g. running the eval harness automatically on
every change) — that's when the portability Docker buys is actually worth
its setup cost.

### Incremental ingestion via content-hash manifest
**Why:** re-running ingestion without this re-embeds every file every
time, even unchanged ones — wasted compute locally, wasted money against
a paid embedding API. A manifest mapping filename → content hash makes
re-ingestion idempotent: unchanged files are skipped, changed files are
re-embedded.
**Cost:** the manifest itself is a small piece of state that can drift
from the actual Chroma collection if someone edits the DB out-of-band —
acceptable for single-user local use, a real concern in any concurrent
or multi-writer setup.

## Non-goals (and why)

This is scoped deliberately as a **backend/retrieval engineering artifact**,
not a product. Explicitly out of scope for now:

- **Authentication / multi-tenancy** — meaningless without a real deployed
  service behind it, and building it now would mean designing against
  guesses rather than actual usage.
- **Web UI / file-picker UX** — a genuine gap for the target user (see
  above), deferred because it's downstream of the retrieval engine being
  trustworthy, not because it doesn't matter.
- **Deployment/Docker** — deferred until there's an actual second
  environment (a teammate, a server, CI) that needs it; see reasoning above.
- **Structure-aware chunking, reranking, hybrid search** — known
  next-highest-leverage improvements to retrieval quality, sequenced behind
  having an eval harness that can actually measure whether they help.

The ordering isn't arbitrary: each of these builds on trusting the layer
below it. Shipping a web UI in front of an unevaluated retrieval engine
just means rebuilding the UI once the eval harness reveals retrieval
problems underneath it.

## Setup

1. Install [Ollama](https://ollama.com/download).
2. Pull the models used here:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
# 1. Drop PDFs/.txt/.md files into data/, then ingest them
python ingest.py --folder data

# 2. Ask questions
python query.py "What are the main conclusions in these documents?"
python query.py                       # interactive mode

# 3. Evaluate retrieval + answer quality
cp eval/eval_set.example.json eval/eval_set.json
# edit eval_set.json with real questions from your own documents
python evaluate.py --eval-set eval/eval_set.json
```

Re-running `ingest.py` only re-embeds files that changed (via
`ingest_manifest.json`). Use `--force` to re-embed everything, `--reset` to
wipe the collection and start fresh.

## Evaluation harness

| Metric | Measures | Why it's separate from the others |
|---|---|---|
| **Recall@k** | Was the correct source file anywhere in the top-k retrieved chunks? | The floor metric — nothing downstream can be correct if the right chunk was never retrieved. |
| **MRR** | How highly the correct source ranked on average | Distinguishes "technically retrieved, buried at position 8" from "confidently ranked first." |
| **LLM-as-judge (1-5)** | Full pipeline (retrieve + generate) graded against a reference answer | Catches generation-side failure (hallucination, misreading correct context) that retrieval metrics can't see. |

Run with `--skip-judge` for a fast retrieval-only pass while iterating on
chunk size, `top_k`, or embedding model — the judge pass costs one extra
LLM call per question and is the slow part.

## Business framing

Local inference isn't a cost optimization here — for legal/medical
document handling, it's frequently the only compliance-viable option,
which makes it the pitch rather than a limitation to caveat. The metrics
that matter to a business stakeholder aren't Recall@k or MRR — they're
time-to-answer and percentage of questions resolved without escalating to
a human. The eval harness produces the engineering-side numbers;
translating "Recall@4 is 84%" into "this finds the right clause in
seconds instead of a 20-minute manual search" is the translation layer
that makes the project legible outside a technical audience.

## Roadmap

Roughly in priority order:

1. ~~**Typed answer schema + citation validation**~~ — done. `schema.py`'s
   `RAGAnswer` replaced the raw-string answer; `query.validate_citations()`
   catches fabricated source attribution that the old bracket heuristic
   couldn't. See [`schema.py`](./schema.py) and the Design Decisions
   section above.
2. **Reranking** — wider initial retrieval (`top_k=20`) + cross-encoder
   rerank before the final top-k reaches the LLM. Usually the single
   highest-ROI retrieval quality improvement available.
3. **Deeper faithfulness check** — the current fix catches citations to
   sources that were never retrieved. It doesn't yet catch a real citation
   applied to a misread chunk (source is correct, claim isn't). That needs
   an NLI-style entailment check between answer and cited chunk content.
4. **Hybrid search (BM25 + vector)** — catches exact terms (case numbers,
   drug names, statute citations) that embeddings alone can miss.
5. **Structure-aware chunking** — respect document layout instead of
   character windows.
6. **Web UI** — local, `localhost`-bound, no data leaves the machine;
   sequenced last because it depends on the layers above being trustworthy.
