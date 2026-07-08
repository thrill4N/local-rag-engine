"""
errors.py
---------
Custom exception hierarchy. The goal is that when something breaks, the
person running the CLI sees a message that tells them what to *do* about
it, not a raw stack trace from three layers down in a dependency.
"""


class RAGError(Exception):
    """Base class for all errors raised by this project."""


class OllamaUnavailableError(RAGError):
    """Raised when the Ollama server can't be reached at all."""

    def __init__(self):
        super().__init__(
            "Could not reach Ollama. Is it running? Start it with `ollama serve`, "
            "or check that the Ollama desktop app is open."
        )


class ModelNotPulledError(RAGError):
    """Raised when Ollama is reachable but the requested model isn't installed."""

    def __init__(self, model_name: str):
        super().__init__(
            f"Model '{model_name}' isn't available locally. Pull it first with:\n"
            f"  ollama pull {model_name}"
        )


class DocumentLoadError(RAGError):
    """Raised when a specific file fails to parse (corrupt PDF, bad encoding, etc.)."""

    def __init__(self, filename: str, reason: str):
        super().__init__(f"Failed to load '{filename}': {reason}")


class EmptyCorpusError(RAGError):
    """Raised when a query is attempted before anything has been ingested."""

    def __init__(self):
        super().__init__(
            "No documents ingested yet. Run `python ingest.py --folder data` first."
        )
