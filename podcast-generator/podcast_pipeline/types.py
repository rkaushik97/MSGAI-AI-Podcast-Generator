from typing import TypedDict

class Metadata(TypedDict):
    """Immutable data structure for LLM-generated podcast metadata."""
    HOST_GENDER: str
    GUEST_GENDER: str
    GUEST_NAME: str

class Script(TypedDict):
    """Immutable data structure for the podcast script."""
    topic: str
    dialogue: str
    metadata: Metadata

class ScriptQualityScores(TypedDict):
    """Immutable data structure for the quality scores returned by the LLM Judge."""
    relevance_score: int
    coherence_score: int
    