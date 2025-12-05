from dataclasses import dataclass

@dataclass(frozen=True)
class Metadata:
    """Immutable data structure for LLM-generated podcast metadata."""
    HOST_GENDER: str
    GUEST_GENDER: str
    GUEST_NAME: str

@dataclass(frozen=True)
class Script:
    """Immutable data structure for the podcast script."""
    topic: str
    dialogue: str
    metadata: Metadata
    