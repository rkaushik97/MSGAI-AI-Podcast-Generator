from typing import TypedDict
from dataclasses import dataclass
from typing import Optional


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

@dataclass
class WERMetrics:
    """Detailed Word Error Rate metrics."""
    hits: int
    substitutions: int
    deletions: int
    insertions: int
    wer_score: float

@dataclass
class AudioMetrics:
    """Audio quality metrics."""
    sample_rate: int
    duration_s: float
    clipping_pct: float
    rms_dbfs: float
    spectral_flatness: float
    voice_activity_ratio: float

class AudioQualityScores(TypedDict):
    """Complete audio quality evaluation results."""
    audio_path: str
    transcript_path: str
    wer: float
    detailed_measures: dict
    audio_metrics: dict
    hypothesis_raw_preview: str
    reference_preview: str
