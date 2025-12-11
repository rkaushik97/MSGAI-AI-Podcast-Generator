import re
import json
import math
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

import numpy as np
import soundfile as sf
import librosa
import jiwer
from transformers import pipeline, Pipeline
import torch

from .types import AudioQualityScores, AudioMetrics, WERMetrics
from .utils import ModelConstants, LOGGER

class AudioQualityAnalyzer:
    """
    Analyzes audio quality and computes Word Error Rate (WER) for TTS-generated audio.
    """
    def __init__(
        self, 
        audio_path: str, 
        transcript_md_path: str, 
        asr_model_id: str = ModelConstants.ASR_MODEL_ID,
        asr_device: int = ModelConstants.ASR_DEVICE,
        target_sr: int = 16000,
        chunk_seconds: float = 30.0
    ):
        self.audio_path = Path(audio_path)
        self.transcript_md_path = Path(transcript_md_path)
        self.asr_model_id = asr_model_id
        self.asr_device = asr_device
        self.target_sr = target_sr
        self.chunk_seconds = chunk_seconds
        self._asr_pipeline: Optional[Pipeline] = None
        
        LOGGER.info(f"Initialized AudioQualityAnalyzer for audio: {self.audio_path.name}")


    def _read_audio_mono(self, path: Path, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Read an audio file into a mono float32 numpy array."""
        samples, sr = sf.read(str(path), dtype="float32")
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)
        if target_sr is not None and sr != target_sr:
            samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return samples.astype(np.float32), sr
    
    def _compute_clipping_percentage(self, samples: np.ndarray, clip_threshold: float = 0.999) -> float:
        """Calculate percentage of samples that are clipped."""
        n_total = samples.shape[0]
        n_clipped = int(np.sum(np.abs(samples) >= clip_threshold))
        return float(100.0 * n_clipped / max(1, n_total))

    def _compute_rms_dbfs(self, samples: np.ndarray) -> float:
        """Calculate RMS level in dBFS."""
        rms = np.sqrt(np.mean(samples ** 2) + 1e-12)
        return 20.0 * math.log10(rms + 1e-12)

    def _compute_spectral_flatness(self, samples: np.ndarray, sr: int, n_fft: int = 2048) -> float:
        """Calculate spectral flatness (noisiness measure)."""
        S = np.abs(librosa.stft(samples, n_fft=n_fft))
        flatness = librosa.feature.spectral_flatness(S=S)
        return float(np.mean(flatness))
    
    def _compute_voice_activity_ratio(self, samples: np.ndarray, sr: int, top_db: int = 40) -> float:
        """Calculate ratio of time with voice activity."""
        intervals = librosa.effects.split(samples, top_db=top_db)
        active_samples = sum((end - start) for start, end in intervals)
        return float(active_samples) / float(len(samples))
    
    def _extract_dialogue_from_md(self, md_path: Path) -> str:
        """Extract dialogue from markdown file."""
        text = md_path.read_text(encoding="utf-8")
        parts = re.split(r"---\s*Dialogue\s*---", text, flags=re.IGNORECASE)
        
        if len(parts) < 2:
            m = re.search(r"(HOST\s*:)", text, flags=re.IGNORECASE)
            dialogue = text[m.start():] if m else text
        else:
            dialogue = parts[1]
        
        # Remove metadata and speaker labels
        dialogue = re.sub(r"^#.*$", "", dialogue, flags=re.MULTILINE)
        dialogue = re.sub(r"^Metadata:.*$", "", dialogue, flags=re.MULTILINE)
        dialogue = re.sub(r"^-{3,}", "", dialogue, flags=re.MULTILINE)
        
        lines = [ln.strip() for ln in dialogue.splitlines() if ln.strip() != ""]
        cleaned = []
        for ln in lines:
            ln2 = re.sub(r"^(HOST|GUEST)\s*:\s*", "", ln, flags=re.IGNORECASE)
            ln2 = re.sub(r"\b(HOST|GUEST)\b\s*:\s*", "", ln2, flags=re.IGNORECASE)
            cleaned.append(ln2)
        return " ".join(cleaned)

    def _load_asr_pipeline(self) -> Pipeline:
        """Load ASR pipeline."""
        if self._asr_pipeline is not None:
            return self._asr_pipeline
            
        try:
            LOGGER.info(f"Loading ASR pipeline: {self.asr_model_id}")
            self._asr_pipeline = pipeline(
                "automatic-speech-recognition", 
                model=self.asr_model_id, 
                device=self.asr_device
            )
            return self._asr_pipeline
        except Exception as e:
            LOGGER.warning(f"ASR GPU pipeline failed, falling back to CPU: {e}")
            self._asr_pipeline = pipeline(
                "automatic-speech-recognition", 
                model=self.asr_model_id, 
                device=-1
            )
            return self._asr_pipeline
    
    def evaluate(self) -> AudioQualityScores:
        """
        Perform complete audio quality evaluation including WER calculation.
        """
        LOGGER.info(f"Starting audio quality evaluation for: {self.audio_path.name}")
        
        if not self.audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")
        if not self.transcript_md_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {self.transcript_md_path}")
        
        # 1. Prepare reference text
        ref_text = self._extract_dialogue_from_md(self.transcript_md_path)
        
        # Normalize for WER
        transform = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip()
        ])
        ref_norm = transform(ref_text)
        
        # 2. Load audio
        samples, sr = self._read_audio_mono(self.audio_path, target_sr=None)
        
        # 3. Load and configure ASR
        asr = self._load_asr_pipeline()
        try:
            model_sr = asr.feature_extractor.sampling_rate
        except Exception:
            model_sr = self.target_sr
        
        final_sr = model_sr if model_sr is not None else self.target_sr
        
        if sr != final_sr:
            samples = librosa.resample(samples, orig_sr=sr, target_sr=final_sr).astype("float32")
            sr = final_sr
        
        # 4. Transcribe audio
        def transcribe_array(audio_array: np.ndarray, sampling_rate: int) -> str:
            """Transcribe audio array using ASR pipeline."""
            try:
                out = asr({"array": audio_array, "sampling_rate": sampling_rate})
                return out.get("text", "")
            except Exception as e:
                LOGGER.error(f"ASR transcription failed: {e}")
                return ""
        
        # Chunking for long audio
        hypothesis_parts = []
        if self.chunk_seconds is None:
            hyp_raw = transcribe_array(samples, sr)
        else:
            chunk_len = int(self.chunk_seconds * sr)
            n_chunks = int(np.ceil(len(samples) / chunk_len))
            LOGGER.info(f"Processing {n_chunks} chunks of {self.chunk_seconds}s each")
            
            for i in range(n_chunks):
                start = i * chunk_len
                end = min(len(samples), (i + 1) * chunk_len)
                chunk = samples[start:end]
                part_text = transcribe_array(chunk, sr)
                hypothesis_parts.append(part_text.strip())
            hyp_raw = " ".join([p for p in hypothesis_parts if p])
        
        # Normalize hypothesis
        hyp_norm = transform(hyp_raw)
        
        # 5. Compute WER metrics
        wer_score = jiwer.wer(ref_norm, hyp_norm)
        raw_measures = jiwer.process_words(ref_norm, hyp_norm)
        detailed_measures = {
            "hits": int(raw_measures.hits),
            "substitutions": int(raw_measures.substitutions),
            "deletions": int(raw_measures.deletions),
            "insertions": int(raw_measures.insertions),
        }
        
        # 6. Compute audio metrics
        audio_metrics = {
            "sample_rate": int(sr),
            "duration_s": float(len(samples) / sr),
            "clipping_pct": self._compute_clipping_percentage(samples),
            "rms_dbfs": self._compute_rms_dbfs(samples),
            "spectral_flatness": self._compute_spectral_flatness(samples, sr),
            "voice_activity_ratio": self._compute_voice_activity_ratio(samples, sr),
        }
        
        # 7. Return results as TypedDict
        results: AudioQualityScores = {
            "audio_path": str(self.audio_path),
            "transcript_path": str(self.transcript_md_path),
            "wer": float(wer_score),
            "detailed_measures": detailed_measures,
            "audio_metrics": audio_metrics,
            "hypothesis_raw_preview": hyp_raw[:1000],
            "reference_preview": ref_text[:1000]
        }
        
        LOGGER.info(f"Audio quality evaluation complete. WER: {wer_score:.4f}")
        
        return results

    def save_results(self, results: AudioQualityScores, output_path: str) -> None:
        """Save evaluation results to a JSON file."""
        output_path = Path(output_path)
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        LOGGER.info(f"Saved audio quality report to {output_path}")
    
    def print_summary(self, results: AudioQualityScores) -> None:
        """Print a formatted summary of evaluation results."""
        print("\n" + "="*60)
        print("AUDIO QUALITY EVALUATION RESULTS")
        print("="*60)
        print(f"Audio: {Path(results['audio_path']).name}")
        print(f"Duration: {results['audio_metrics']['duration_s']:.2f}s")
        print(f"WER: {results['wer']:.4f} ({results['wer']*100:.2f}%)")
        print(f"\nDetailed WER measures:")
        for key, value in results['detailed_measures'].items():
            print(f"  {key}: {value}")
        print(f"\nAudio metrics:")
        print(f"  Clipping percentage: {results['audio_metrics']['clipping_pct']:.6f}%")
        print(f"  RMS level: {results['audio_metrics']['rms_dbfs']:.2f} dBFS")
        print(f"  Spectral flatness: {results['audio_metrics']['spectral_flatness']:.4f}")
        print(f"  Voice activity ratio: {results['audio_metrics']['voice_activity_ratio']:.4f}")
        print("="*60)