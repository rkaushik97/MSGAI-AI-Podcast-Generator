import re
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from piper import PiperVoice
import librosa

from .types import Script
from .utils import ModelConstants, LOGGER


class PiperTTSSynthesizer:
    """
    Synthesizes audio from a Script object using Piper TTS.
    Switches voice models based on speaker gender metadata.
    """

    def __init__(self):
        LOGGER.info("Initializing Piper TTS components...")

        # MALE voice model
        male_model_path = hf_hub_download(
            repo_id="rhasspy/piper-voices",
            filename=ModelConstants.PIPER_MALE_MODEL
        )
        male_config_path = hf_hub_download(
            repo_id="rhasspy/piper-voices",
            filename=ModelConstants.PIPER_MALE_CONFIG
        )
        self._male_voice = PiperVoice.load(male_model_path)

        # FEMALE voice model
        female_model_path = hf_hub_download(
            repo_id="rhasspy/piper-voices",
            filename=ModelConstants.PIPER_FEMALE_MODEL
        )
        female_config_path = hf_hub_download(
            repo_id="rhasspy/piper-voices",
            filename=ModelConstants.PIPER_FEMALE_CONFIG
        )
        self._female_voice = PiperVoice.load(female_model_path)

        # Enforce or assume consistent sample rate
        self._sample_rate = self._male_voice.config.sample_rate

        if self._female_voice.config.sample_rate != self._sample_rate:
            LOGGER.warning(
                "Male/Female Piper voices have different sample rates "
                f"({self._sample_rate} vs {self._female_voice.config.sample_rate})."
            )
        
        self.target_sample_rate = ModelConstants.SAMPLING_RATE

        LOGGER.info(
            f"Piper TTS initialized "
            f"(Male SR={self._male_voice.config.sample_rate}, "
            f"Female SR={self._female_voice.config.sample_rate})"
        )

    def _get_voice_model(self, gender: str) -> PiperVoice:
        """Returns the correct Piper voice model based on gender."""
        gender = gender.upper()

        if gender == "MALE":
            return self._male_voice
        elif gender == "FEMALE":
            return self._female_voice
        else:
            LOGGER.warning(f"Unknown gender '{gender}', defaulting to MALE voice.")
            return self._male_voice

    def synthesize(self, script: Script, output_path: str) -> None:
        """Synthesizes the dialogue into a WAV file."""

        segments = re.split(r'(HOST:|GUEST:)', script['dialogue'])

        all_speech_parts = []
        current_voice = None

        LOGGER.info(f"Synthesizing Piper audio for '{script['topic']}'.")

        for segment in segments:
            segment = segment.strip()

            if segment == "HOST:":
                current_voice = self._get_voice_model(
                    script['metadata']['HOST_GENDER']
                )
                continue

            elif segment == "GUEST:":
                current_voice = self._get_voice_model(
                    script['metadata']['GUEST_GENDER']
                )
                continue

            if not segment or current_voice is None:
                continue

            chunks = current_voice.synthesize(segment)
            samples = np.concatenate([c.audio_float_array for c in chunks])

            # Resample to target sample rate
            if current_voice.config.sample_rate != self.target_sample_rate:
                samples = librosa.resample(samples, orig_sr=current_voice.config.sample_rate, target_sr=self.target_sample_rate)

            all_speech_parts.append(samples)

        if not all_speech_parts:
            LOGGER.warning("No speech segments were synthesized.")
            return

        final_audio = np.concatenate(all_speech_parts).astype(np.float32)

        sf.write(
            output_path,
            final_audio,
            self.target_sample_rate,
            subtype="PCM_16"
        )

        LOGGER.info(f"Successfully saved Piper audio to {output_path}")
