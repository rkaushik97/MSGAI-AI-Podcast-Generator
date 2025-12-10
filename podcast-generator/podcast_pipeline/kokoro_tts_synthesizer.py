import re
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro
from .types import Script
from .utils import LOGGER, ModelConstants


class KokoroTTSSynthesizer:
    """
    Synthesizes audio from a Script object using Kokoro ONNX.
    Switches voices via voice IDs (male/female) using metadata.
    """

    def __init__(self):
        LOGGER.info("Initializing Kokoro TTS components...")

        # Load the Kokoro model and voices
        self._tts = Kokoro(
            ModelConstants.KOKORO_MODEL_PATH,
            ModelConstants.KOKORO_VOICES_PATH
        )

        self._sample_rate = 24000
        LOGGER.info(f"Kokoro TTS initialized (sample_rate={self._sample_rate})")

    def _get_voice(self, gender: str) -> str:
        """Returns Kokoro voice ID based on gender."""
        gender = gender.upper()
        if gender == "MALE":
            return ModelConstants.KOKORO_MALE_VOICE
        elif gender == "FEMALE":
            return ModelConstants.KOKORO_FEMALE_VOICE
        else:
            LOGGER.warning(f"Unknown gender '{gender}', defaulting to MALE voice.")
            return ModelConstants.KOKORO_MALE_VOICE

    @staticmethod
    def _chunk_text(text: str, max_words: int = 200):
        """Split text into smaller chunks of max_words words."""
        words = text.split()
        for i in range(0, len(words), max_words):
            yield " ".join(words[i:i + max_words])

    def synthesize(self, script: Script, output_path: str) -> None:
        """Synthesizes the dialogue into a WAV file."""
        segments = re.split(r'(HOST:|GUEST:)', script['dialogue'])
        all_speech_parts = []
        current_voice = None

        LOGGER.info(f"Synthesizing Kokoro audio for '{script['topic']}'.")

        for segment in segments:
            segment = segment.strip()
            if segment == "HOST:":
                current_voice = self._get_voice(script['metadata']['HOST_GENDER'])
                continue
            elif segment == "GUEST:":
                current_voice = self._get_voice(script['metadata']['GUEST_GENDER'])
                continue

            if not segment or current_voice is None:
                continue

            # Clean up segment text
            segment = re.sub(r'\s+', ' ', segment).strip()
            LOGGER.info(f"Synthesizing segment ({current_voice}): {segment[:60]}...")

            # Split into smaller chunks for Kokoro
            for chunk in self._chunk_text(segment, max_words=200):
                try:
                    result = self._tts.create(chunk, voice=current_voice, speed=1.0, lang="en-us")

                    # Handle different return types
                    if isinstance(result, tuple) and len(result) == 2:
                        samples, sample_rate = result
                        if sample_rate != self._sample_rate:
                            LOGGER.warning("Hardcoded sample rate for kokoro and what it generates " \
                            "do not coincide, ", self._sample_rate, sample_rate)
                    else:
                        samples = result

                    all_speech_parts.append(samples)

                except Exception as e:
                    LOGGER.warning(f"Kokoro failed on chunk: {chunk[:50]}... | Error: {e}")
                    continue

        if not all_speech_parts:
            LOGGER.warning("No speech segments were synthesized.")
            return

        # Concatenate all segments
        final_audio = np.concatenate(all_speech_parts).astype(np.float32)

        # Write to WAV
        sf.write(output_path, final_audio, self._sample_rate, subtype="PCM_16")
        LOGGER.info(f"Successfully saved Kokoro audio to {output_path}")
