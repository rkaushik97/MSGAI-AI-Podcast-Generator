import re
import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

from .types import Script
from .utils import ModelConstants, LOGGER, load_speaker_embeddings

class TTSSynthesizer:
    """
    Synthesizes audio from a Script object using Microsoft SpeechT5,
    switching voices based on the script's metadata.
    """
    def __init__(self):
        LOGGER.info("Initializing SpeechT5 TTS components...")
        self._processor = SpeechT5Processor.from_pretrained(ModelConstants.TTS_MODEL_ID)
        self._model = SpeechT5ForTextToSpeech.from_pretrained(ModelConstants.TTS_MODEL_ID)
        self._vocoder = SpeechT5HifiGan.from_pretrained(ModelConstants.VOCODER_MODEL_ID)
        self._male_embedding, self._female_embedding = load_speaker_embeddings()
        self._silence = torch.zeros(
            int(ModelConstants.SAMPLING_RATE * ModelConstants.SILENCE_DURATION_SEC)
        )

    def _get_speaker_embedding(self, gender: str) -> torch.Tensor:
        """Returns the correct embedding based on gender string."""
        if gender.upper() == "MALE":
            return self._male_embedding
        elif gender.upper() == "FEMALE":
            return self._female_embedding
        else:
            LOGGER.warning(f"Unknown gender '{gender}'. Defaulting to Male embedding.")
            return self._male_embedding

    def synthesize(self, script: Script, output_path: str) -> None:
        """Synthesizes the dialogue into a WAV file."""
        host_embed = self._get_speaker_embedding(script['metadata']['HOST_GENDER'])
        guest_embed = self._get_speaker_embedding(script['metadata']['GUEST_GENDER'])
        segments = re.split(r'(HOST:|GUEST:)', script['dialogue'])

        all_speech_parts = []
        current_speaker_embedding = None

        LOGGER.info(f"Synthesizing audio segments for '{script['topic']}'.")
        for segment in segments:
            segment = segment.strip()
            
            if segment == "HOST:":
                current_speaker_embedding = host_embed
                continue
            elif segment == "GUEST:":
                current_speaker_embedding = guest_embed
                continue
            
            if not segment or current_speaker_embedding is None:
                continue

            # Generate speech
            inputs = self._processor(text=segment, return_tensors="pt")
            with torch.no_grad():
                speech = self._model.generate_speech(
                    inputs["input_ids"], 
                    current_speaker_embedding, 
                    vocoder=self._vocoder
                )
            
            all_speech_parts.append(speech)
            all_speech_parts.append(self._silence)

        if all_speech_parts:
            final_audio = torch.cat(all_speech_parts, dim=0)
            sf.write(
                output_path, 
                final_audio.numpy(), 
                samplerate=ModelConstants.SAMPLING_RATE
            )
            LOGGER.info(f"Successfully saved audio to {output_path}")
        else:
            LOGGER.warning("No speech parts generated.")
