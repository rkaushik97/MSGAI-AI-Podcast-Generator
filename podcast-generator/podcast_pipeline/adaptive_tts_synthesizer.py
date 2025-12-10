from podcast_pipeline.kokoro_tts_synthesizer import KokoroTTSSynthesizer
from podcast_pipeline.piper_tts_synthesizer import PiperTTSSynthesizer

class AdaptiveTTSSynthesizer:
    """
    Wrapper for multiple TTS backends (Kokoro, Piper).
    You can switch which backend is active via `set_backend`.
    """

    def __init__(self, backend : str = 'kokoro'):
        self.kokoro = KokoroTTSSynthesizer()
        self.piper = PiperTTSSynthesizer()
        self._current_backend = backend

    def set_backend(self, backend: str):
        backend = backend.lower()
        if backend not in ["kokoro", "piper"]:
            raise ValueError(f"Unsupported TTS backend: {backend}")
        self._current_backend = backend

    @property
    def current_backend(self):
        return self._current_backend

    def synthesize(self, script, output_path: str):
        if self._current_backend == "kokoro":
            self.kokoro.synthesize(script, output_path)
        elif self._current_backend == "piper":
            self.piper.synthesize(script, output_path)
