import os
from huggingface_hub import login
from dotenv import load_dotenv
# from datasets import load_dataset
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

# Environment and Token setup 

def setup_environment(hf_token_env_var: str = "HUGGINGFACE_API_KEY", hf_token: str | None = None) -> None:
    """Loads environment and logs into Hugging Face"""
    load_dotenv()
    if hf_token is None:
        hf_token = os.getenv(hf_token_env_var)
    if hf_token:
        login(hf_token, add_to_git_credential=False)
    else:
        LOGGER.warning("Hugging Face token not found. Please set the HUGGINGFACE_API_KEY environment variable.")

# Model Constants

class ModelConstants:
    """ Centralizes model IDs and paths."""
    LLM_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    TTS_MODEL_ID = "microsoft/speecht5_tts"
    SILENCE_DURATION_SEC = 0.05
    SAMPLING_RATE = 24000
    # piper settings
    PIPER_MALE_MODEL = "en/en_US/bryce/medium/en_US-bryce-medium.onnx"
    PIPER_MALE_CONFIG = "en/en_US/bryce/medium/en_US-bryce-medium.onnx.json"
    PIPER_FEMALE_MODEL = "en/en_US/amy/medium/en_US-amy-medium.onnx"
    PIPER_FEMALE_CONFIG = "en/en_US/amy/medium/en_US-amy-medium.onnx.json"
    # kokoro settings
    KOKORO_MODEL_PATH = "../kokoro/kokoro-v1.0.fp16.onnx" # "../kokoro/kokoro-v1.0.onnx"
    KOKORO_VOICES_PATH = "../kokoro/voices-v1.0.bin"
    KOKORO_MALE_VOICE = "am_michael"
    KOKORO_FEMALE_VOICE = "af_sarah"
    # audio quality settings
    ASR_MODEL_ID = "openai/whisper-small"
    ASR_DEVICE = 0  # -1 for CPU, 0 for GPU 0
    

# Speaker Embeddings Loader 

# def load_speaker_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
#     """Loads and returns the Male and Female SpeechT5 embeddings."""
#     LOGGER.info("Loading speaker embeddings...")
#     try:
#         embeddings_dataset = load_dataset(ModelConstants.EMBEDDINGS_DATASET, split="validation")
#         # Index 0 is 'bdl' (Male), Index 7306 is 'slt' (Female)
#         male_embedding = torch.tensor(embeddings_dataset[0]['xvector']).unsqueeze(0)
#         female_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
#         return male_embedding, female_embedding
#     except Exception as e:
#         LOGGER.error(f"Failed to load speaker embeddings: {str(e)}")
#         raise
