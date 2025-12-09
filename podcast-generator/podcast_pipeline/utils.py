import os
import torch
from huggingface_hub import login
from dotenv import load_dotenv
from datasets import load_dataset
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

# Environment and Token setup 

def setup_environment(hf_token_env_var: str = "HUGGINGFACE_API_KEY") -> None:
    """Loads environment and logs into Hugging Face"""
    load_dotenv()
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
    VOCODER_MODEL_ID = "microsoft/speecht5_hifigan"
    EMBEDDINGS_DATASET = "Matthijs/cmu-arctic-xvectors"
    SILENCE_DURATION_SEC = 0.05
    SAMPLING_RATE = 16000

# Speaker Embeddings Loader 

def load_speaker_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    """Loads and returns the Male and Female SpeechT5 embeddings."""
    LOGGER.info("Loading speaker embeddings...")
    try:
        embeddings_dataset = load_dataset(ModelConstants.EMBEDDINGS_DATASET, split="validation")
        # Index 0 is 'bdl' (Male), Index 7306 is 'slt' (Female)
        male_embedding = torch.tensor(embeddings_dataset[0]['xvector']).unsqueeze(0)
        female_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        return male_embedding, female_embedding
    except Exception as e:
        LOGGER.error(f"Failed to load speaker embeddings: {str(e)}")
        raise