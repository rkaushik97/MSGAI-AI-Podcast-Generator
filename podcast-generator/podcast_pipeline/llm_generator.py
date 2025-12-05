import re
from transformers import pipeline, Pipeline
import torch

from .types import Metadata, Script
from .utils import ModelConstants, LOGGER
from .prompt_manager import PromptManager


class LLMScriptGenerator:
    """
    Generates a structured podcast script using a large language model (LLM).
    Uses the Singleton pattern implicitly via class-level initialization
    if used correctly, but implemented as a simple Factory pattern for clarity.
    """
    def __init__(self, model_id: str = ModelConstants.LLM_MODEL_ID):
        LOGGER.info(f"Initializing LLM Pipeline: {model_id}")
        self._pipeline: Pipeline = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self._prompt_manager = PromptManager()

    def _create_prompt(self, topic: str, template_key: str = "podcast_script_v1") -> list[dict]:
        """
        Constructs the structured prompt using a template key.
        The template_key defaults to the standard podcast script.
        """
        prompt_content = self._prompt_manager.format_prompt(
            template_key, 
            topic=topic
        )
        
        return [{"role": "user", "content": prompt_content}]

    def _parse_output(self, generated_text: str) -> tuple[Metadata, str]:
        """Parses the LLM's structured output into Metadata and Dialogue."""
        metadata = {"HOST_GENDER": "MALE", "GUEST_GENDER": "FEMALE", "GUEST_NAME": "Unknown Guest"}
        dialogue = generated_text

        try:
            _, metadata_block_and_dialogue = generated_text.split("---METADATA---", 1)
            metadata_block, dialogue_block = metadata_block_and_dialogue.split("---DIALOGUE---", 1)
            dialogue = dialogue_block.strip()

            pattern = re.compile(r"(\w+)_GENDER:\s*(\w+)|GUEST_NAME:\s*(.*)", re.IGNORECASE)
            for line in metadata_block.split('\n'):
                match = pattern.search(line)
                if match:
                    if match.group(1) and match.group(2): 
                        key = f"{match.group(1).upper()}_GENDER"
                        metadata[key] = match.group(2).strip().upper()
                    elif match.group(3):
                        metadata["GUEST_NAME"] = match.group(3).strip()
        except ValueError:
            LOGGER.warning("LLM output delimiters missing. Using raw output.")
        
        return Metadata(**metadata), dialogue

    def generate(self, topic: str, template_key: str = "podcast_script_v1") -> Script:
        """The primary public method to generate a full script.

        Args:
            topic: The topic for the podcast.
            template_key: The key of the template to use (e.g., 'podcast_script_v1').
        """
        messages = self._create_prompt(topic, template_key)
        
        outputs = self._pipeline(
            messages,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.8,
            pad_token_id=self._pipeline.tokenizer.eos_token_id,
        )
        
        generated_text = outputs[0]["generated_text"][-1]['content']
        metadata, dialogue = self._parse_output(generated_text)
        
        return Script(topic=topic, dialogue=dialogue, metadata=metadata)
