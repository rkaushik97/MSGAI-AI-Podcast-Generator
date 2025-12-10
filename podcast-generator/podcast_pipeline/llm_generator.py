import re
import json
import os
from transformers import pipeline, Pipeline
import torch
from typing import Tuple, Optional, Dict 

from .types import Metadata, Script, ScriptQualityScores
from .utils import ModelConstants, LOGGER
from .prompt_manager import PromptManager


class LLMScriptGenerator:
    """
    Generates a structured podcast script using a large language model (LLM).
    Uses the Singleton pattern implicitly via class-level initialization
    if used correctly, but implemented as a simple Factory pattern for clarity.
    """
    def __init__(self, model_id: str = ModelConstants.LLM_MODEL_ID, few_shot_examples_path: str = "./input/few_shot_examples_responses.json"):
        LOGGER.info(f"Initializing LLM Pipeline: {model_id}")
        # Initialize the pipeline once for both script generation and judge calls.
        self._pipeline: Pipeline = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self._prompt_manager = PromptManager()
        if not os.path.exists(few_shot_examples_path):
            LOGGER.info(f"Few-shot examples .json file path invalid")
            self.few_shot_examples = []
        else:
            with open(few_shot_examples_path) as infile:
                self.few_shot_examples = json.load(infile)
            LOGGER.info(f"Few-shot examples json file loaded, it contains {len(self.few_shot_examples)} few-shot examples")

    def _create_prompt(self, topic: str, template_key: str = "podcast_script_v1", few_shot_examples_nr: int = 0) -> list[dict]:
        """
        Constructs the structured prompt using a template key.
        The template_key defaults to the standard podcast script.
        Adds up to `few_shot_examples_nr` demo examples before the main prompt.
        """
        messages = []

        # Add few-shot examples (in conversation format)
        if few_shot_examples_nr > 0 and self.few_shot_examples:
            for ex in self.few_shot_examples[:few_shot_examples_nr]:
                messages.append({"role": "user", "content": ex["prompt"][0]["content"]})
                messages.append({"role": "assistant", "content": ex["script"]})

        # Add the real prompt
        prompt_content = self._prompt_manager.format_prompt(template_key, topic=topic)
        messages.append({"role": "user", "content": prompt_content})

        return messages

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

    def _execute_judge_local(self, original_topic: str, generated_script: str) -> Dict[str, Optional[int]]:
        """
        Executes the LLM Judge inference locally using the existing pipeline.
        The pipeline runs a structured generation call to output JSON scores.
        """
        if not generated_script:
            LOGGER.error("Generated script is empty. Cannot evaluate locally.")
            return {"relevance_score": None, "coherence_score": None}
            
        try:
            # Get structured prompt (system instruction + user query)
            system_prompt, user_query = self._prompt_manager.format_judge_prompt(
                "llm_judge_v1", 
                original_topic, 
                generated_script
            )
            
            # prompt formatted as a conversation
            judge_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]

            # Run inference on the local pipeline
            # Use low temperature and limited tokens for deterministic, concise JSON output.
            judge_output = self._pipeline(
                judge_messages,
                max_new_tokens=128, 
                do_sample=False, 
                temperature=0.1,
                pad_token_id=self._pipeline.tokenizer.eos_token_id,
            )
            
            # Extract the generated content (assuming it's the last message content)
            judge_text = judge_output[0]["generated_text"][-1]['content']
            
            # Parse JSON - search for the JSON object within the output
            json_match = re.search(r'\{.*\}', judge_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0).strip()
                scores = json.loads(json_text)
                return {
                    "relevance_score": int(scores.get("relevance_score")),
                    "coherence_score": int(scores.get("coherence_score"))
                }
            
            LOGGER.error(f"Local Judge failed to output valid JSON. Raw output: {judge_text[:200]}...")
            return {"relevance_score": None, "coherence_score": None}

        except Exception as e:
            LOGGER.error(f"Error during local Judge execution: {e}")
            return {"relevance_score": None, "coherence_score": None}


    def generate(self, topic: str, template_key: str = "podcast_script_v1") -> Script:
        """The primary public method to generate a full script.

        Args:
            topic: The topic for the podcast.
            template_key: The key of the template to use (e.g., 'podcast_script_v1').
        """
        messages = self._create_prompt(topic, template_key)
        
        outputs = self._pipeline(
            messages,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.8,
            pad_token_id=self._pipeline.tokenizer.eos_token_id,
        )
        
        generated_text = outputs[0]["generated_text"][-1]['content']
        metadata, dialogue = self._parse_output(generated_text)
        
        script = Script(topic=topic, dialogue=dialogue, metadata=metadata)
        # LLM Judge Integration (NO JUDGE AT THE MOMENT)
        # scores = self._execute_judge_local(topic, dialogue)
        scores = {'relevance_score': 5, 'coherence_score': 5}
        LOGGER.info(f"Automated Script Quality Scores: Relevance={scores.get('relevance_score')}, Coherence={scores.get('coherence_score')}")

        return script, scores
