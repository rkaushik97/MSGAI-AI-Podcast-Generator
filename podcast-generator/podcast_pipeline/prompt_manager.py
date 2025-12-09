"""
Manages and retrieves prompt templates for the LLM.
"""

# LLM Judge System Instruction
# This instruction is critical for forcing the Judge LLM to output clean JSON scores.
LLM_JUDGE_SYSTEM_INSTRUCTION = (
    "You are an impartial, expert evaluator of podcast scripts. Your sole task is to analyze a generated script "
    "against the user's original request. You MUST output a JSON object containing only the keys 'relevance_score' "
    "and 'coherence_score', both as integers from 1 (Very Poor) to 5 (Excellent). DO NOT provide any external "
    "commentary, reasoning, or text outside of the required JSON object."
    "\n\nScoring Criteria:"
    "\n- Relevance (1-5): How closely and accurately does the script address the user's original topic request?"
    "\n- Coherence (1-5): Does the script flow logically? Are the transitions between speakers smooth? Is the structure sound?"
)

PROMPT_TEMPLATES = {
    "podcast_script_v1": """
Your entire output MUST start with the exact token: ---METADATA---

---METADATA---
HOST_GENDER: [MALE or FEMALE]
GUEST_GENDER: [MALE or FEMALE]
GUEST_NAME: Create a relevant fictional or historical expert name based on the topic
---DIALOGUE---

Your entire output must contain a two-person podcast conversation between HOST and GUEST, formatted strictly as:
HOST: …
GUEST: …
(continue dialogue)

No other text, instructions, or blank lines are permitted AFTER the '---DIALOGUE---' line.

Content Requirements
Podcast: Adaptive AI Podcast
Topic: {topic}
Length: 150–200 words

Dialogue Structure (follow exactly):
HOST: hook about topic
HOST: podcast + host intro
HOST: guest intro (MUST use the GUEST_NAME you generated)
GUEST: brief greeting
HOST: simple first question
GUEST: 3–5 sentence answer
HOST: follow-up question
GUEST: short answer
HOST: closing line: “Let’s get started.”

---BEGIN DIALOGUE (must immediately follow the '---DIALOGUE---' line)---
HOST:""",

    # Template for the LLM Judge API call
    "llm_judge_v1": {
        "system_instruction": LLM_JUDGE_SYSTEM_INSTRUCTION,
        "user_query": """
EVALUATION TASK:
Please evaluate the following GENERATED SCRIPT based on the ORIGINAL TOPIC REQUEST.

--- ORIGINAL TOPIC REQUEST ---
{original_topic}

--- GENERATED SCRIPT ---
{generated_script}
"""
    }
}


class PromptManager:
    """
    A class to retrieve and format specific prompt templates.
    """
    def __init__(self, templates: dict = PROMPT_TEMPLATES):
        self._templates = templates

    def get_template(self, template_key: str) -> str:
        """Retrieves a prompt template by its key."""
        if template_key not in self._templates:
            raise ValueError(f"Prompt template key '{template_key}' not found in manager.")
        return self._templates[template_key]

    def format_prompt(self, template_key: str, **kwargs) -> str:
        """
        Retrieves a template and formats it using the provided keyword arguments.
        
        Args:
            template_key: The key of the template to use (e.g., 'podcast_script_v1').
            **kwargs: Arguments to substitute into the template (e.g., 'topic').
            
        Returns:
            The fully formatted prompt string.
        """
        template = self.get_template(template_key)
        if isinstance(template, dict):
            # This is for structured templates like the judge, not simple formatting
            raise TypeError(f"Template '{template_key}' is structured and cannot be formatted with this method.")
        
        return template.format(**kwargs)

    def format_judge_prompt(self, template_key: str, original_topic: str, generated_script: str) -> tuple[str, str]:
        """
        Retrieves and formats a structured judge template, returning system instruction and user query.
        """
        template_data = self.get_template(template_key)
        
        if not isinstance(template_data, dict) or 'system_instruction' not in template_data:
            raise TypeError(f"Template '{template_key}' is not a valid structured judge template.")

        user_query = template_data['user_query'].format(
            original_topic=original_topic,
            generated_script=generated_script
        )
        return template_data['system_instruction'], user_query
