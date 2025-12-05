"""
Manages and retrieves prompt templates for the LLM.
"""

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
HOST:"""
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
        return template.format(**kwargs)
