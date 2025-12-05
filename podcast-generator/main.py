import os
import argparse
from podcast_pipeline.llm_generator import LLMScriptGenerator
from podcast_pipeline.tts_synthesizer import TTSSynthesizer
from podcast_pipeline.utils import setup_environment, LOGGER


def run_podcast_pipeline(topic: str, prompt_template: str, output_dir: str = "output"):
    """
    Executes the end-to-end LLM-to-TTS pipeline.
    
    Args:
        topic: The podcast topic to generate a script for.
        output_dir: Directory to save the final WAV file.
    """
    
    setup_environment() # Load token and set up logging

    try:
        # LLM Generation Stage
        generator = LLMScriptGenerator()
        script = generator.generate(topic, prompt_template)
        
        LOGGER.info(f"Generated Script Metadata: Host={script.metadata.HOST_GENDER}, Guest={script.metadata.GUEST_GENDER}")
        
        # TTS Synthesis Stage
        synthesizer = TTSSynthesizer()
        
        # Create output filename
        os.makedirs(output_dir, exist_ok=True)
        clean_topic = script.topic.replace(":", "").replace("?", "").replace("/", "").strip()
        filename_prefix = clean_topic[:30].replace(' ', '_').lower()
        output_filename = f"{output_dir}/{filename_prefix}_H{script.metadata.HOST_GENDER[0]}_G{script.metadata.GUEST_GENDER[0]}.wav"
        
        synthesizer.synthesize(script, output_filename)
        
    except Exception as e:
        LOGGER.error(f"Pipeline failed during execution: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a podcast script and audio from a topic using LLM and TTS models.")
    parser.add_argument(
        "--topic", 
        type=str,  
        default="Theoretical physics of time travel",
        help="The topic for the podcast, e.g., 'The theoretical physics of time travel'."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The directory to save the final WAV file."
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="podcast_script_v1",
        help="The template to use for the LLM prompt. It is defined in the prompt_manager.py file."
    )
    
    args = parser.parse_args()
    
    run_podcast_pipeline(args.topic, args.prompt_template, args.output_dir)
