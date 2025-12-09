import os
import argparse
import json
import time 
from podcast_pipeline.llm_generator import LLMScriptGenerator
from podcast_pipeline.tts_synthesizer import TTSSynthesizer
from podcast_pipeline.utils import setup_environment, LOGGER
from podcast_pipeline.types import ScriptQualityScores, Script 


def process_single_topic(topic: str, prompt_template: str, output_dir: str, generator: LLMScriptGenerator, synthesizer: TTSSynthesizer) -> dict | None:
    """Processes a single topic through the LLM and TTS pipeline."""
    try:
        # LLM Generation Stage
        script: Script
        quality_scores: ScriptQualityScores
        script, quality_scores = generator.generate(topic, prompt_template)
        
        # Check if the automated quality check failed
        if not isinstance(quality_scores, dict) or quality_scores.get('relevance_score') is None:
            LOGGER.error(f"Skipping TTS synthesis for '{topic}' because automated LLM Judge evaluation failed.")
            return None

        # Log the critical offline analysis metrics
        LOGGER.info(f"Generated Script Metadata: Host={script['metadata']['HOST_GENDER']}, Guest={script['metadata']['GUEST_GENDER']}")
        LOGGER.info(f"Script Quality: Relevance={quality_scores['relevance_score']}, Coherence={quality_scores['coherence_score']}")
        
        # Create output filename prefix and directory
        os.makedirs(output_dir, exist_ok=True)
        clean_topic = script['topic'].replace(":", "").replace("?", "").replace("/", "").strip()
        filename_prefix = clean_topic[:30].replace(' ', '_').lower()
        
        # Save the Podcast Script
        script_filename_local = f"{filename_prefix}_script.md"
        script_filename_full = os.path.join(output_dir, script_filename_local)
        with open(script_filename_full, 'w', encoding='utf-8') as f:
            f.write(f"# Podcast Script: {script['topic']}\n\n")
            f.write(f"Metadata: Host={script['metadata']['HOST_GENDER']}, Guest={script['metadata']['GUEST_GENDER']}\n\n")
            f.write("--- Dialogue ---\n\n")
            f.write(script['dialogue'])

        LOGGER.info(f"Podcast script saved to: {script_filename_full}")
        
        # TTS Synthesis Stage
        audio_filename_local = f"{filename_prefix}_H{script['metadata']['HOST_GENDER'][0]}_G{script['metadata']['GUEST_GENDER'][0]}.wav"
        output_filename_full = os.path.join(output_dir, audio_filename_local)
        
        synthesizer.synthesize(script, output_filename_full)
        
        LOGGER.info(f"Audio file saved to: {output_filename_full}")

        # Return results for batch summary
        return {
            "topic": topic,
            "relevance_score": quality_scores['relevance_score'],
            "coherence_score": quality_scores['coherence_score'],
            "script_file": script_filename_local,
            "audio_file": audio_filename_local,
            "host_gender": script['metadata']['HOST_GENDER'],
            "guest_gender": script['metadata']['GUEST_GENDER'],
        }

    except Exception as e:
        LOGGER.error(f"Pipeline failed for topic '{topic}' during execution: {e}")
        return None


def run_podcast_pipeline(topic_input: str, prompt_template: str, output_dir: str = "output"):
    """
    Executes the end-to-end LLM-to-TTS pipeline for a batch of topics.
    
    Args:
        topic_input: Either a single topic string or a path to a file containing topics.
        output_dir: Directory to save the final files.
    """
    
    setup_environment(hf_token=hf_token) # Load token and set up logging

    topics = []
    # Check if input is a file path
    if os.path.exists(topic_input):
        LOGGER.info(f"Reading topics from file: {topic_input}")
        try:
            with open(topic_input, 'r', encoding='utf-8') as f:
                topics = [line.strip() for line in f if line.strip()]
        except IOError as e:
            LOGGER.error(f"Could not read topic file: {e}")
            return
    else:
        # Treat as a single topic
        LOGGER.info(f"Processing single topic: {topic_input}")
        topics.append(topic_input)

    if not topics:
        LOGGER.error("No topics to process. Exiting.")
        return
        
    LOGGER.info(f"Starting batch process for {len(topics)} topic(s).")
    
    all_results = []
    
    # Initialize generators outside the loop for efficiency
    generator = LLMScriptGenerator()
    synthesizer = TTSSynthesizer() 
    
    for i, topic in enumerate(topics):
        LOGGER.info(f"--- Processing Topic {i+1}/{len(topics)}: {topic} ---")

        start_time = time.time() # Roughly how long is it taking to run the pipeline for 1 podcast request? 

        result = process_single_topic(topic, prompt_template, output_dir, generator, synthesizer)

        elapsed_time = time.time() - start_time

        if result:
            # Add processing time to results before appending
            result['processing_time_sec'] = round(elapsed_time, 2)
            all_results.append(result)
            LOGGER.info(f"--- Topic {i+1} finished successfully in {elapsed_time:.2f} seconds. ---")
        else:
            LOGGER.info(f"--- Topic {i+1} failed after {elapsed_time:.2f} seconds. ---")

    # Save all results to a summary file
    summary_path = os.path.join(output_dir, "batch_results.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    
    LOGGER.info(f"Batch processing complete. Results summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a podcast script and audio from a topic using LLM and TTS models.")
    parser.add_argument(
        "--topic", 
        type=str,  
        default="input/topics_batch3.txt",
        help="The path to a txt file containing a list of topics (one per line)."
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
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging face token to download LLAMA model."
    )
    
    args = parser.parse_args()
    
    run_podcast_pipeline(args.topic, args.prompt_template, args.output_dir, args.hf_token)
