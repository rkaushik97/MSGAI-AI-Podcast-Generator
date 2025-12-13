import os
import argparse
import json
import time
import statistics
from datetime import datetime
from typing import List, Dict, Any
from tabulate import tabulate
from podcast_pipeline.llm_generator import LLMScriptGenerator
from podcast_pipeline.adaptive_tts_synthesizer import AdaptiveTTSSynthesizer
from podcast_pipeline.audio_quality_analyzer import AudioQualityAnalyzer
from podcast_pipeline.utils import setup_environment, LOGGER
from podcast_pipeline.types import ScriptQualityScores, Script, AudioQualityScores


def calculate_latency_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate latency statistics from processing results."""
    if not results:
        return {}
    
    processing_times = [r.get('processing_time_sec', 0) for r in results]
    valid_times = [t for t in processing_times if t > 0]
    
    if not valid_times:
        return {}
    
    return {
        'average_latency': statistics.mean(valid_times),
        'median_latency': statistics.median(valid_times),
        'min_latency': min(valid_times),
        'max_latency': max(valid_times),
        'std_deviation': statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
        'total_requests': len(results),
        'successful_requests': len(valid_times),
        'total_processing_time': sum(valid_times),
        'requests_per_minute': len(valid_times) / (sum(valid_times) / 60) if sum(valid_times) > 0 else 0
    }


def generate_performance_table(results: List[Dict[str, Any]], latency_stats: Dict[str, Any]) -> str:
    """Generate a formatted performance table."""
    if not results:
        return "No results to display."
    
    # Prepare table data
    table_data = []
    for i, result in enumerate(results, 1):
        topic = result.get('topic', 'Unknown')
        processing_time = result.get('processing_time_sec', 0)
        backend = result.get('tts_backend', 'N/A')
        
        # Truncate long topics for display
        display_topic = (topic[:40] + '...') if len(topic) > 40 else topic
        
        table_data.append([
            i,
            display_topic,
            f"{processing_time:.2f}s",
            backend,
            result.get('host_gender', 'N/A'),
            result.get('guest_gender', 'N/A')
        ])
    
    # Create table
    headers = ["#", "Topic", "Latency", "TTS Backend", "Host Gender", "Guest Gender"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    
    # Add summary section
    summary = "\n" + "="*80 + "\n"
    summary += "PERFORMANCE SUMMARY\n"
    summary += "="*80 + "\n"
    
    if latency_stats:
        summary += f"Average Latency:     {latency_stats.get('average_latency', 0):.2f} seconds\n"
        summary += f"Median Latency:      {latency_stats.get('median_latency', 0):.2f} seconds\n"
        summary += f"Min Latency:         {latency_stats.get('min_latency', 0):.2f} seconds\n"
        summary += f"Max Latency:         {latency_stats.get('max_latency', 0):.2f} seconds\n"
        summary += f"Standard Deviation:  {latency_stats.get('std_deviation', 0):.2f} seconds\n"
        summary += f"Total Processing:    {latency_stats.get('total_processing_time', 0):.2f} seconds\n"
        summary += f"Successful Requests: {latency_stats.get('successful_requests', 0)}/{latency_stats.get('total_requests', 0)}\n"
        summary += f"Throughput:          {latency_stats.get('requests_per_minute', 0):.2f} requests/minute\n"
    
    summary += "="*80
    
    return f"{table}\n{summary}"


def save_performance_report(
    results: List[Dict[str, Any]], 
    latency_stats: Dict[str, Any], 
    output_dir: str,
    config: Dict[str, Any]
) -> str:
    """Save detailed performance report to JSON file."""
    report_path = os.path.join(output_dir, f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "configuration": config,
        "latency_statistics": latency_stats,
        "detailed_results": results,
        "summary": {
            "total_topics_processed": len(results),
            "successful_topics": len([r for r in results if r.get('processing_time_sec', 0) > 0]),
            "failed_topics": len([r for r in results if r.get('processing_time_sec', 0) <= 0]),
            "tts_backend_distribution": {
                "kokoro": len([r for r in results if r.get('tts_backend') == 'kokoro']),
                "piper": len([r for r in results if r.get('tts_backend') == 'piper'])
            }
        }
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, default=str)
    
    LOGGER.info(f"Performance report saved to: {report_path}")
    return report_path


def process_single_topic(
    topic: str,
    prompt_template: str, 
    output_dir: str, 
    generator: LLMScriptGenerator, 
    synthesizer: AdaptiveTTSSynthesizer,
    max_token_len: int,
    few_shot_nr: int,
    evaluate_audio_quality: bool = False,
    track_timing: bool = False
    ) -> dict | None:
    """Processes a single topic through the LLM and TTS pipeline."""
    timing_data = {}
    stage_start_time = time.time()
    
    try:
        # LLM Generation Stage
        script: Script
        quality_scores: ScriptQualityScores
        script, quality_scores = generator.generate(
            topic, 
            prompt_template, 
            few_shot_examples_nr=few_shot_nr, 
            max_new_tokens=max_token_len
        )

        if track_timing:
            timing_data['llm_generation_time'] = time.time() - stage_start_time
            stage_start_time = time.time()

        quality_scores = {'relevance_score': None, 'coherence_score': None}
        
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
        audio_filename_local = f"{filename_prefix}_H{script['metadata']['HOST_GENDER'][0]}_G{script['metadata']['GUEST_GENDER'][0]}_tts_{synthesizer.current_backend}.wav"
        output_filename_full = os.path.join(output_dir, audio_filename_local)
        
        synthesizer.synthesize(script, output_filename_full)
        
        if track_timing:
            timing_data['tts_synthesis_time'] = time.time() - stage_start_time
            stage_start_time = time.time()

        LOGGER.info(f"Audio file saved to: {output_filename_full}")

        # Audio Quality Evaluation Stage
        audio_quality_scores = None
        if evaluate_audio_quality:
            LOGGER.info("Starting audio quality evaluation...")
            try:
                analyzer = AudioQualityAnalyzer()
                audio_quality_results = analyzer.evaluate(
                    audio_path=output_filename_full,
                    transcript_md_path=script_filename_full
                )
                audio_quality_scores = {
                    "wer": audio_quality_results["wer"],
                    "detailed_measures": audio_quality_results["detailed_measures"],
                    "audio_metrics": audio_quality_results["audio_metrics"]
                }
                
                # Save audio quality report
                audio_quality_report = f"{filename_prefix}_audio_quality.json"
                analyzer.save_results(audio_quality_results, os.path.join(output_dir, audio_quality_report))
                
                # Print summary
                analyzer.print_summary(audio_quality_results)
                
                LOGGER.info(f"Audio quality evaluation complete. WER: {audio_quality_results['wer']:.4f}")
                
                if track_timing:
                    timing_data['audio_quality_evaluation_time'] = time.time() - stage_start_time
                
            except Exception as e:
                LOGGER.error(f"Audio quality evaluation failed: {e}")
                audio_quality_scores = {
                    "wer": None,
                    "detailed_measures": None,
                    "audio_metrics": None,
                    "error": str(e)
                }

        # result for batch summary
        result ={
            "topic": topic,
            "relevance_score": quality_scores['relevance_score'],
            "coherence_score": quality_scores['coherence_score'],
            "script_file": script_filename_local,
            "audio_file": audio_filename_local,
            "host_gender": script['metadata']['HOST_GENDER'],
            "guest_gender": script['metadata']['GUEST_GENDER'],
            "tts_backend": synthesizer.current_backend
        }

        # Add timing data if tracking
        if track_timing:
            result['timing_data'] = timing_data
            # Calculate total processing time from individual stages
            total_time = sum(timing_data.values())
            result['processing_time_sec'] = round(total_time, 2)
            
            # Log detailed timing
            LOGGER.debug(f"Timing breakdown for '{topic[:30]}...':")
            for stage, duration in timing_data.items():
                LOGGER.debug(f"  {stage}: {duration:.2f}s")

        # Add audio quality scores if available
        if audio_quality_scores:
            result["audio_quality"] = audio_quality_scores
        
        return result

    except Exception as e:
        LOGGER.error(f"Pipeline failed for topic '{topic}' during execution: {e}")
        if track_timing:
            LOGGER.error(f"Processing failed after {time.time() - stage_start_time:.2f} seconds")
        return None


def run_podcast_pipeline(
    topic_input: str,
    prompt_template: str, 
    output_dir: str = "output", 
    hf_token: str | None = None, 
    tts_backend: str = "kokoro",
    max_token_len: int = 256,
    few_shot_nr: int = 3,
    evaluate_script: bool = False,
    evaluate_audio: bool = False,
    analyze_performance: bool = False,
    ):
    """
    Executes the end-to-end LLM-to-TTS pipeline for a batch of topics.
    
    Args:
        topic_input: Either a single topic string or a path to a file containing topics.
        output_dir: Directory to save the final files.
        evaluate_audio: Whether to run audio quality evaluation.
        analyze_performance: Whether to calculate and display performance statistics.
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
    if evaluate_audio:
        LOGGER.info("Audio quality evaluation is ENABLED (this will add processing time)")
    else:
        LOGGER.info("Audio quality evaluation is DISABLED")
    
    if analyze_performance:
        LOGGER.info("Performance analysis is ENABLED")
        LOGGER.info("Detailed timing data will be collected for each stage")
    
    all_results = []
    
    # Initialize generators outside the loop for efficiency
    generator = LLMScriptGenerator()

    # Initialize adaptive TTS wrapper
    adaptive_tts = AdaptiveTTSSynthesizer(tts_backend)
    
    # Track overall start time
    batch_start_time = time.time()
    
    for i, topic in enumerate(topics):
        LOGGER.info(f"--- Processing Topic {i+1}/{len(topics)}: {topic} ---")

        start_time = time.time()

        result = process_single_topic(
            topic, 
            prompt_template, 
            output_dir, 
            generator, 
            adaptive_tts,
            few_shot_nr=few_shot_nr,
            max_token_len=max_token_len,
            evaluate_audio_quality=evaluate_audio,
            track_timing=analyze_performance
            )

        elapsed_time = time.time() - start_time

        if result:
            # If not tracking detailed timing, use overall elapsed time
            if not analyze_performance:
                result['processing_time_sec'] = round(elapsed_time, 2)
            
            all_results.append(result)
            LOGGER.info(f"--- Topic {i+1} finished successfully in {elapsed_time:.2f} seconds. ---")
        else:
            LOGGER.info(f"--- Topic {i+1} failed after {elapsed_time:.2f} seconds. ---")

    # Calculate batch statistics
    batch_total_time = time.time() - batch_start_time
    
    # Save all results to a summary file
    summary_path = os.path.join(output_dir, f"batch_results_{tts_backend}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    
    LOGGER.info(f"Batch processing complete in {batch_total_time:.2f} seconds.")
    LOGGER.info(f"Results summary saved to: {summary_path}")
    
    # Performance Analysis
    if analyze_performance and all_results:
        LOGGER.info("\n" + "="*80)
        LOGGER.info("PERFORMANCE ANALYSIS")
        LOGGER.info("="*80)
        
        # Calculate latency statistics
        latency_stats = calculate_latency_statistics(all_results)
        
        # Generate and display performance table
        performance_table = generate_performance_table(all_results, latency_stats)
        print(performance_table)  # Print to console for immediate visibility
        
        # Save detailed performance report
        config = {
            "topic_input": topic_input,
            "prompt_template": prompt_template,
            "output_dir": output_dir,
            "tts_backend": tts_backend,
            "max_token_len": max_token_len,
            "few_shot_nr": few_shot_nr,
            "evaluate_audio": evaluate_audio,
            "analyze_performance": analyze_performance
        }
        
        report_path = save_performance_report(all_results, latency_stats, output_dir, config)
        
        # Log summary statistics
        if latency_stats:
            LOGGER.info(f"Average Latency: {latency_stats['average_latency']:.2f}s")
            LOGGER.info(f"Median Latency: {latency_stats['median_latency']:.2f}s")
            LOGGER.info(f"Throughput: {latency_stats['requests_per_minute']:.2f} requests/minute")
            LOGGER.info(f"Performance report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a podcast script and audio from a topic using LLM and TTS models."
    )
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
    parser.add_argument(
        "--tts_backend",
        type=str,
        default="kokoro",
        choices=["kokoro", "piper", "adaptive"],
        help="Which TTS backend to use for synthesis."
    )
    parser.add_argument(
        "--max_token_len",
        default=256,
        type=int,
        help="Max token length for script generation."
    )
    parser.add_argument(
        "--few_shot_nr",
        default=3,
        type=int,
        help="Number of few shot examples to format the prompt (max 10)"
    )
    parser.add_argument(
        "--evaluate_script",
        action="store_true",
        help="Enable script quality evaluation (relevance, coherence and compliance scores 1-10)"
    )
    parser.add_argument(
        "--evaluate_audio",
        action="store_true",
        help="Enable audio quality evaluation (adds WER calculation and audio metrics)"
    )
    parser.add_argument(
        "--analyze_performance",
        action="store_true",
        help="Enable performance analysis and latency calculation"
    )
    
    args = parser.parse_args()
    
    run_podcast_pipeline(
        args.topic, 
        args.prompt_template, 
        args.output_dir, 
        args.hf_token, 
        args.tts_backend,
        args.max_token_len,
        args.few_shot_nr,
        args.evaluate_script,
        args.evaluate_audio,
        args.analyze_performance
    )