# podcast_pipeline/queuing_analysis.py
import os
import time
import json
import yaml
import statistics
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .utils import LOGGER
from .types import Script


class TTSBackend(Enum):
    """TTS backend options with their performance characteristics."""
    KOKORO = "kokoro"   # High quality, slow (~40s)
    PIPER = "piper"     # Lower quality, fast (~20s)


@dataclass
class TTSProfile:
    """Performance profile for a TTS backend."""
    name: str
    avg_service_time: float  # in seconds
    
    @property
    def service_rate(self) -> float:
        """Service rate μ (requests/second)."""
        return 1.0 / self.avg_service_time if self.avg_service_time > 0 else 0


# Define the TTS profiles based on your measurements
KOKORO_PROFILE = TTSProfile(
    name="Kokoro",
    avg_service_time=45.0  # ~40 seconds
)

PIPER_PROFILE = TTSProfile(
    name="Piper", 
    avg_service_time=20.0  # ~20 seconds
)


class AdaptiveTTSSynthesizer:
    """
    Adaptive TTS synthesizer that switches between Kokoro and Piper
    based on queue conditions and QoS constraints.
    """
    
    def __init__(self, config: Dict[str, Any], t_max: float = 90.0):
        """
        Initialize adaptive synthesizer.
        
        Args:
            t_max: Maximum acceptable response time (seconds)
        """
        self.config = config
        self.t_max = t_max  # QoS constraint: 90 seconds max response time
        
        # Initialize both synthesizers
        from .adaptive_tts_synthesizer import AdaptiveTTSSynthesizer as BaseSynthesizer
        self.kokoro = BaseSynthesizer(backend="kokoro")
        self.piper = BaseSynthesizer(backend="piper")
        
        # Current backend
        self.current_backend = TTSBackend.KOKORO
        self.backend_usage = {TTSBackend.KOKORO: 0, TTSBackend.PIPER: 0}
        
        # Statistics
        self.service_times = []
        
        LOGGER.info(f"Adaptive TTS initialized with T_max={t_max}s")
    
    def select_backend(self, queue_length: int) -> TTSBackend:
        """
        Select TTS backend based on queue length and QoS constraint.
        
        Uses the rule:
        Use Kokoro if: n × 40 + 40 ≤ T_max
        Switch to Piper if: n × 40 + 40 > T_max
        
        Where n = queue_length
        """
        # Calculate expected response time with Kokoro
        expected_response_kokoro = queue_length * KOKORO_PROFILE.avg_service_time + KOKORO_PROFILE.avg_service_time
        
        if expected_response_kokoro <= self.t_max:
            backend = TTSBackend.KOKORO
        else:
            backend = TTSBackend.PIPER
        
        if backend != self.current_backend:
            LOGGER.info(f"Switching TTS backend: {self.current_backend.value} -> {backend.value} "
                       f"(queue={queue_length}, expected_response={expected_response_kokoro:.1f}s)")
            self.current_backend = backend
        
        return backend
    
    def synthesize(self, script: Script, output_path: str, queue_length: int = 0) -> float:
        """
        Synthesize audio with adaptive backend selection.
        
        Returns:
            service_time
        """
        # Select backend based on queue
        backend = self.select_backend(queue_length)
        self.backend_usage[backend] += 1
        
        # Get the appropriate synthesizer
        synthesizer = self.kokoro if backend == TTSBackend.KOKORO else self.piper
        
        # Measure service time
        start_time = time.time()
        synthesizer.synthesize(script, output_path)
        service_time = time.time() - start_time
        
        # Store statistics
        self.service_times.append(service_time)
        
        return service_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about backend usage and performance."""
        total_requests = sum(self.backend_usage.values())
        
        return {
            "total_requests": total_requests,
            "kokoro_usage": self.backend_usage[TTSBackend.KOKORO],
            "piper_usage": self.backend_usage[TTSBackend.PIPER],
            "kokoro_percentage": self.backend_usage[TTSBackend.KOKORO] / total_requests * 100 if total_requests > 0 else 0,
            "avg_service_time": statistics.mean(self.service_times) if self.service_times else 0,
        }


class SequentialPipeline:
    """
    Processes podcast requests sequentially with adaptive TTS switching.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize LLM generator
        from .llm_generator import LLMScriptGenerator
        llm_config = config.get("llm", {})
        self.generator = LLMScriptGenerator(
            model_id=llm_config.get("model_id", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        )
        
        # Initialize adaptive TTS
        t_max = config.get("pipeline", {}).get("t_max", 90.0)  # QoS constraint
        self.synthesizer = AdaptiveTTSSynthesizer(config, t_max)
        
        # Statistics
        self.llm_times = []
        self.total_times = []
        self.response_times = []
        
        LOGGER.info(f"Sequential Pipeline initialized with adaptive TTS (T_max={t_max}s)")
    
    def process_request(self, topic: str, arrival_time: float, queue_length: int) -> Dict[str, Any]:
        """Process a single request with adaptive TTS."""
        pipeline_config = self.config.get("pipeline", {})
        output_dir = self.config.get("experiment", {}).get("output_dir", "output")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        
        # Stage 1: LLM Script Generation
        llm_start = time.time()
        script, _ = self.generator.generate(
            topic,
            pipeline_config.get("prompt_template", "podcast_script_v1"),
        )
        llm_time = time.time() - llm_start
        
        # Stage 2: Adaptive TTS Synthesis
        clean_topic = script["topic"].replace(":", "").replace("?", "").replace("/", "").strip()
        filename_prefix = clean_topic[:30].replace(' ', '_').lower()
        
        audio_filename = f"{filename_prefix}_adaptive_tts.wav"
        audio_file_path = os.path.join(output_dir, audio_filename)
        
        # Save script
        script_filename = f"{filename_prefix}_script.md"
        script_file_path = os.path.join(output_dir, script_filename)
        
        with open(script_file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Podcast Script: {script['topic']}\n\n")
            f.write(f"Metadata: Host={script['metadata']['HOST_GENDER']}, Guest={script['metadata']['GUEST_GENDER']}\n\n")
            f.write("--- Dialogue ---\n\n")
            f.write(script['dialogue'])
        
        # Use adaptive TTS with queue-aware backend selection
        tts_time = self.synthesizer.synthesize(
            script, 
            audio_file_path,
            queue_length=queue_length
        )
        
        total_time = time.time() - start_time
        
        # Calculate response time (arrival to completion)
        response_time = (time.time() - arrival_time) if arrival_time > 0 else total_time
        
        # Store statistics
        self.llm_times.append(llm_time)
        self.total_times.append(total_time)
        self.response_times.append(response_time)
        
        LOGGER.debug(f"Processed '{topic[:30]}...' in {total_time:.2f}s (LLM: {llm_time:.2f}s, TTS: {tts_time:.2f}s)")
        
        return {
            "topic": topic,
            "llm_time": llm_time,
            "tts_time": tts_time,
            "total_time": total_time,
            "response_time": response_time,
            "audio_file": audio_file_path,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from processed requests."""
        if not self.total_times:
            return {}
        
        # Get pipeline stats
        pipeline_stats = {
            "total_requests": len(self.total_times),
            "avg_llm_time": statistics.mean(self.llm_times) if self.llm_times else 0,
            "avg_total_time": statistics.mean(self.total_times) if self.total_times else 0,
            "avg_response_time": statistics.mean(self.response_times) if self.response_times else 0,
            "service_rate": 1 / statistics.mean(self.total_times) if self.total_times else 0,
        }
        
        # Get TTS stats
        tts_stats = self.synthesizer.get_statistics()
        
        return {**pipeline_stats, **tts_stats}


class QueuingSimulator:
    """
    Simulates M/G/1 queue with adaptive TTS switching policy.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        
    def simulate_arrival_rate(self, arrival_rate: float) -> Dict[str, Any]:
        """
        Simulate processing with given arrival rate (λ) and adaptive switching.
        """
        experiment_config = self.config.get("experiment", {})
        duration = experiment_config.get("duration_seconds", 300)
        topics_file = experiment_config.get("topics_file", "input/topics.txt")
        
        # Read topics
        with open(topics_file, 'r') as f:
            topics = [line.strip() for line in f if line.strip()]
        
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"Simulating M/G/1 with adaptive TTS: λ={arrival_rate}/s")
        LOGGER.info(f"{'='*80}")
        
        # Initialize pipeline with adaptive TTS
        pipeline = SequentialPipeline(self.config)
        
        # Generate arrival times (Poisson process)
        arrival_times = []
        current_time = 0
        while current_time < duration:
            inter_arrival = np.random.exponential(1.0 / arrival_rate)
            current_time += inter_arrival
            if current_time < duration:
                arrival_times.append(current_time)
        
        # Simulate M/G/1 queue with adaptive service
        response_times = []
        service_times = []
        queue_length_history = []
        current_queue = 0
        
        # Track when the server will be free
        server_free_time = 0
        
        request_idx = 0
        
        for arrival_time in arrival_times:
            # Update queue based on server status
            if arrival_time >= server_free_time:
                current_queue = 0  # Server is idle
            else:
                current_queue += 1  # New arrival joins queue
            
            queue_length_history.append(current_queue)
            
            # Select topic
            topic = topics[request_idx % len(topics)]
            request_idx += 1
            
            # Start processing time (when server becomes available)
            start_processing = max(arrival_time, server_free_time)
            
            # Process with adaptive TTS (queue length affects TTS choice)
            result = pipeline.process_request(topic, arrival_time, current_queue)
            service_time = result["total_time"]
            
            # Update server free time
            server_free_time = start_processing + service_time
            
            # Calculate response time
            response_time = server_free_time - arrival_time
            
            response_times.append(response_time)
            service_times.append(service_time)
            
            # Log progress
            if len(response_times) % 5 == 0:
                LOGGER.info(f"Processed {len(response_times)}/{len(arrival_times)} requests")
        
        # Calculate statistics
        pipeline_stats = pipeline.get_statistics()
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            avg_service_time = statistics.mean(service_times)
            avg_queue_length = statistics.mean(queue_length_history) if queue_length_history else 0
            
            # Calculate service rate (µ) - this varies due to adaptive switching!
            service_rate = 1 / avg_service_time if avg_service_time > 0 else 0
            
            # Stability condition: λ < µ_piper (hard limit)
            µ_piper = PIPER_PROFILE.service_rate  # ~0.05 req/s
            is_stable = arrival_rate < µ_piper
            
            # Calculate throughput
            total_processing_time = sum(service_times)
            throughput_rpm = (len(service_times) / total_processing_time * 60) if total_processing_time > 0 else 0
            
            result = {
                "arrival_rate": arrival_rate,
                "service_rate": service_rate,
                "avg_service_time": avg_service_time,
                "avg_response_time": avg_response_time,
                "avg_queue_length": avg_queue_length,
                "is_stable": is_stable,
                "total_requests": len(arrival_times),
                "completed_requests": len(response_times),
                "throughput_rpm": throughput_rpm,
                "avg_llm_time": pipeline_stats.get("avg_llm_time", 0),
                "tts_kokoro_usage": pipeline_stats.get("kokoro_percentage", 0),
                "tts_piper_usage": 100 - pipeline_stats.get("kokoro_percentage", 0),
                "tts_kokoro_count": pipeline_stats.get("kokoro_usage", 0),
                "tts_piper_count": pipeline_stats.get("piper_usage", 0),
            }
            
            stability_msg = "STABLE" if is_stable else f"UNSTABLE (λ={arrival_rate:.3f} > µ_piper={µ_piper:.3f})"
            LOGGER.info(f"Simulation completed: µ={service_rate:.3f}, {stability_msg}")
            LOGGER.info(f"TTS usage: Kokoro={result['tts_kokoro_usage']:.1f}%, Piper={result['tts_piper_usage']:.1f}%")
            
            return result
        
        return {}
    
    def run_simulations(self) -> List[Dict[str, Any]]:
        """Run simulations for all configured arrival rates."""
        arrival_rates = self.config.get("queuing", {}).get("arrival_rates", [0.01, 0.02, 0.03, 0.04, 0.05])
        
        results = []
        for arrival_rate in arrival_rates:
            result = self.simulate_arrival_rate(arrival_rate)
            if result:
                results.append(result)
                time.sleep(1)  # Brief pause
        
        self.results = {r["arrival_rate"]: r for r in results}
        return results


class ExperimentVisualizer:
    """Creates plots and reports for adaptive TTS experiments."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = config.get("experiment", {}).get("output_dir", "experiment_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_adaptive_performance(self, results: List[Dict[str, Any]]):
        """Plot adaptive TTS performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        arrival_rates = [r["arrival_rate"] for r in results]
        
        # Plot 1: Response Time vs Arrival Rate
        ax1 = axes[0, 0]
        response_times = [r["avg_response_time"] for r in results]
        ax1.plot(arrival_rates, response_times, 'o-', linewidth=2, markersize=8)
        ax1.axhline(y=self.config.get("pipeline", {}).get("t_max", 90), color='r', linestyle='--', 
                   label=f'QoS constraint (T_max={self.config.get("pipeline", {}).get("t_max", 90)}s)')
        ax1.set_xlabel('Arrival Rate (λ, requests/second)')
        ax1.set_ylabel('Average Response Time (seconds)')
        ax1.set_title('Response Time vs Arrival Rate with Adaptive TTS')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: TTS Backend Usage
        ax2 = axes[0, 1]
        kokoro_usage = [r["tts_kokoro_usage"] for r in results]
        piper_usage = [r["tts_piper_usage"] for r in results]
        
        width = 0.35
        x = np.arange(len(arrival_rates))
        ax2.bar(x - width/2, kokoro_usage, width, label='Kokoro (HQ)', color='#2E86AB', alpha=0.7)
        ax2.bar(x + width/2, piper_usage, width, label='Piper (LQ)', color='#A23B72', alpha=0.7)
        ax2.set_xlabel('Arrival Rate (λ, requests/second)')
        ax2.set_ylabel('Usage Percentage (%)')
        ax2.set_title('TTS Backend Usage vs Load')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{λ:.3f}' for λ in arrival_rates])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Queue Length vs Arrival Rate
        ax3 = axes[1, 0]
        queue_lengths = [r["avg_queue_length"] for r in results]
        ax3.plot(arrival_rates, queue_lengths, '^-', linewidth=2, markersize=8, color='purple')
        ax3.set_xlabel('Arrival Rate (λ, requests/second)')
        ax3.set_ylabel('Average Queue Length')
        ax3.set_title('Queue Length vs Arrival Rate')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Service Rate vs Arrival Rate
        ax4 = axes[1, 1]
        service_rates = [r["service_rate"] for r in results]
        ax4.plot(arrival_rates, service_rates, 's-', linewidth=2, markersize=8, color='green')
        ax4.axhline(y=KOKORO_PROFILE.service_rate, color='#2E86AB', linestyle=':', label=f'Kokoro µ={KOKORO_PROFILE.service_rate:.3f}')
        ax4.axhline(y=PIPER_PROFILE.service_rate, color='#A23B72', linestyle=':', label=f'Piper µ={PIPER_PROFILE.service_rate:.3f}')
        ax4.set_xlabel('Arrival Rate (λ, requests/second)')
        ax4.set_ylabel('Service Rate (µ, requests/second)')
        ax4.set_title('Service Rate vs Arrival Rate')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "adaptive_tts_performance.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        LOGGER.info(f"Adaptive TTS performance plot saved to {plot_path}")
    
    def plot_comparison_static_vs_adaptive(self, results: List[Dict[str, Any]]):
        """Plot comparison between static Kokoro-only and adaptive strategy."""
        # Calculate what Kokoro-only would achieve
        kokoro_only_results = []
        for r in results:
            λ = r["arrival_rate"]
            µ_kokoro = KOKORO_PROFILE.service_rate  # ~0.025
            
            # For M/G/1 with only Kokoro
            ρ = λ / µ_kokoro
            avg_queue_kokoro = ρ / (1 - ρ) if ρ < 1 else float('inf')
            avg_response_kokoro = avg_queue_kokoro * KOKORO_PROFILE.avg_service_time + KOKORO_PROFILE.avg_service_time
            
            kokoro_only_results.append({
                "arrival_rate": λ,
                "avg_response_time": avg_response_kokoro if ρ < 1 else float('inf'),
                "avg_queue_length": avg_queue_kokoro if ρ < 1 else float('inf'),
                "is_stable": ρ < 1
            })
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Response Time Comparison
        ax1 = axes[0]
        λ_values = [r["arrival_rate"] for r in results]
        
        # Adaptive response times (from simulation)
        adaptive_response = [r["avg_response_time"] for r in results]
        
        # Kokoro-only response times (calculated)
        kokoro_response = [r["avg_response_time"] for r in kokoro_only_results]
        
        ax1.plot(λ_values, adaptive_response, 'o-', linewidth=2, markersize=6, 
                label='Adaptive TTS', color='#2E86AB')
        ax1.plot(λ_values, kokoro_response, 's--', linewidth=2, markersize=6, 
                label='Kokoro-only', color='#A23B72')
        
        ax1.axhline(y=self.config.get("pipeline", {}).get("t_max", 90), color='r', 
                   linestyle=':', label='QoS constraint')
        ax1.set_xlabel('Arrival Rate (λ, requests/second)')
        ax1.set_ylabel('Average Response Time (seconds)')
        ax1.set_title('Adaptive vs Static TTS: Response Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Queue Length Comparison
        ax2 = axes[1]
        adaptive_queue = [r["avg_queue_length"] for r in results]
        kokoro_queue = [r["avg_queue_length"] for r in kokoro_only_results]
        
        ax2.plot(λ_values, adaptive_queue, 'o-', linewidth=2, markersize=6, 
                label='Adaptive TTS', color='#2E86AB')
        ax2.plot(λ_values, kokoro_queue, 's--', linewidth=2, markersize=6, 
                label='Kokoro-only', color='#A23B72')
        
        ax2.set_xlabel('Arrival Rate (λ, requests/second)')
        ax2.set_ylabel('Average Queue Length')
        ax2.set_title('Adaptive vs Static TTS: Queue Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "static_vs_adaptive_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        LOGGER.info(f"Static vs adaptive comparison plot saved to {plot_path}")
    
    def generate_markdown_report(self, results: List[Dict[str, Any]]):
        """Generate comprehensive markdown report for adaptive TTS experiments."""
        report_path = os.path.join(self.output_dir, "adaptive_tts_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# Adaptive TTS Queuing Analysis Report\n\n")
            f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            # System Characteristics
            f.write("## System Characteristics\n\n")
            f.write("### TTS Backend Profiles\n")
            f.write("| Backend | Service Time | Service Rate (µ) |\n")
            f.write("|---------|--------------|------------------|\n")
            f.write(f"| Kokoro (HQ) | {KOKORO_PROFILE.avg_service_time:.1f}s | {KOKORO_PROFILE.service_rate:.3f} req/s |\n")
            f.write(f"| Piper (LQ) | {PIPER_PROFILE.avg_service_time:.1f}s | {PIPER_PROFILE.service_rate:.3f} req/s |\n")
            f.write("\n")
            
            f.write(f"### QoS Constraint\n")
            f.write(f"- Maximum acceptable response time (T_max): **{self.config.get('pipeline', {}).get('t_max', 90)} seconds**\n")
            f.write(f"- Switching policy: Use Kokoro if n × 40 + 40 ≤ T_max, otherwise use Piper\n")
            f.write(f"- Stability condition: System stable only if λ < µ_piper = {PIPER_PROFILE.service_rate:.3f} req/s\n\n")
            
            # Results Table
            f.write("## Experiment Results\n\n")
            f.write("| λ (req/s) | µ (req/s) | Response Time | Queue Length | Stable | Kokoro % | Piper % |\n")
            f.write("|-----------|-----------|---------------|--------------|--------|----------|---------|\n")
            
            for result in results:
                λ = result["arrival_rate"]
                µ = result["service_rate"]
                response = result["avg_response_time"]
                queue = result["avg_queue_length"]
                stable = "✅" if result["is_stable"] else "❌"
                kokoro_pct = result["tts_kokoro_usage"]
                piper_pct = result["tts_piper_usage"]
                
                f.write(f"| {λ:.3f} | {µ:.3f} | {response:.1f}s | {queue:.1f} | {stable} | {kokoro_pct:.1f}% | {piper_pct:.1f}% |\n")
            
            f.write("\n")
            
            # Key Findings
            f.write("## Key Findings\n\n")
            
            # Find stability limit
            stable_rates = [r["arrival_rate"] for r in results if r["is_stable"]]
            if stable_rates:
                max_stable = max(stable_rates)
                f.write(f"1. **Stability Limit:** System remains stable up to λ = {max_stable:.3f} req/s\n")
                f.write(f"   - This confirms λ < µ_piper = {PIPER_PROFILE.service_rate:.3f} req/s constraint\n")
            else:
                f.write("1. **Stability Limit:** System unstable for all tested arrival rates\n")
            
            # QoS compliance
            t_max = self.config.get("pipeline", {}).get("t_max", 90)
            compliant_experiments = sum(1 for r in results if r["avg_response_time"] <= t_max)
            f.write(f"2. **QoS Compliance:** {compliant_experiments}/{len(results)} experiments meet T_max = {t_max}s constraint\n")
            
            # Adaptive behavior analysis
            f.write("3. **Adaptive Behavior:**\n")
            for r in results:
                λ = r["arrival_rate"]
                kokoro_pct = r["tts_kokoro_usage"]
                if kokoro_pct > 90:
                    f.write(f"   - At λ = {λ:.3f}: Mostly Kokoro ({kokoro_pct:.1f}%) - light load\n")
                elif kokoro_pct > 50:
                    f.write(f"   - At λ = {λ:.3f}: Mixed use ({kokoro_pct:.1f}% Kokoro) - moderate load\n")
                else:
                    f.write(f"   - At λ = {λ:.3f}: Mostly Piper ({100-kokoro_pct:.1f}%) - heavy load\n")
            f.write("\n")
            
            # Theoretical vs Practical
            f.write("## Theoretical vs Practical Insights\n\n")
            f.write("1. **M/G/1 Model Validation:** The adaptive switching creates a state-dependent M/G/1 queue\n")
            f.write("2. **Capacity Planning:** Maximum sustainable load is defined by Piper's service rate\n")
            f.write("3. **Queue-Length Threshold:** Switching threshold n = (T_max - 40)/40 ≈ {(t_max - 40)/40:.1f} works effectively\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("1. **For Production:** Implement similar adaptive policies for other bottleneck components\n")
            f.write("2. **For Scaling:** Add more Piper instances before Kokoro instances (Piper defines capacity)\n")
            f.write("3. **For Optimization:** Fine-tune T_max based on user tolerance studies\n")
            f.write("4. **For Monitoring:** Track backend usage percentages as load indicators\n\n")
            
            # Plots Section
            f.write("## Generated Visualizations\n\n")
            f.write("1. **Adaptive TTS Performance:** `adaptive_tts_performance.png`\n")
            f.write("   - Shows response time, backend usage, queue length, and service rate\n")
            f.write("   - Visualizes the adaptive switching behavior\n\n")
            
            f.write("2. **Static vs Adaptive Comparison:** `static_vs_adaptive_comparison.png`\n")
            f.write("   - Compares adaptive strategy with Kokoro-only baseline\n")
            f.write("   - Shows QoS improvement\n\n")
            
            # Raw Data
            f.write("## Raw Data\n\n")
            f.write("Complete results: `experiment_results.json`\n\n")
            
            # Configuration
            f.write("## Experiment Configuration\n\n")
            f.write("```yaml\n")
            f.write(yaml.dump(self.config, default_flow_style=False))
            f.write("```\n")
        
        LOGGER.info(f"Adaptive TTS report saved to {report_path}")
    
    def save_results_json(self, results: List[Dict[str, Any]]):
        """Save results to JSON file."""
        json_path = os.path.join(self.output_dir, "adaptive_tts_results.json")
        
        with open(json_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "tts_profiles": {
                    "kokoro": KOKORO_PROFILE.__dict__,
                    "piper": PIPER_PROFILE.__dict__
                },
                "config": self.config,
                "results": results
            }, f, indent=2)
        
        LOGGER.info(f"Results saved to {json_path}")


def run_adaptive_tts_analysis(config_file: str = "config.yaml"):
    """Main function to run adaptive TTS queuing analysis."""
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup environment
    from .utils import setup_environment
    setup_environment()
    
    LOGGER.info("Starting adaptive TTS queuing analysis...")
    
    # Run simulations
    simulator = QueuingSimulator(config)
    results = simulator.run_simulations()
    
    if not results:
        LOGGER.error("No results generated from simulations")
        return
    
    # Create visualizations and reports
    visualizer = ExperimentVisualizer(config)
    
    # Generate plots
    visualizer.plot_adaptive_performance(results)
    visualizer.plot_comparison_static_vs_adaptive(results)
    
    # Generate reports
    visualizer.generate_markdown_report(results)
    visualizer.save_results_json(results)
    
    # Print summary
    print("\n" + "="*80)
    print("ADAPTIVE TTS QUEUING ANALYSIS COMPLETE")
    print("="*80)
    print(f"System Characteristics:")
    print(f"  Kokoro (HQ): µ={KOKORO_PROFILE.service_rate:.3f} req/s")
    print(f"  Piper (LQ):  µ={PIPER_PROFILE.service_rate:.3f} req/s")
    print(f"  QoS Constraint: T_max={config.get('pipeline', {}).get('t_max', 90)}s")
    print(f"  Stability Condition: λ < µ_piper = {PIPER_PROFILE.service_rate:.3f} req/s")
    print("\nExperiment Results:")
    
    for result in results:
        λ = result["arrival_rate"]
        µ = result["service_rate"]
        response = result["avg_response_time"]
        kokoro_pct = result["tts_kokoro_usage"]
        stable = "STABLE" if result["is_stable"] else "UNSTABLE"
        
        print(f"  λ={λ:.3f}: µ={µ:.3f} | Response: {response:.1f}s | "
              f"Kokoro: {kokoro_pct:.1f}% | {stable}")
    
    print(f"\nReports saved to: {config.get('experiment', {}).get('output_dir', 'experiment_results')}")
    print("="*80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run adaptive TTS queuing analysis")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    args = parser.parse_args()
    run_adaptive_tts_analysis(args.config)