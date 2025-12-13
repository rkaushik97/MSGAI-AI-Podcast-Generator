# Adaptive TTS Queuing Analysis Report

*Generated on: 2025-12-13 22:23:58*

## System Characteristics

### TTS Backend Profiles
| Backend | Service Time | Service Rate (µ) |
|---------|--------------|------------------|
| Kokoro (HQ) | 40.0s | 0.025 req/s |
| Piper (LQ) | 20.0s | 0.050 req/s |

### QoS Constraint
- Maximum acceptable response time (T_max): **90.0 seconds**
- Switching policy: Use Kokoro if n × 40 + 40 ≤ T_max, otherwise use Piper
- Stability condition: System stable only if λ < µ_piper = 0.050 req/s

## Experiment Results

| λ (req/s) | µ (req/s) | Response Time | Queue Length | Stable | Kokoro % | Piper % |
|-----------|-----------|---------------|--------------|--------|----------|---------|
| 0.015 | 0.031 | 55.9s | 1.0 | ✅ | 68.0% | 32.0% |
| 0.020 | 0.028 | 49.5s | 0.7 | ✅ | 77.8% | 22.2% |
| 0.025 | 0.033 | 46.4s | 1.5 | ✅ | 61.9% | 38.1% |
| 0.030 | 0.033 | 52.7s | 1.5 | ✅ | 60.0% | 40.0% |
| 0.035 | 0.038 | 50.3s | 1.9 | ✅ | 49.1% | 50.9% |
| 0.040 | 0.042 | 52.8s | 2.9 | ✅ | 40.3% | 59.7% |
| 0.045 | 0.054 | 55.5s | 4.5 | ✅ | 24.2% | 75.8% |

## Key Findings

1. **Stability Limit:** System remains stable up to λ = 0.045 req/s
   - This confirms λ < µ_piper = 0.050 req/s constraint
2. **QoS Compliance:** 7/7 experiments meet T_max = 90.0s constraint
3. **Adaptive Behavior:**
   - At λ = 0.015: Mixed use (68.0% Kokoro) - moderate load
   - At λ = 0.020: Mixed use (77.8% Kokoro) - moderate load
   - At λ = 0.025: Mixed use (61.9% Kokoro) - moderate load
   - At λ = 0.030: Mixed use (60.0% Kokoro) - moderate load
   - At λ = 0.035: Mostly Piper (50.9%) - heavy load
   - At λ = 0.040: Mostly Piper (59.7%) - heavy load
   - At λ = 0.045: Mostly Piper (75.8%) - heavy load

## Theoretical vs Practical Insights

1. **M/G/1 Model Validation:** The adaptive switching creates a state-dependent M/G/1 queue
2. **Capacity Planning:** Maximum sustainable load is defined by Piper's service rate
3. **Queue-Length Threshold:** Switching threshold n = (T_max - 40)/40 ≈ {(t_max - 40)/40:.1f} works effectively

## Recommendations

1. **For Production:** Implement similar adaptive policies for other bottleneck components
2. **For Scaling:** Add more Piper instances before Kokoro instances (Piper defines capacity)
3. **For Optimization:** Fine-tune T_max based on user tolerance studies
4. **For Monitoring:** Track backend usage percentages as load indicators

## Generated Visualizations

1. **Adaptive TTS Performance:** `adaptive_tts_performance.png`
   - Shows response time, backend usage, queue length, and service rate
   - Visualizes the adaptive switching behavior

2. **Static vs Adaptive Comparison:** `static_vs_adaptive_comparison.png`
   - Compares adaptive strategy with Kokoro-only baseline
   - Shows QoS improvement

## Raw Data

Complete results: `experiment_results.json`

## Experiment Configuration

```yaml
experiment:
  duration_seconds: 1800
  name: Adaptive TTS Queuing Analysis
  output_dir: adaptive_experiment_results
  topics_file: input/topics_batch1.txt
llm:
  max_new_tokens: 200
  model_id: meta-llama/Meta-Llama-3.1-8B-Instruct
  temperature: 0.7
logging:
  file: adaptive_experiment.log
  format: '%(asctime)s - %(levelname)s - %(message)s'
  level: INFO
pipeline:
  enable_audio_analysis: false
  prompt_template: podcast_script_v1
  t_max: 90.0
  tts_backend: adaptive
queuing:
  arrival_rates:
  - 0.015
  - 0.02
  - 0.025
  - 0.03
  - 0.035
  - 0.04
  - 0.045
```
