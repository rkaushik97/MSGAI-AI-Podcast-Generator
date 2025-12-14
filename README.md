# Adaptive AI Podcast Clip Generator – Wizard of POdz

This repository contains the implementation of an adaptive AI system that generates short podcast introduction audio clips by combining a Large Language Model (LLM) and Text-to-Speech (TTS) models. The project was developed as part of the *Modeling and Scaling of Generative AI Systems* course.

---

## Overview

The pipeline consists of two stages:

1. **Text Generation**: A structured podcast script is generated using an LLM.

2. **Speech Synthesis**: The script is converted into audio using a TTS model (Kokoro or Piper).

The system supports multiple configurations and is designed to study latency–quality tradeoffs under different workloads.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <REPO_URL>
cd MSGAI-AI-Podcast-Generator
```

### 2. Download Kokoro TTS model files

Create a directory called kokoro at the root of the repository and download the following files:

```bash
mkdir kokoro
```

- Model file (ONNX)

    <https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx>

- Voice embeddings file (BIN)

    <https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin>

### 3. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  #Linux/macOS
# venv\Scripts\Activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the podcast generator

Navigate to the main module and run:

```bash
cd podcast-generator
python main.py
```

Command-line flags are available to control generation settings. Use:

```bash
python main.py --help
```

to see all available options.

