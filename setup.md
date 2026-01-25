This repository demonstrates how to run an open-source DeepSeek model locally using vLLM with tensor parallelism, achieving website-like chat performance on lab hardware.

Hardware Used

GPUs: 2 × NVIDIA RTX 4090 (24 GB VRAM each)

System RAM: 64 GB

Storage: 1 TB SSD

OS: Linux (Ubuntu recommended)

CUDA: 12.x

This setup is sufficient for large dense DeepSeek models, but not data-center-scale MoE models like DeepSeek-R1.

Model Choice

Selected Model:
👉 deepseek-ai/DeepSeek-V2

Why this model?

Strong reasoning and chat quality

Optimized for inference

Runs smoothly on dual consumer GPUs

Practical alternative to MoE models that require servers

Project Structure
deepseek-local/
├── README.md
├── setup.sh
├── run_server.sh
└── test_chat.py

Step 1: Environment Setup

Create and activate a Python environment:

conda create -n deepseek python=3.10 -y
conda activate deepseek


Install PyTorch (CUDA 12):

pip install torch --index-url https://download.pytorch.org/whl/cu121


Install vLLM:

pip install vllm


(Optional but recommended)

pip install transformers accelerate sentencepiece

Step 2: Download DeepSeek Model

You can either let vLLM download automatically or clone manually.

Option A: Automatic (recommended)

vLLM will fetch the model on first run.

Option B: Manual clone
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-V2

Step 3: Start the DeepSeek Inference Server

Create run_server.sh:

#!/bin/bash

python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-V2 \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8000


Make it executable:

chmod +x run_server.sh


Run the server:

./run_server.sh


This launches an OpenAI-compatible API server backed by DeepSeek.

Step 4: Test via CURL
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V2",
    "messages": [
      {"role": "user", "content": "Explain Mixture of Experts in simple terms"}
    ]
  }'


Expected behavior:

Response in a few seconds

Smooth multi-sentence output

No GPU OOM errors

Step 5: Test via Python Client

Create test_chat.py:

import requests

url = "http://localhost:8000/v1/chat/completions"

payload = {
    "model": "deepseek-ai/DeepSeek-V2",
    "messages": [
        {"role": "user", "content": "Explain attention mechanisms simply"}
    ],
    "temperature": 0.7
}

response = requests.post(url, json=payload)
print(response.json()["choices"][0]["message"]["content"])


Run:

python test_chat.py

Expected Performance
Metric	Value
VRAM usage	~22–24 GB per GPU
RAM usage	~20–30 GB
Tokens/sec	~15–30
Context length	Up to 8K tokens
Latency	Comparable to mid-tier hosted LLMs
Important Notes

This setup does NOT support DeepSeek-R1 or large MoE models

MoE models require:

80 GB VRAM GPUs

NVLink / InfiniBand

Data-center infrastructure

DeepSeek-V2 provides the best quality-to-hardware ratio for local labs

Key Technical Concepts Demonstrated:

Transformer attention mechanisms

Large-model inference optimization

Tensor parallelism across GPUs

KV-cache management (via vLLM)

Production-style LLM serving
