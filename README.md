# Chromatic Orchestrator 🎨

**A blazing-fast, production-ready vision classification pipeline powered by multimodal LLMs with advanced data parallel support**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![vLLM](https://img.shields.io/badge/vLLM-0.9.1-green.svg)](https://github.com/vllm-project/vllm)

Chromatic Orchestrator brings the architectural excellence of [TrialMesh](https://github.com/mikeS141618/trialmesh) to vision AI, implementing a sophisticated three-stage pipeline that processes images through classification, scoring, and summarization using state-of-the-art vision language models. Now featuring **advanced data parallel processing** for massive scale deployments.

## 🚀 First-Shot Success Story

```bash
chromatic-pipeline \
  --model-path /path/to/vision-model \
  --image-dir ./images \
  --recursive \
  --output-dir ./run \
  --tensor-parallel-size 2 \
  --batch-size 16
```

## 🎯 Key Features

### Three-Stage Vision Pipeline
1. **Classification** - Dynamically categorize images based on visual content
2. **Scoring** - Apply category-specific quality assessment (0-10 scale)
3. **Summarization** - Generate detailed, contextual descriptions

### Production-Ready Architecture
- **Advanced Data Parallel Processing** - Scale across multiple GPU clusters with isolated processes
- **Massive Cross-Image Batching** - Process thousands of images efficiently
- **Hash-Based Caching with Rank Isolation** - Never reprocess, even across distributed workers
- **Multi-GPU Support** - Configurable tensor parallelism for any scale
- **Fault Tolerance** - Graceful handling of errors with comprehensive logging
- **GPU Memory Optimization** - Intelligent GPU assignment and memory management

### Flexible Configuration
- **File-Based Prompts** - Customize behavior without touching code
- **Dynamic Categories** - Automatically extract categories from prompts
- **Conditional Processing** - Score thresholds, category filters, and more
- **Multiple Model Support** - Llama 3.2 Vision, Llama 4 Scout
- **Scalable GPU Allocation** - From single GPU to multi-node clusters

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/mikeS141618/chromatic-orchestrator.git
cd chromatic-orchestrator

# Install in development mode
pip install -e .

# Verify installation
chromatic-pipeline --model-path /path/to/vision-model --list-prompts
```

## 🎮 Quick Start

### Basic Single-GPU Pipeline
```bash
# Process all images in a directory
chromatic-pipeline \
  --model-path /path/to/vision-model \
  --image-dir ./images \
  --output-dir ./results \
  --tensor-parallel-size 1 \
  --batch-size 8
```

### Multi-GPU Tensor Parallel
```bash
# Use 2 GPUs for tensor parallelism
chromatic-pipeline \
  --model-path /path/to/vision-model \
  --image-dir ./images \
  --recursive \
  --batch-size 32 \
  --tensor-parallel-size 2 \
  --available-gpus 0 1 \
  --output-dir ./results
```

### **NEW: Data Parallel Processing**
```bash
# 3 data parallel ranks, 2 GPUs per rank (6 GPUs total)
chromatic-pipeline \
  --model-path /path/to/Llama-3.2-11B-Vision-Instruct-FP8-dynamic \
  --image-dir ./images \
  --recursive \
  --output-dir ./run \
  --dp-size 3 \
  --tensor-parallel-size 2 \
  --batch-size 8 \
  --available-gpus 0 1 2 3 4 5 \
  --log-level INFO
```

### Large Model Deployment
```bash
# Llama 4 Scout on 4 GPUs with massive batching
chromatic-pipeline \
  --model-path /path/to/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16 \
  --image-dir ./images \
  --recursive \
  --output-dir ./run \
  --dp-size 1 \
  --tensor-parallel-size 4 \
  --batch-size 64 \
  --available-gpus 0 1 2 3 \
  --log-level INFO
```

### Advanced Conditional Processing
```bash
# Full pipeline with filtering and thresholds
chromatic-pipeline \
  --model-path /path/to/vision-model \
  --image-dir ./images \
  --recursive \
  --batch-size 32 \
  --tensor-parallel-size 2 \
  --classification-prompt my_custom_classifier \
  --score-threshold 7.0 \
  --categories-to-summarize A B C \
  --cache-dir ./my_cache \
  --dp-size 2 \
  --available-gpus 0 1 2 3
```

## 🛠️ Command-Line Tools

### chromatic-pipeline
Complete three-stage pipeline with data parallel support:
```bash
chromatic-pipeline --help
```

**Key Parameters:**
- `--dp-size`: Number of data parallel ranks (default: 1)
- `--tensor-parallel-size`: GPUs per rank (default: 2)
- `--available-gpus`: Specific GPU IDs to use
- `--batch-size`: Images per batch (default: 8)

### chromatic-classify
Run classification only:
```bash
chromatic-classify \
  --model-path /path/to/model \
  --image-dir ./images \
  --classification-prompt custom_classifier \
  --tensor-parallel-size 2
```

### chromatic-score
Score pre-classified images:
```bash
chromatic-score \
  --model-path /path/to/model \
  --classification-results ./run/classification/results.json \
  --categories-to-process A B
```

### chromatic-summarize
Generate summaries for scored images:
```bash
chromatic-summarize \
  --model-path /path/to/model \
  --scoring-results ./run/scoring/results.json \
  --score-threshold 7.0
```

### chromatic-codemd
Generate documentation:
```bash
chromatic-codemd  # Creates codecomplete.md and prompts.md
```

## 📝 Prompt System

Prompts are stored as text files in `./prompts/`:

```
prompts/
├── classification_default.txt      # Main classifier
├── scoring_category_a.txt         # Scoring for natural landscapes
├── scoring_category_b.txt         # Scoring for urban scenes
├── summary_category_a.txt         # Summaries for natural landscapes
└── summary_category_b.txt         # Summaries for urban scenes
```

### Example Classification Prompt

```text
==== SYSTEM PROMPT ====
You are an expert image classifier.

==== USER PROMPT ====
Classify this image into one of the following categories:

Category A: Natural landscapes
Category B: Urban/architectural scenes

Respond with:
CATEGORY: [A or B]
REASONING: [Brief explanation]
```

## 📁 Output Structure

### Single Process Output
```
run/
├── classification/
│   └── classification_results.json    # Category assignments
├── scoring/
│   └── scoring_results.json          # Quality scores
├── summaries/
│   └── summary_results.json          # Generated descriptions
└── pipeline_results.json             # Complete results
```

### Data Parallel Output
```
run/
├── rank_0/
│   ├── classification/
│   ├── scoring/
│   ├── summaries/
│   └── pipeline_results.json
├── rank_1/
│   └── ...
├── rank_2/
│   └── ...
└── pipeline_results.json             # Merged results from all ranks
```

### Sample Output

```json
{
  "image_path": "landscapes/mountain_sunset.jpg",
  "classification": {
    "category": "A",
    "reasoning": "Shows natural mountain landscape with sunset"
  },
  "scoring": {
    "score": 8.5,
    "reasoning": "Excellent composition, dramatic lighting..."
  },
  "summary": {
    "summary": "This breathtaking mountain landscape captures..."
  }
}
```

## 🚀 Supported Models

- **Llama-3.2-11B-Vision-Instruct** (FP8/FP16)
- **Llama-3.2-90B-Vision-Instruct-FP8-dynamic**
- **Llama-4-Scout-17B** (quantized/FP8)
- Any vLLM-compatible vision model

## 🔧 GPU Configuration Guide

### Single Node Configurations
```bash
# 2x RTX 4090 (24GB each)
--dp-size 1 --tensor-parallel-size 2 --batch-size 16

# 4x RTX 4090 - Data Parallel
--dp-size 2 --tensor-parallel-size 2 --batch-size 8 --available-gpus 0 1 2 3

# 8x A100 (80GB each) - Large Scale
--dp-size 4 --tensor-parallel-size 2 --batch-size 32 --available-gpus 0 1 2 3 4 5 6 7
```

### Batch Size Optimization
```bash
# Conservative (stable)
--batch-size 8

# Balanced (recommended)
--batch-size 16-32

# Aggressive (high memory)
--batch-size 64+
```

### Cache Management
```bash
# Use custom cache directory
--cache-dir /fast/ssd/chromatic_cache

# Disable caching (not recommended)
export CHROMATIC_NO_CACHE=1
```

## 🏗️ Architecture

Chromatic Orchestrator adapts the proven architecture from TrialMesh with major enhancements:

### **NEW: Data Parallel Architecture**
- **Process Isolation**: Each DP rank runs in a separate process with dedicated GPUs
- **Independent Caching**: Rank-isolated cache to prevent conflicts
- **Result Merging**: Automatic aggregation of results from all ranks
- **Fault Tolerance**: Individual rank failures don't crash the entire job

### Core Features
- **Massive Batching Strategy** - Process entire datasets in cross-image batches
- **Efficient Caching** - Hash-based caching with image metadata awareness
- **Modular Pipeline** - Each stage can run independently
- **Extensible Design** - Easy to add new categories, prompts, or stages
- **GPU Memory Optimization** - Intelligent memory management across devices

### Scaling Characteristics
```
Single GPU:     10-50 images/minute
Multi-GPU TP:   50-200 images/minute  
Data Parallel:  200-1000+ images/minute
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/
```

## 📚 Citation

If you use Chromatic Orchestrator in your research, please cite:

```bibtex
@software{chromatic_orchestrator_2025,
  title = {Chromatic Orchestrator: Production-Ready Vision Classification Pipeline with Data Parallel Processing},
  author = {mikeS141618},
  year = {2025},
  url = {https://github.com/mikeS141618/chromatic-orchestrator}
}
```

## 🙏 Acknowledgments

- [TrialMesh](https://github.com/mikeS141618/trialmesh) - Architectural inspiration
- [vLLM Project](https://github.com/vllm-project/vllm) - Inference engine
- [Meta Llama](https://ai.meta.com/llama/) - Vision language models

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE) for details.
