#!/usr/bin/env python3
# src/chromatic/cli/summarize.py

import argparse
import json
import logging
import sys
from pathlib import Path

from chromatic.vision.llama_vision_runner import LlamaVisionRunner
from chromatic.vision.summarizer import ImageSummarizer
from chromatic.vision.scorer import ScoringResult
from chromatic.utils.prompt_registry import PromptRegistry


def setup_logging(log_level: str = "INFO"):
    """Configure logging."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run image summarization stage on scored images"
    )

    # Model configuration
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the vision LLM model directory")
    parser.add_argument("--tensor-parallel-size", type=int, default=2,
                        help="Number of GPUs for tensor parallelism (default: 2)")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Maximum model context length (default: 8192)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for processing (default: 8)")

    # Input/output paths
    parser.add_argument("--scoring-results", type=str, required=True,
                        help="Path to scoring results JSON file")
    parser.add_argument("--output-dir", type=str, default="./run/summaries",
                        help="Output directory for results (default: ./run/summaries)")
    parser.add_argument("--cache-dir", type=str, default="./cache/vision_responses",
                        help="Cache directory for LLM responses")
    parser.add_argument("--prompt-dir", type=str, default="./prompts",
                        help="Directory containing prompt files")

    # Summarization options
    parser.add_argument("--max-tokens", type=int,
                        help="Max tokens for summarization")
    parser.add_argument("--temperature", type=float,
                        help="Temperature for summarization")
    parser.add_argument("--score-threshold", type=float,
                        help="Minimum score to process")
    parser.add_argument("--categories-to-process", type=str, nargs="+",
                        help="Only summarize these categories")

    # Utility options
    parser.add_argument("--list-prompts", action="store_true",
                        help="List available prompts and exit")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level (default: INFO)")

    return parser.parse_args()


def load_scoring_results(file_path: str) -> list:
    """Load scoring results from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    results = []
    for item in data:
        result = ScoringResult(
            image_path=item['image_path'],
            category=item['category'],
            score=item['score'],
            raw_response=item['raw_response'],
            input_tokens=item['input_tokens'],
            output_tokens=item['output_tokens'],
            reasoning=item.get('reasoning')
        )
        results.append(result)

    return results


def main():
    """Main entry point for the summarization script."""
    args = parse_args()
    setup_logging(args.log_level)

    # Handle prompt listing
    if args.list_prompts:
        prompt_registry = PromptRegistry(args.prompt_dir)
        available_prompts = prompt_registry.list_available_prompts()
        print("Available prompts:")
        for prompt in available_prompts:
            print(f"  - {prompt}")
        return

    # Load scoring results
    logging.info(f"Loading scoring results from: {args.scoring_results}")
    scoring_results = load_scoring_results(args.scoring_results)
    logging.info(f"Loaded {len(scoring_results)} scoring results")

    # Initialize components
    llm = LlamaVisionRunner(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_batch_size=args.batch_size
    )

    prompt_registry = PromptRegistry(args.prompt_dir)

    summarizer = ImageSummarizer(
        llm=llm,
        prompt_registry=prompt_registry,
        output_dir=args.output_dir
    )

    # Run summarization
    results = summarizer.summarize_images(
        scoring_results=scoring_results,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        score_threshold=args.score_threshold,
        categories_to_process=args.categories_to_process
    )

    logging.info(f"Summarization complete. Processed {len(results)} images.")
    logging.info(f"Results saved to: {args.output_dir}")


def cli_main():
    """Entry point for console script."""
    main()


if __name__ == "__main__":
    main()
