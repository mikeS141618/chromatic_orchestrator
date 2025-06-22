#!/usr/bin/env python3
# src/chromatic/cli/run_pipeline.py

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from chromatic.pipeline.orchestrator import ChromaticOrchestrator
from chromatic.utils.image_loader import load_images_from_directory, validate_image_paths
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
        description="Run the complete chromatic vision classification pipeline with data parallel support"
    )

    # Model configuration
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the vision LLM model directory")
    parser.add_argument("--tensor-parallel-size", type=int, default=2,
                        help="Number of GPUs for tensor parallelism per DP rank (default: 2)")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Maximum model context length (default: 8192)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for processing (default: 8)")

    # Data parallel configuration
    parser.add_argument("--dp-size", type=int, default=1,
                        help="Data parallel size (number of model instances, default: 1)")
    parser.add_argument("--available-gpus", type=int, nargs="+",
                        help="List of available GPU IDs (default: auto-detect)")

    # Input/output paths
    parser.add_argument("--image-dir", type=str,
                        help="Directory containing images to process")
    parser.add_argument("--image-paths", type=str, nargs="+",
                        help="Individual image paths to process")
    parser.add_argument("--output-dir", type=str, default="./run",
                        help="Output directory for results (default: ./run)")
    parser.add_argument("--cache-dir", type=str, default="./cache/vision_responses",
                        help="Cache directory for LLM responses")
    parser.add_argument("--prompt-dir", type=str, default="./prompts",
                        help="Directory containing prompt files")

    # Image loading options
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively search subdirectories for images")
    parser.add_argument("--limit", type=int,
                        help="Maximum number of images to process")
    parser.add_argument("--extensions", type=str, nargs="+",
                        default=[".jpg", ".jpeg", ".png", ".bmp", ".gif"],
                        help="Image file extensions to include")

    # Pipeline control
    parser.add_argument("--classification-prompt", type=str, default="classification_default",
                        help="Name of classification prompt to use")
    parser.add_argument("--classification-max-tokens", type=int,
                        help="Max tokens for classification")
    parser.add_argument("--classification-temperature", type=float,
                        help="Temperature for classification")

    parser.add_argument("--skip-classification", action="store_true",
                        help="Skip classification stage")
    parser.add_argument("--skip-scoring", action="store_true",
                        help="Skip scoring stage")
    parser.add_argument("--skip-summarization", action="store_true",
                        help="Skip summarization stage")

    parser.add_argument("--score-threshold", type=float,
                        help="Minimum score to proceed to summarization")
    parser.add_argument("--categories-to-score", type=str, nargs="+",
                        help="Only score these categories")
    parser.add_argument("--categories-to-summarize", type=str, nargs="+",
                        help="Only summarize these categories")

    # Utility options
    parser.add_argument("--list-prompts", action="store_true",
                        help="List available prompts and exit")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level (default: INFO)")
    parser.add_argument("--disable-vllm-progress", action="store_true", default=True,
                        help="Disable vLLM progress bars while keeping others (default: True)")

    return parser.parse_args()


def main():
    """Main entry point for the pipeline runner."""
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

    # Validate data parallel configuration
    if args.dp_size > 1:
        total_gpus = args.dp_size * args.tensor_parallel_size
        logging.info(f"Data parallel configuration:")
        logging.info(f"  DP size: {args.dp_size}")
        logging.info(f"  TP size per DP rank: {args.tensor_parallel_size}")
        logging.info(f"  Total GPUs required: {total_gpus}")

    # Validate required arguments
    if not args.image_dir and not args.image_paths:
        logging.error("Either --image-dir or --image-paths must be specified")
        sys.exit(1)

    # Collect image paths
    image_paths = []

    if args.image_dir:
        logging.info(f"Loading images from directory: {args.image_dir}")
        dir_images = load_images_from_directory(
            directory=args.image_dir,
            extensions=args.extensions,
            recursive=args.recursive,
            limit=args.limit
        )
        image_paths.extend(dir_images)

    if args.image_paths:
        image_paths.extend(args.image_paths)

    # Validate images
    image_paths = validate_image_paths(image_paths)

    if not image_paths:
        logging.error("No valid images found to process")
        sys.exit(1)

    logging.info(f"Processing {len(image_paths)} images")

    # Initialize orchestrator
    orchestrator = ChromaticOrchestrator(
        model_path=args.model_path,
        prompt_dir=args.prompt_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        dp_size=args.dp_size,
        available_gpus=args.available_gpus
    )

    # Run pipeline
    results = orchestrator.run_pipeline(
        image_paths=image_paths,
        classification_prompt=args.classification_prompt,
        classification_max_tokens=args.classification_max_tokens,
        classification_temperature=args.classification_temperature,
        skip_classification=args.skip_classification,
        skip_scoring=args.skip_scoring,
        skip_summarization=args.skip_summarization,
        score_threshold=args.score_threshold,
        categories_to_score=args.categories_to_score,
        categories_to_summarize=args.categories_to_summarize
    )

    logging.info(f"Pipeline complete. Processed {len(results)} images.")
    logging.info(f"Results saved to: {args.output_dir}")


def cli_main():
    """Entry point for console script."""
    main()


if __name__ == "__main__":
    main()