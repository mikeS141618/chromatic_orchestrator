#!/usr/bin/env python3
# src/chromatic/cli/classify.py

import argparse
import logging
import sys
from pathlib import Path

from chromatic.vision.llama_vision_runner import LlamaVisionRunner
from chromatic.vision.classifier import ImageClassifier
from chromatic.utils.prompt_registry import PromptRegistry
from chromatic.utils.image_loader import load_images_from_directory, validate_image_paths


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
        description="Run image classification stage"
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
    parser.add_argument("--image-dir", type=str,
                        help="Directory containing images to process")
    parser.add_argument("--image-paths", type=str, nargs="+",
                        help="Individual image paths to process")
    parser.add_argument("--output-dir", type=str, default="./run/classification",
                        help="Output directory for results (default: ./run/classification)")
    parser.add_argument("--cache-dir", type=str, default="./cache/vision_responses",
                        help="Cache directory for LLM responses")
    parser.add_argument("--prompt-dir", type=str, default="./prompts",
                        help="Directory containing prompt files")

    # Image loading options
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively search subdirectories for images")
    parser.add_argument("--limit", type=int,
                        help="Maximum number of images to process")

    # Classification options
    parser.add_argument("--classification-prompt", type=str, default="classification_default",
                        help="Name of classification prompt to use")
    parser.add_argument("--max-tokens", type=int,
                        help="Max tokens for classification")
    parser.add_argument("--temperature", type=float,
                        help="Temperature for classification")

    # Utility options
    parser.add_argument("--list-prompts", action="store_true",
                        help="List available prompts and exit")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level (default: INFO)")

    return parser.parse_args()


def main():
    """Main entry point for the classification script."""
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

    # Initialize components
    llm = LlamaVisionRunner(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_batch_size=args.batch_size
    )

    prompt_registry = PromptRegistry(args.prompt_dir)

    classifier = ImageClassifier(
        llm=llm,
        prompt_registry=prompt_registry,
        output_dir=args.output_dir
    )

    # Run classification
    results = classifier.classify_images(
        image_paths=image_paths,
        classification_prompt=args.classification_prompt,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    logging.info(f"Classification complete. Processed {len(results)} images.")
    logging.info(f"Results saved to: {args.output_dir}")


def cli_main():
    """Entry point for console script."""
    main()


if __name__ == "__main__":
    main()
