# src/chromatic/vision/summarizer.py

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass

from chromatic.vision.llama_vision_runner import LlamaVisionRunner, VisionResponse
from chromatic.vision.classifier import ClassificationResult
from chromatic.vision.scorer import ScoringResult
from chromatic.utils.prompt_registry import PromptRegistry


@dataclass
class SummaryResult:
    """Container for summary results."""
    image_path: str
    category: str
    score: float
    summary: str
    raw_response: str
    input_tokens: int
    output_tokens: int


class ImageSummarizer:
    """Stage 3: Generate summaries for images based on category and score."""

    def __init__(
        self,
        llm: LlamaVisionRunner,
        prompt_registry: PromptRegistry,
        output_dir: str = "./run/summaries"
    ):
        """Initialize the summarizer."""
        self.llm = llm
        self.prompt_registry = prompt_registry
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def summarize_images(
        self,
        scoring_results: List[ScoringResult],
        batch_size: int = 8,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        score_threshold: Optional[float] = None,
        categories_to_process: Optional[List[str]] = None
    ) -> List[SummaryResult]:
        """Generate summaries for scored images.

        Args:
            scoring_results: Results from scoring stage
            batch_size: Number of images to process in each batch
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            score_threshold: Minimum score to process (None = all)
            categories_to_process: Optional list of categories to process (None = all)

        Returns:
            List of SummaryResult objects
        """
        # Filter by score threshold if specified
        if score_threshold is not None:
            filtered_results = [r for r in scoring_results if r.score >= score_threshold]
            logging.info(f"Processing {len(filtered_results)} images with score >= {score_threshold}")
        else:
            filtered_results = scoring_results
            logging.info(f"Processing all {len(filtered_results)} scored images")

        # Filter by categories if specified
        if categories_to_process:
            filtered_results = [r for r in filtered_results 
                              if r.category in categories_to_process]
            logging.info(f"Processing {len(filtered_results)} images from categories: {categories_to_process}")

        # Group by category
        results_by_category = self._group_by_category(filtered_results)

        all_summary_results = []

        # Process each category
        for category, category_results in results_by_category.items():
            logging.info(f"Summarizing {len(category_results)} images in category {category}")

            # Get category-specific summary prompt
            prompt_name = f"summary_category_{category.lower()}"
            prompt_data = self.prompt_registry.get(prompt_name)

            if not prompt_data or not prompt_data.get("user"):
                logging.warning(f"Summary prompt '{prompt_name}' not found, skipping category {category}")
                continue

            system_prompt = prompt_data.get("system")
            user_prompt_template = prompt_data.get("user")

            # Process in batches
            for i in tqdm(range(0, len(category_results), batch_size), 
                         desc=f"Summarizing category {category}"):
                batch_results = category_results[i:i + batch_size]

                # Create image-prompt pairs with score context
                image_prompts = []
                for r in batch_results:
                    # Format the prompt with score information
                    formatted_prompt = user_prompt_template.format(
                        score=r.score,
                        category=r.category
                    )
                    image_prompts.append((r.image_path, formatted_prompt))

                # Run batch summarization
                responses = self.llm.generate_batch(
                    image_prompts=image_prompts,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                # Process responses
                for scoring_result, response in zip(batch_results, responses):
                    if response:
                        summary_result = SummaryResult(
                            image_path=scoring_result.image_path,
                            category=category,
                            score=scoring_result.score,
                            summary=response.text.strip(),
                            raw_response=response.text,
                            input_tokens=response.input_tokens,
                            output_tokens=response.output_tokens
                        )
                        all_summary_results.append(summary_result)
                    else:
                        logging.error(f"No response for image: {scoring_result.image_path}")

        # Save results
        self._save_results(all_summary_results)

        # Log summary statistics
        self._log_summary_statistics(all_summary_results)

        return all_summary_results

    def _group_by_category(self, results: List[ScoringResult]) -> Dict[str, List[ScoringResult]]:
        """Group scoring results by category."""
        grouped = {}
        for result in results:
            category = result.category
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(result)
        return grouped

    def _save_results(self, results: List[SummaryResult]):
        """Save summary results to JSON."""
        output_file = self.output_dir / "summary_results.json"

        data = []
        for result in results:
            data.append({
                "image_path": result.image_path,
                "category": result.category,
                "score": result.score,
                "summary": result.summary,
                "raw_response": result.raw_response,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens
            })

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logging.info(f"Saved {len(results)} summary results to {output_file}")

    def _log_summary_statistics(self, results: List[SummaryResult]):
        """Log statistics about the summaries."""
        if not results:
            return

        # Token statistics
        total_input_tokens = sum(r.input_tokens for r in results)
        total_output_tokens = sum(r.output_tokens for r in results)
        avg_summary_length = sum(len(r.summary) for r in results) / len(results)

        logging.info(f"Summary statistics:")
        logging.info(f"  Total images summarized: {len(results)}")
        logging.info(f"  Total input tokens: {total_input_tokens}")
        logging.info(f"  Total output tokens: {total_output_tokens}")
        logging.info(f"  Average summary length: {avg_summary_length:.0f} characters")

        # Per-category counts
        category_counts = {}
        for result in results:
            category_counts[result.category] = category_counts.get(result.category, 0) + 1

        logging.info("Per-category summary counts:")
        for category, count in sorted(category_counts.items()):
            logging.info(f"  {category}: {count}")
