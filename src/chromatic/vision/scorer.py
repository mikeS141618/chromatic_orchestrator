# src/chromatic/vision/scorer.py

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass

from chromatic.vision.llama_vision_runner import LlamaVisionRunner, VisionResponse
from chromatic.vision.classifier import ClassificationResult
from chromatic.utils.prompt_registry import PromptRegistry


@dataclass
class ScoringResult:
    """Container for scoring results."""
    image_path: str
    category: str
    score: float
    raw_response: str
    input_tokens: int
    output_tokens: int
    reasoning: Optional[str] = None


class ImageScorer:
    """Stage 2: Score images based on their category using vision LLM."""

    def __init__(
        self,
        llm: LlamaVisionRunner,
        prompt_registry: PromptRegistry,
        output_dir: str = "./run/scoring"
    ):
        """Initialize the scorer."""
        self.llm = llm
        self.prompt_registry = prompt_registry
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def score_images(
        self,
        classification_results: List[ClassificationResult],
        batch_size: int = 8,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        categories_to_process: Optional[List[str]] = None
    ) -> List[ScoringResult]:
        """Score images based on their categories.

        Args:
            classification_results: Results from classification stage
            batch_size: Number of images to process in each batch
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            categories_to_process: Optional list of categories to process (None = all)

        Returns:
            List of ScoringResult objects
        """
        # Filter by categories if specified
        if categories_to_process:
            filtered_results = [r for r in classification_results 
                              if r.category in categories_to_process]
            logging.info(f"Processing {len(filtered_results)} images from categories: {categories_to_process}")
        else:
            filtered_results = classification_results
            logging.info(f"Processing all {len(filtered_results)} classified images")

        # Group by category
        results_by_category = self._group_by_category(filtered_results)

        all_scoring_results = []

        # Process each category
        for category, category_results in results_by_category.items():
            logging.info(f"Scoring {len(category_results)} images in category {category}")

            # Get category-specific scoring prompt
            prompt_name = f"scoring_category_{category.lower()}"
            prompt_data = self.prompt_registry.get(prompt_name)

            if not prompt_data or not prompt_data.get("user"):
                logging.warning(f"Scoring prompt '{prompt_name}' not found, skipping category {category}")
                continue

            system_prompt = prompt_data.get("system")
            user_prompt = prompt_data.get("user")

            # Process in batches
            for i in tqdm(range(0, len(category_results), batch_size), 
                         desc=f"Scoring category {category}"):
                batch_results = category_results[i:i + batch_size]

                # Create image-prompt pairs
                image_prompts = [(r.image_path, user_prompt) for r in batch_results]

                # Run batch scoring
                responses = self.llm.generate_batch(
                    image_prompts=image_prompts,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                # Process responses
                for classification_result, response in zip(batch_results, responses):
                    if response:
                        score = self.llm.extract_score(response.text)
                        if score is None:
                            logging.warning(f"Could not extract score for {classification_result.image_path}")
                            score = -1.0

                        # Extract reasoning if present
                        reasoning = self._extract_reasoning(response.text)

                        scoring_result = ScoringResult(
                            image_path=classification_result.image_path,
                            category=category,
                            score=score,
                            raw_response=response.text,
                            input_tokens=response.input_tokens,
                            output_tokens=response.output_tokens,
                            reasoning=reasoning
                        )
                        all_scoring_results.append(scoring_result)
                    else:
                        logging.error(f"No response for image: {classification_result.image_path}")

        # Save results
        self._save_results(all_scoring_results)

        # Log score statistics
        self._log_score_statistics(all_scoring_results)

        return all_scoring_results

    def _group_by_category(self, results: List[ClassificationResult]) -> Dict[str, List[ClassificationResult]]:
        """Group classification results by category."""
        grouped = {}
        for result in results:
            category = result.category
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(result)
        return grouped

    def _extract_reasoning(self, response_text: str) -> Optional[str]:
        """Extract reasoning from the response."""
        import re

        # Look for reasoning section
        match = re.search(r'REASONING:\s*(.*?)(?=\n\n|\Z)', response_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Alternative patterns
        match = re.search(r'Explanation:\s*(.*?)(?=\n\n|\Z)', response_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return None

    def _save_results(self, results: List[ScoringResult]):
        """Save scoring results to JSON."""
        output_file = self.output_dir / "scoring_results.json"

        data = []
        for result in results:
            data.append({
                "image_path": result.image_path,
                "category": result.category,
                "score": result.score,
                "raw_response": result.raw_response,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "reasoning": result.reasoning
            })

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logging.info(f"Saved {len(results)} scoring results to {output_file}")

    def _log_score_statistics(self, results: List[ScoringResult]):
        """Log statistics about the scores."""
        if not results:
            return

        # Overall statistics
        scores = [r.score for r in results]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)

        logging.info(f"Overall score statistics:")
        logging.info(f"  Average: {avg_score:.2f}")
        logging.info(f"  Min: {min_score}")
        logging.info(f"  Max: {max_score}")

        # Per-category statistics
        category_scores = {}
        for result in results:
            if result.category not in category_scores:
                category_scores[result.category] = []
            category_scores[result.category].append(result.score)

        logging.info("Per-category score statistics:")
        for category, scores in sorted(category_scores.items()):
            avg = sum(scores) / len(scores)
            logging.info(f"  {category}: avg={avg:.2f}, n={len(scores)}")
