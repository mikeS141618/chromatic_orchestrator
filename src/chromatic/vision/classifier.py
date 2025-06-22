# src/chromatic/vision/classifier.py

import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass

from chromatic.vision.llama_vision_runner import LlamaVisionRunner, VisionResponse
from chromatic.utils.prompt_registry import PromptRegistry


@dataclass
class ClassificationResult:
    """Container for classification results."""
    image_path: str
    category: str
    raw_response: str
    input_tokens: int
    output_tokens: int
    confidence: Optional[float] = None


class ImageClassifier:
    """Stage 1: Classify images into categories using vision LLM."""

    def __init__(
        self,
        llm: LlamaVisionRunner,
        prompt_registry: PromptRegistry,
        output_dir: str = "./run/classification"
    ):
        """Initialize the classifier."""
        self.llm = llm
        self.prompt_registry = prompt_registry
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def classify_images(
        self,
        image_paths: List[str],
        classification_prompt: str = "classification_default",
        batch_size: int = 8,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[ClassificationResult]:
        """Classify a batch of images into categories.

        Args:
            image_paths: List of paths to images
            classification_prompt: Name of the classification prompt to use
            batch_size: Number of images to process in each batch
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            List of ClassificationResult objects
        """
        logging.info(f"Classifying {len(image_paths)} images with prompt: {classification_prompt}")

        # Get the classification prompt
        prompt_data = self.prompt_registry.get(classification_prompt)
        if not prompt_data or not prompt_data.get("user"):
            raise ValueError(f"Classification prompt '{classification_prompt}' not found")

        system_prompt = prompt_data.get("system")
        user_prompt = prompt_data.get("user")

        # Determine categories from prompt
        categories = self._extract_categories_from_prompt(user_prompt)
        logging.info(f"Detected categories from prompt: {categories}")

        results = []

        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Classifying images"):
            batch_paths = image_paths[i:i + batch_size]

            # Create image-prompt pairs
            image_prompts = [(path, user_prompt) for path in batch_paths]

            # Run batch classification
            responses = self.llm.generate_batch(
                image_prompts=image_prompts,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Process responses
            for path, response in zip(batch_paths, responses):
                if response:
                    category = self.llm.extract_category(response.text)
                    if not category:
                        logging.warning(f"Could not extract category for {path}")
                        logging.warning(f"response\n {response}")
                        category = "UNKNOWN"

                    result = ClassificationResult(
                        image_path=path,
                        category=category,
                        raw_response=response.text,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens
                    )
                    results.append(result)
                else:
                    logging.error(f"No response for image: {path}")

        # Save results
        self._save_results(results)

        # Log category distribution
        self._log_category_distribution(results)

        return results

    def _extract_categories_from_prompt(self, prompt_text: str) -> List[str]:
        """Extract possible categories from the prompt text."""
        import re

        # Look for patterns like "Category A:", "Category B:", etc.
        category_pattern = r'Category\s+([A-Z]):'
        matches = re.findall(category_pattern, prompt_text, re.IGNORECASE)

        if matches:
            return sorted(set(matches))

        # Fallback: look for any single capital letters followed by colon or parenthesis
        fallback_pattern = r'\b([A-Z])\s*[:\)]'
        matches = re.findall(fallback_pattern, prompt_text)

        if matches:
            return sorted(set(matches))

        # Default categories if none found
        return ["A", "B", "C", "D", "OTHER"]

    def _save_results(self, results: List[ClassificationResult]):
        """Save classification results to JSON."""
        output_file = self.output_dir / "classification_results.json"

        data = []
        for result in results:
            data.append({
                "image_path": result.image_path,
                "category": result.category,
                "raw_response": result.raw_response,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "confidence": result.confidence
            })

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logging.info(f"Saved {len(results)} classification results to {output_file}")

    def _log_category_distribution(self, results: List[ClassificationResult]):
        """Log the distribution of categories."""
        category_counts = {}
        for result in results:
            category = result.category
            category_counts[category] = category_counts.get(category, 0) + 1

        logging.info("Category distribution:")
        for category, count in sorted(category_counts.items()):
            percentage = (count / len(results)) * 100
            logging.info(f"  {category}: {count} ({percentage:.1f}%)")
