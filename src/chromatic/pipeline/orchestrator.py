# src/chromatic/pipeline/orchestrator.py

import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from multiprocessing import Process, Queue
import time

from chromatic.vision.llama_vision_runner import LlamaVisionRunner
from chromatic.vision.classifier import ImageClassifier, ClassificationResult
from chromatic.vision.scorer import ImageScorer, ScoringResult
from chromatic.vision.summarizer import ImageSummarizer, SummaryResult
from chromatic.utils.prompt_registry import PromptRegistry


@dataclass
class PipelineResult:
    """Container for complete pipeline results."""
    image_path: str
    classification: Optional[ClassificationResult] = None
    scoring: Optional[ScoringResult] = None
    summary: Optional[SummaryResult] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {"image_path": self.image_path}

        if self.classification:
            result["classification"] = {
                "category": self.classification.category,
                "raw_response": self.classification.raw_response,
                "input_tokens": self.classification.input_tokens,
                "output_tokens": self.classification.output_tokens
            }

        if self.scoring:
            result["scoring"] = {
                "score": self.scoring.score,
                "reasoning": self.scoring.reasoning,
                "raw_response": self.scoring.raw_response,
                "input_tokens": self.scoring.input_tokens,
                "output_tokens": self.scoring.output_tokens
            }

        if self.summary:
            result["summary"] = {
                "summary": self.summary.summary,
                "raw_response": self.summary.raw_response,
                "input_tokens": self.summary.input_tokens,
                "output_tokens": self.summary.output_tokens
            }

        return result


def worker_process(
        rank_id: int,
        gpu_ids: List[int],
        image_paths: List[str],
        model_path: str,
        prompt_dir: str,
        cache_dir: str,
        output_dir: str,
        tensor_parallel_size: int,
        max_model_len: int,
        batch_size: int,
        classification_prompt: str = "classification_default",
        classification_max_tokens: Optional[int] = None,
        classification_temperature: Optional[float] = None,
        skip_classification: bool = False,
        skip_scoring: bool = False,
        skip_summarization: bool = False,
        score_threshold: Optional[float] = None,
        categories_to_score: Optional[List[str]] = None,
        categories_to_summarize: Optional[List[str]] = None,
        result_queue: Optional[Queue] = None
):
    """Worker process for data parallel execution."""

    try:
        logging.info(f"Rank {rank_id}: Starting with {len(image_paths)} images on GPUs {gpu_ids}")

        # Initialize LLM for this rank with specific GPUs
        llm = LlamaVisionRunner(
            model_path=model_path,
            cache_dir=cache_dir,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            max_batch_size=batch_size,
            gpu_ids=gpu_ids,
            rank_id=rank_id
        )

        # Initialize prompt registry
        prompt_registry = PromptRegistry(prompt_dir)

        # Initialize pipeline stages with rank-specific output directories
        rank_output_dir = Path(output_dir) / f"rank_{rank_id}"

        classifier = ImageClassifier(
            llm=llm,
            prompt_registry=prompt_registry,
            output_dir=str(rank_output_dir / "classification")
        )

        scorer = ImageScorer(
            llm=llm,
            prompt_registry=prompt_registry,
            output_dir=str(rank_output_dir / "scoring")
        )

        summarizer = ImageSummarizer(
            llm=llm,
            prompt_registry=prompt_registry,
            output_dir=str(rank_output_dir / "summaries")
        )

        # Initialize results
        pipeline_results = {path: PipelineResult(image_path=path) for path in image_paths}

        # Stage 1: Classification
        if not skip_classification and image_paths:
            logging.info(f"Rank {rank_id}: Stage 1 - Classification")
            classification_results = classifier.classify_images(
                image_paths=image_paths,
                classification_prompt=classification_prompt,
                batch_size=batch_size,
                max_tokens=classification_max_tokens,
                temperature=classification_temperature
            )

            # Map classification results
            for result in classification_results:
                pipeline_results[result.image_path].classification = result
        else:
            logging.info(f"Rank {rank_id}: Skipping classification stage")
            # Create dummy classification results for testing
            classification_results = [
                ClassificationResult(
                    image_path=path,
                    category="A",
                    raw_response="Skipped",
                    input_tokens=0,
                    output_tokens=0
                ) for path in image_paths
            ]

        # Stage 2: Scoring
        if not skip_scoring and classification_results:
            logging.info(f"Rank {rank_id}: Stage 2 - Scoring")
            scoring_results = scorer.score_images(
                classification_results=classification_results,
                batch_size=batch_size,
                categories_to_process=categories_to_score
            )

            # Map scoring results
            for result in scoring_results:
                pipeline_results[result.image_path].scoring = result
        else:
            logging.info(f"Rank {rank_id}: Skipping scoring stage")
            # Create dummy scoring results
            scoring_results = [
                ScoringResult(
                    image_path=r.image_path,
                    category=r.category,
                    score=5.0,
                    raw_response="Skipped",
                    input_tokens=0,
                    output_tokens=0
                ) for r in classification_results
            ]

        # Stage 3: Summarization
        if not skip_summarization and scoring_results:
            logging.info(f"Rank {rank_id}: Stage 3 - Summarization")
            summary_results = summarizer.summarize_images(
                scoring_results=scoring_results,
                batch_size=batch_size,
                score_threshold=score_threshold,
                categories_to_process=categories_to_summarize
            )

            # Map summary results
            for result in summary_results:
                pipeline_results[result.image_path].summary = result
        else:
            logging.info(f"Rank {rank_id}: Skipping summarization stage")

        # Convert to list and save rank-specific results
        final_results = list(pipeline_results.values())

        # Save rank-specific results
        rank_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = rank_output_dir / "pipeline_results.json"
        data = [result.to_dict() for result in final_results]
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logging.info(f"Rank {rank_id}: Saved {len(final_results)} results to {output_file}")

        # Send results back through queue if provided
        if result_queue:
            result_queue.put((rank_id, final_results))

        return final_results

    except Exception as e:
        logging.error(f"Rank {rank_id}: Exception occurred: {e}")
        if result_queue:
            result_queue.put((rank_id, None))
        raise


class ChromaticOrchestrator:
    """Main pipeline orchestrator for the chromatic vision classification system.

    This class implements the complete 3-stage pipeline with proper data parallel support:
    1. Classification: Categorize images into dynamic categories
    2. Scoring: Score images based on their category (0-10)
    3. Summarization: Generate summaries based on category and score

    The pipeline uses massive cross-image batching for efficiency,
    with proper data parallel coordination using independent GPU-isolated processes.
    """

    def __init__(
            self,
            model_path: str,
            prompt_dir: str = "./prompts",
            cache_dir: str = "./cache/vision_responses",
            output_dir: str = "./run",
            tensor_parallel_size: int = 2,
            max_model_len: int = 8192,
            batch_size: int = 8,
            # Data parallel parameters
            dp_size: int = 1,
            available_gpus: Optional[List[int]] = None
    ):
        """Initialize the orchestrator with model and configuration."""

        self.model_path = model_path
        self.prompt_dir = prompt_dir
        self.cache_dir = cache_dir
        self.output_dir = Path(output_dir)
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.batch_size = batch_size

        # Data parallel configuration
        self.dp_size = dp_size

        # GPU assignment for data parallel
        if available_gpus is None:
            # Auto-detect available GPUs
            import torch
            self.available_gpus = list(range(torch.cuda.device_count()))
        else:
            self.available_gpus = available_gpus

        # Validate GPU configuration
        total_gpus_needed = dp_size * tensor_parallel_size
        if len(self.available_gpus) < total_gpus_needed:
            raise ValueError(f"Need {total_gpus_needed} GPUs but only {len(self.available_gpus)} available")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"ChromaticOrchestrator initialized with model: {model_path}")
        logging.info(f"Output directory: {output_dir}")
        if dp_size > 1:
            logging.info(f"Data parallel configuration: {dp_size} ranks, {tensor_parallel_size} GPUs per rank")
            logging.info(f"Available GPUs: {self.available_gpus}")

    def _split_images_for_dp(self, image_paths: List[str]) -> List[List[str]]:
        """Split image paths evenly across DP ranks."""
        if self.dp_size == 1:
            return [image_paths]

        # Distribute images evenly across DP ranks
        images_per_rank = len(image_paths) // self.dp_size
        remainder = len(image_paths) % self.dp_size

        rank_images = []
        start_idx = 0

        for rank in range(self.dp_size):
            # Give remainder images to first few ranks
            rank_size = images_per_rank + (1 if rank < remainder else 0)
            end_idx = start_idx + rank_size

            rank_images.append(image_paths[start_idx:end_idx])
            start_idx = end_idx

        return rank_images

    def _assign_gpus_to_ranks(self) -> List[List[int]]:
        """Assign GPU IDs to each DP rank."""
        gpu_assignments = []

        for rank in range(self.dp_size):
            start_gpu = rank * self.tensor_parallel_size
            end_gpu = start_gpu + self.tensor_parallel_size
            rank_gpus = self.available_gpus[start_gpu:end_gpu]
            gpu_assignments.append(rank_gpus)

        return gpu_assignments

    def run_pipeline(
            self,
            image_paths: List[str],
            classification_prompt: str = "classification_default",
            classification_max_tokens: Optional[int] = None,
            classification_temperature: Optional[float] = None,
            skip_classification: bool = False,
            skip_scoring: bool = False,
            skip_summarization: bool = False,
            score_threshold: Optional[float] = None,
            categories_to_score: Optional[List[str]] = None,
            categories_to_summarize: Optional[List[str]] = None
    ) -> List[PipelineResult]:
        """Run the complete vision classification pipeline with data parallel support.

        Args:
            image_paths: List of paths to images to process
            classification_prompt: Name of the classification prompt
            classification_max_tokens: Max tokens for classification
            classification_temperature: Temperature for classification
            skip_classification: Skip classification stage
            skip_scoring: Skip scoring stage
            skip_summarization: Skip summarization stage
            score_threshold: Minimum score to proceed to summarization
            categories_to_score: Only score these categories
            categories_to_summarize: Only summarize these categories

        Returns:
            List of PipelineResult objects with complete pipeline results
        """
        logging.info(f"Starting pipeline for {len(image_paths)} images")

        if self.dp_size == 1:
            # Single process execution
            return worker_process(
                rank_id=0,
                gpu_ids=self.available_gpus[:self.tensor_parallel_size],
                image_paths=image_paths,
                model_path=self.model_path,
                prompt_dir=self.prompt_dir,
                cache_dir=self.cache_dir,
                output_dir=str(self.output_dir),
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len,
                batch_size=self.batch_size,
                classification_prompt=classification_prompt,
                classification_max_tokens=classification_max_tokens,
                classification_temperature=classification_temperature,
                skip_classification=skip_classification,
                skip_scoring=skip_scoring,
                skip_summarization=skip_summarization,
                score_threshold=score_threshold,
                categories_to_score=categories_to_score,
                categories_to_summarize=categories_to_summarize
            )
        else:
            # Multi-process data parallel execution
            return self._run_data_parallel(
                image_paths=image_paths,
                classification_prompt=classification_prompt,
                classification_max_tokens=classification_max_tokens,
                classification_temperature=classification_temperature,
                skip_classification=skip_classification,
                skip_scoring=skip_scoring,
                skip_summarization=skip_summarization,
                score_threshold=score_threshold,
                categories_to_score=categories_to_score,
                categories_to_summarize=categories_to_summarize
            )

    def _run_data_parallel(
            self,
            image_paths: List[str],
            classification_prompt: str = "classification_default",
            classification_max_tokens: Optional[int] = None,
            classification_temperature: Optional[float] = None,
            skip_classification: bool = False,
            skip_scoring: bool = False,
            skip_summarization: bool = False,
            score_threshold: Optional[float] = None,
            categories_to_score: Optional[List[str]] = None,
            categories_to_summarize: Optional[List[str]] = None
    ) -> List[PipelineResult]:
        """Run pipeline with data parallel processing."""

        # Split images across ranks
        rank_image_lists = self._split_images_for_dp(image_paths)

        # Assign GPUs to ranks
        gpu_assignments = self._assign_gpus_to_ranks()

        # Create result queue for collecting results
        result_queue = Queue()

        # Start processes for each DP rank
        processes = []
        for rank_id in range(self.dp_size):
            rank_images = rank_image_lists[rank_id]
            rank_gpus = gpu_assignments[rank_id]

            logging.info(f"Starting rank {rank_id} with {len(rank_images)} images on GPUs {rank_gpus}")

            proc = Process(
                target=worker_process,
                args=(
                    rank_id,
                    rank_gpus,
                    rank_images,
                    self.model_path,
                    self.prompt_dir,
                    self.cache_dir,
                    str(self.output_dir),
                    self.tensor_parallel_size,
                    self.max_model_len,
                    self.batch_size,
                    classification_prompt,
                    classification_max_tokens,
                    classification_temperature,
                    skip_classification,
                    skip_scoring,
                    skip_summarization,
                    score_threshold,
                    categories_to_score,
                    categories_to_summarize,
                    result_queue
                )
            )
            proc.start()
            processes.append((proc, rank_id))

        # Wait for all processes to complete and collect results
        all_results = []
        completed_ranks = 0

        while completed_ranks < self.dp_size:
            try:
                rank_id, rank_results = result_queue.get(timeout=3600)  # 1 hour timeout
                if rank_results is not None:
                    all_results.extend(rank_results)
                    logging.info(f"Collected results from rank {rank_id}: {len(rank_results)} images")
                else:
                    logging.error(f"Rank {rank_id} returned no results")
                completed_ranks += 1
            except Exception as e:
                logging.error(f"Error collecting results: {e}")
                break

        # Wait for all processes to finish
        exit_code = 0
        for proc, rank_id in processes:
            proc.join(timeout=60)  # Short timeout since work should be done
            if proc.exitcode is None:
                logging.error(f"Killing rank {rank_id} process that didn't complete")
                proc.kill()
                exit_code = 1
            elif proc.exitcode and proc.exitcode != 0:
                logging.error(f"Rank {rank_id} exited with code {proc.exitcode}")
                exit_code = proc.exitcode

        if exit_code != 0:
            raise RuntimeError(f"Data parallel execution failed with exit code {exit_code}")

        # Save merged results
        self._save_pipeline_results(all_results)
        self._log_pipeline_statistics(all_results)

        return all_results

    def _save_pipeline_results(self, results: List[PipelineResult]):
        """Save complete pipeline results to JSON."""
        output_file = self.output_dir / "pipeline_results.json"

        data = [result.to_dict() for result in results]

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logging.info(f"Saved {len(results)} merged pipeline results to {output_file}")

    def _log_pipeline_statistics(self, results: List[PipelineResult]):
        """Log statistics about the pipeline run."""
        total_images = len(results)
        classified = sum(1 for r in results if r.classification)
        scored = sum(1 for r in results if r.scoring)
        summarized = sum(1 for r in results if r.summary)

        logging.info("Final pipeline statistics:")
        logging.info(f"  Total images: {total_images}")
        logging.info(f"  Classified: {classified}")
        logging.info(f"  Scored: {scored}")
        logging.info(f"  Summarized: {summarized}")

        # Token usage
        total_tokens = 0
        for result in results:
            if result.classification:
                total_tokens += result.classification.input_tokens + result.classification.output_tokens
            if result.scoring:
                total_tokens += result.scoring.input_tokens + result.scoring.output_tokens
            if result.summary:
                total_tokens += result.summary.input_tokens + result.summary.output_tokens

        logging.info(f"  Total tokens used: {total_tokens}")