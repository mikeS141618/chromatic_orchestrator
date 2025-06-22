# src/chromatic/vision/llama_vision_runner.py

import hashlib
import json
import os
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, NamedTuple
from PIL import Image

from vllm import LLM, EngineArgs, SamplingParams
from transformers import AutoTokenizer


class VisionResponse(NamedTuple):
    """Container for vision model response with token counts and metadata."""
    text: str
    input_tokens: int
    output_tokens: int
    image_path: str
    user_prompt: str
    system_prompt: Optional[str] = None
    category: Optional[str] = None
    score: Optional[float] = None


class LlamaVisionRunner:
    """Runner for vision-enabled LLaMA models using vLLM with caching support."""

    def __init__(self, model_path: str, cache_dir: Optional[str] = None,
                 tensor_parallel_size: int = 2, max_tokens: int = 1024,
                 max_model_len: int = 8192, max_batch_size: int = 8,
                 use_cache: bool = True, temperature: float = 0.0,
                 top_p: float = 1.0, top_k: int = -1,
                 limit_mm_per_prompt: int = 1,
                 gpu_ids: Optional[List[int]] = None,
                 rank_id: Optional[int] = None):
        """Initialize the vision runner with model and caching configuration."""

        self.model_path = model_path
        self.use_cache = use_cache
        self.rank_id = rank_id or 0
        self.gpu_ids = gpu_ids or list(range(tensor_parallel_size))

        # Set up cache directory with rank isolation
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/vision_responses")
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set CUDA_VISIBLE_DEVICES to isolate GPUs for this rank
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            tensor_parallel_size = len(gpu_ids)

        # Initialize tokenizer for chat template formatting
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Initialize engine args with vision support
        engine_args = EngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_seq_len_to_capture=max_tokens,
            max_num_seqs=max_batch_size,
            max_model_len=max_model_len,
            enforce_eager=True,
            seed=42,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": limit_mm_per_prompt}
        )
        #, disable_custom_all_reduce=True

        # Initialize default sampling params
        self.default_sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            min_tokens=10,
            seed=42
        )

        # Initialize LLM engine
        self.llm = LLM(**vars(engine_args))

        logging.info(f"LlamaVisionRunner initialized with model: {model_path}")
        logging.info(f"Tensor parallel size: {tensor_parallel_size}")
        logging.info(f"GPU IDs: {gpu_ids}")
        if rank_id is not None:
            logging.info(f"Rank ID: {rank_id}")

    def _format_prompt_with_chat_template(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format the prompt using the model's chat template."""

        # Build messages list - try without system first for vision models
        messages = []

        # For vision models, we need to structure the user content properly
        user_content = [
            {"type": "image"},  # This tells the model where to place the image
            {"type": "text", "text": user_prompt}
        ]

        # If we have a system prompt, incorporate it into the user message
        if system_prompt:
            combined_text = f"{system_prompt}\n\n{user_prompt}"
            user_content = [
                {"type": "image"},
                {"type": "text", "text": combined_text}
            ]

        messages.append({
            "role": "user",
            "content": user_content
        })

        try:
            # Use the tokenizer's chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            return formatted_prompt
        except Exception as e:
            logging.warning(f"Failed to use chat template: {e}")
            # Fallback to simple format if chat template fails
            return self._fallback_format(user_prompt, system_prompt)

    def _fallback_format(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Fallback prompt format for models without proper chat templates."""
        if system_prompt:
            return f"{system_prompt}\n\nUser: <image>\n{user_prompt}\n\nAssistant:"
        else:
            return f"User: <image>\n{user_prompt}\n\nAssistant:"

    def _get_cache_key(self, image_path: str, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a deterministic hash key for caching based on image and prompt."""
        image_stat = os.stat(image_path)
        image_info = f"{image_path}:{image_stat.st_mtime}:{image_stat.st_size}"

        content_to_hash = f"{image_info}|||{prompt}"
        if system_prompt:
            content_to_hash = f"{system_prompt}|||{content_to_hash}"

        key = hashlib.sha256(content_to_hash.encode('utf-8')).hexdigest()
        return key

    def _get_cached_response(self, cache_key: str) -> Optional[VisionResponse]:
        """Retrieve cached response if it exists."""
        if not self.use_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logging.debug(f"Cache hit for key: {cache_key}")
                return VisionResponse(
                    text=data['text'],
                    input_tokens=data['input_tokens'],
                    output_tokens=data['output_tokens'],
                    image_path=data['image_path'],
                    user_prompt=data['user_prompt'],
                    system_prompt=data.get('system_prompt'),
                    category=data.get('category'),
                    score=data.get('score')
                )
        return None

    def _cache_response(self, cache_key: str, response: VisionResponse) -> None:
        """Save response to cache."""
        if not self.use_cache:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        data = {
            'text': response.text,
            'input_tokens': response.input_tokens,
            'output_tokens': response.output_tokens,
            'image_path': response.image_path,
            'user_prompt': response.user_prompt,
            'system_prompt': response.system_prompt,
            'category': response.category,
            'score': response.score
        }

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logging.debug(f"Cached response for key: {cache_key}")

    def generate(self, image_path: str, prompt: str, system_prompt: Optional[str] = None,
                 max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> VisionResponse:
        """Generate response for a single image with prompt."""

        # Try to get from cache first
        cache_key = self._get_cache_key(image_path, prompt, system_prompt)
        cached_response = self._get_cached_response(cache_key)
        if cached_response is not None:
            return cached_response

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Format prompt using chat template
        formatted_prompt = self._format_prompt_with_chat_template(prompt, system_prompt)

        # Create sampling params with any overrides
        sampling_params = SamplingParams(
            temperature=temperature if temperature is not None else self.default_sampling_params.temperature,
            top_p=self.default_sampling_params.top_p,
            top_k=self.default_sampling_params.top_k,
            max_tokens=max_tokens if max_tokens is not None else self.default_sampling_params.max_tokens,
            min_tokens=self.default_sampling_params.min_tokens,
        )

        # Run generation with image
        inputs = {
            "prompt": formatted_prompt,
            "multi_modal_data": {"image": image}
        }

        outputs = self.llm.generate([inputs], sampling_params)
        response_text = outputs[0].outputs[0].text
        input_tokens = len(outputs[0].prompt_token_ids) if outputs[0].prompt_token_ids is not None else 0
        output_tokens = len(outputs[0].outputs[0].token_ids) if outputs[0].outputs[0].token_ids is not None else 0

        # Create response object
        response = VisionResponse(
            text=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            image_path=image_path,
            user_prompt=prompt,
            system_prompt=system_prompt
        )

        # Cache the response
        self._cache_response(cache_key, response)

        return response

    def generate_batch(self, image_prompts: List[Tuple[str, str]],
                       system_prompt: Optional[str] = None,
                       max_tokens: Optional[int] = None,
                       temperature: Optional[float] = None) -> List[VisionResponse]:
        """Generate responses for a batch of image-prompt pairs."""

        # Find which prompts are cached and which need generation
        uncached_inputs = []
        uncached_indices = []
        results = [None] * len(image_prompts)

        # Check cache for each image-prompt pair
        for i, (image_path, prompt) in enumerate(image_prompts):
            cache_key = self._get_cache_key(image_path, prompt, system_prompt)
            cached_response = self._get_cached_response(cache_key)

            if cached_response is not None:
                results[i] = cached_response
            else:
                uncached_inputs.append((image_path, prompt))
                uncached_indices.append(i)

        # If all were cached, return early
        if not uncached_inputs:
            return results

        # Create sampling params
        sampling_params = SamplingParams(
            temperature=temperature if temperature is not None else self.default_sampling_params.temperature,
            top_p=self.default_sampling_params.top_p,
            top_k=self.default_sampling_params.top_k,
            max_tokens=max_tokens if max_tokens is not None else self.default_sampling_params.max_tokens,
            min_tokens=self.default_sampling_params.min_tokens,
        )

        # Prepare batch inputs for vLLM
        batch_inputs = []
        for image_path, prompt in uncached_inputs:
            image = Image.open(image_path).convert("RGB")
            formatted_prompt = self._format_prompt_with_chat_template(prompt, system_prompt)
            batch_inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": image}
            })

        # Run batch generation
        outputs = self.llm.generate(batch_inputs, sampling_params)

        # Process outputs and cache
        new_responses = []
        for i, (output, (image_path, prompt)) in enumerate(zip(outputs, uncached_inputs)):
            response_text = output.outputs[0].text
            input_tokens = len(output.prompt_token_ids) if output.prompt_token_ids is not None else 0
            output_tokens = len(output.outputs[0].token_ids) if output.outputs[0].token_ids is not None else 0

            response = VisionResponse(
                text=response_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                image_path=image_path,
                user_prompt=prompt,
                system_prompt=system_prompt
            )

            new_responses.append(response)

            # Cache the response
            cache_key = self._get_cache_key(image_path, prompt, system_prompt)
            self._cache_response(cache_key, response)

        # Fill in results
        for i, response in zip(uncached_indices, new_responses):
            results[i] = response

        return results

    def extract_category(self, response_text: str) -> Optional[str]:
        """Extract category with refusal detection FIRST."""

        # CRITICAL: Check for refusal patterns BEFORE any regex extraction
        refusal_patterns = [
            r"I can't help with that",
            r"I cannot analyze this image",
            r"I'm not able to classify",
            r"I cannot provide",
            r"I'm unable to",
            r"I apologize, but I cannot"
        ]

        for pattern in refusal_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                logging.warning(f"Model refused classification: {response_text.strip()}")
                return "REFUSED"

        # Now safe to do category extraction...
        # Primary: "CATEGORY: A"
        match = re.search(r'CATEGORY:\s*([A-J])', response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Secondary: "Category A:"
        match = re.search(r'Category\s+([A-J]):', response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # More conservative fallback
        match = re.search(r'(?:classified as|category is|falls under)\s+([A-J])', response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        return None

    def extract_score(self, response_text: str) -> Optional[float]:
        """Extract numerical score from scoring response."""
        # Look for patterns like "SCORE: 8" or "Score: 7.5"
        match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
        if match:
            return float(match.group(1))

        # Also try to find score in other formats
        match = re.search(r'rating.*?(\d+(?:\.\d+)?)\s*(?:/\s*10)?', response_text, re.IGNORECASE)
        if match:
            return float(match.group(1))

        return None