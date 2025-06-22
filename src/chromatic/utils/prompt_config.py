# src/chromatic/utils/prompt_config.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptConfig:
    """Configuration for a single prompt in the pipeline."""
    name: str
    max_tokens: int
    output_suffix: Optional[str] = None
    temperature: float = 0.0

    def __post_init__(self):
        """Set default output suffix if not provided."""
        if self.output_suffix is None:
            parts = self.name.split('_')
            if len(parts) >= 2:
                self.output_suffix = parts[1]
            else:
                self.output_suffix = self.name
