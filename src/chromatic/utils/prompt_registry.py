# src/chromatic/utils/prompt_registry.py
import os
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple, List


class PromptRegistry:
    """Registry that loads prompts from individual text files."""

    def __init__(self, prompt_dir: str = "./prompts"):
        """Initialize the registry with prompts from the specified directory."""
        self.prompt_dir = Path(prompt_dir)
        self.prompts = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load all prompt files from the prompt directory."""
        if not self.prompt_dir.exists():
            logging.warning(f"Prompt directory '{self.prompt_dir}' does not exist")
            return

        for file_path in self.prompt_dir.glob("*.txt"):
            prompt_name = file_path.stem
            system_prompt, user_prompt = self._parse_prompt_file(file_path)

            if user_prompt:
                self.prompts[prompt_name] = {
                    "system": system_prompt,
                    "user": user_prompt
                }
                logging.debug(f"Loaded prompt '{prompt_name}' from {file_path}")
            else:
                logging.warning(f"Failed to load prompt from {file_path}")

    def _parse_prompt_file(self, file_path: Path) -> Tuple[str, str]:
        """Parse a prompt file into system and user prompts."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            system_marker = "==== SYSTEM PROMPT ===="
            user_marker = "==== USER PROMPT ===="

            system_prompt = ""
            user_prompt = ""

            if system_marker in content and user_marker in content:
                parts = content.split(system_marker, 1)
                if len(parts) > 1:
                    system_user_parts = parts[1].split(user_marker, 1)
                    if len(system_user_parts) > 1:
                        system_prompt = system_user_parts[0].strip()
                        user_prompt = system_user_parts[1].strip()
            elif user_marker in content:
                parts = content.split(user_marker, 1)
                if len(parts) > 1:
                    user_prompt = parts[1].strip()
            else:
                user_prompt = content.strip()

            return system_prompt, user_prompt

        except Exception as e:
            logging.error(f"Error parsing prompt file {file_path}: {e}")
            return "", ""

    def get(self, name: str) -> dict:
        """Get prompt pair by name."""
        prompt = self.prompts.get(name)
        if not prompt:
            file_path = self.prompt_dir / f"{name}.txt"
            if file_path.exists():
                system_prompt, user_prompt = self._parse_prompt_file(file_path)
                prompt = {"system": system_prompt, "user": user_prompt}
                self.prompts[name] = prompt
                return prompt
            else:
                logging.warning(f"Prompt '{name}' not found")
                return {"system": "", "user": ""}
        return prompt

    def list_available_prompts(self) -> List[str]:
        """List all available prompt names."""
        available_prompts = set(self.prompts.keys())
        if self.prompt_dir.exists():
            file_prompts = {file_path.stem for file_path in self.prompt_dir.glob("*.txt")}
            available_prompts.update(file_prompts)
        return sorted(list(available_prompts))
