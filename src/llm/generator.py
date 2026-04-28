from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


@dataclass
class GenerationConfig:
    model_name: str = "Qwen/Qwen1.5-7B-Chat"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    num_return_sequences: int = 1
    device_map: str = "auto"
    load_in_4bit: bool = True


class LLMGenerator:
    """Wraps a local HuggingFace model for inference.

    Designed to be swappable — pass a different model_name to GenerationConfig
    without changing any downstream code.
    """

    def __init__(self, config: Optional[GenerationConfig] = None) -> None:
        self.config = config or GenerationConfig()
        self._pipe = None

    def _load(self) -> None:
        logger.info(f"Loading model: {self.config.model_name}")
        quantization_config = None
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map=self.config.device_map,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )
        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        logger.info("Model loaded.")

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Run inference on a messages list and return the assistant response text."""
        if self._pipe is None:
            self._load()

        outputs = self._pipe(
            messages,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            num_return_sequences=self.config.num_return_sequences,
            do_sample=True,
        )
        return outputs[0]["generated_text"][-1]["content"]

    def generate_batch(self, batch: list[list[dict[str, str]]]) -> list[str]:
        """Run inference on a batch of message lists."""
        return [self.generate(messages) for messages in batch]
