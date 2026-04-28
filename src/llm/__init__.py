from .prompts import build_cot_prompt
from .few_shot import select_few_shot_examples, load_few_shot_examples, SEED_EXAMPLES

# LLMGenerator is intentionally not imported here — it pulls in transformers
# and torch which are optional heavy dependencies. Import it directly:
#   from src.llm.generator import LLMGenerator

__all__ = [
    "build_cot_prompt",
    "select_few_shot_examples",
    "load_few_shot_examples",
    "SEED_EXAMPLES",
]
