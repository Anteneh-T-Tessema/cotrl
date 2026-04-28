"""Evaluate a trained model checkpoint and compare against baseline.

Loads a model, runs it against the eval split of the puzzle dataset, and
prints a comparison table. Use this after each RL iteration to confirm
solve rate improvement.

Usage:
    # Evaluate baseline
    python scripts/evaluate.py --model Qwen/Qwen1.5-7B-Chat --run-name baseline

    # Evaluate a fine-tuned checkpoint
    python scripts/evaluate.py --model checkpoints/grpo/final --run-name iter_005

    # Compare two saved runs
    python scripts/evaluate.py --compare results/baseline.json results/iter_005.json
"""

import argparse
from pathlib import Path

from loguru import logger

from src.data.puzzles import load_puzzles, generate_puzzles
from src.eval.metrics import EvalResult, evaluate_model, compare_runs
from src.llm.generator import LLMGenerator, GenerationConfig
from src.llm.prompts import build_cot_prompt
from src.verifier.core import extract_expression


def _make_generate_fn(generator: LLMGenerator):
    def generate(numbers: list[int]) -> str:
        messages = build_cot_prompt(numbers)
        return generator.generate(messages)
    return generate


def run_eval(model_name: str, run_name: str, dataset_path: Path, output_path: Path) -> EvalResult:
    if dataset_path.exists():
        dataset = load_puzzles(dataset_path)
        logger.info(f"Loaded {len(dataset)} puzzles from {dataset_path}")
    else:
        logger.warning(f"{dataset_path} not found — generating 200 puzzles on the fly.")
        dataset = generate_puzzles(n=200, seed=99)  # use different seed from train

    _, eval_set = dataset.split(train_frac=0.8)
    logger.info(f"Evaluating on {len(eval_set)} puzzles.")

    generator = LLMGenerator(GenerationConfig(model_name=model_name))
    result = evaluate_model(
        generate_fn=_make_generate_fn(generator),
        dataset=eval_set,
        run_name=run_name,
    )
    result.save(output_path)
    return result


def main(args: argparse.Namespace) -> None:
    if args.compare:
        results = [EvalResult.load(Path(p)) for p in args.compare]
        print(compare_runs(results))
        return

    output_path = Path("results") / f"{args.run_name}.json"
    result = run_eval(
        model_name=args.model,
        run_name=args.run_name,
        dataset_path=Path(args.dataset),
        output_path=output_path,
    )
    print(result.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-7B-Chat")
    parser.add_argument("--run-name", type=str, default="eval")
    parser.add_argument("--dataset", type=str, default="data/raw/puzzles.jsonl")
    parser.add_argument("--compare", nargs="+", help="Compare saved EvalResult JSON files")
    args = parser.parse_args()
    main(args)
