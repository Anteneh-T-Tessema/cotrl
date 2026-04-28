"""GRPO training loop using HuggingFace TRL.

GRPO (Group Relative Policy Optimization) is preferred over PPO here because:
  - No value/critic network required — reduces memory footprint significantly
  - Reward signal is binary (0/1 from verifier) — PPO's GAE advantage offers
    little benefit over group-normalized rewards in this setting
  - Simpler hyperparameter surface for a small team

Why ALL trajectories, not just solved ones:
  GRPO normalises rewards within each group of G completions:
    A_i = (r_i - mean(r_g)) / (std(r_g) + ε)
  With binary rewards many groups have zero variance (all-0 or all-1),
  producing zero gradient. Shaped rewards give every group non-zero variance
  so more iterations produce useful gradient signal.
  Trajectories with reward == 0.0 are still excluded — they carry no
  information about what made a response better or worse.

See ADRs: docs/adr/002-implement-grpo-instead-of-ppo.md
          docs/adr/003-shaped-rewards-for-grpo.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from .trajectory import Trajectory, TrajectoryBuffer


@dataclass
class GRPOConfig:
    model_name: str = "Qwen/Qwen1.5-7B-Chat"
    output_dir: str = "checkpoints/grpo"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    group_size: int = 8          # completions per prompt for GRPO group advantage
    kl_coef: float = 0.05        # KL penalty against reference model
    max_new_tokens: int = 512
    min_reward_threshold: float = 0.0  # exclude pure zero-reward trajectories
    seed: int = 42


class GRPOTrainer:
    """Thin wrapper around TRL's GRPOTrainer for the Game of 24 task."""

    def __init__(self, config: Optional[GRPOConfig] = None) -> None:
        self.config = config or GRPOConfig()

    def _select_trajectories(self, buffer: TrajectoryBuffer) -> list[Trajectory]:
        """Return trajectories with reward above threshold, sorted descending.

        Including partial-credit trajectories (format bonus, numbers bonus)
        gives GRPO more within-group variance and faster early learning.
        """
        return sorted(
            [t for t in buffer.all() if t.reward > self.config.min_reward_threshold],
            key=lambda t: t.reward,
            reverse=True,
        )

    def train(self, buffer: TrajectoryBuffer) -> None:
        """Run a GRPO training step using all trajectories with reward > 0.

        Args:
            buffer: Trajectory buffer populated by a rollout phase. Trajectories
                with shaped rewards are preferred — binary 0/1 rewards still work
                but produce less gradient signal per iteration.
        """
        try:
            from trl import GRPOTrainer as TRLGRPOTrainer, GRPOConfig as TRLGRPOConfig
        except ImportError:
            raise RuntimeError("Install trl>=0.8.6: pip install trl")

        trajectories = self._select_trajectories(buffer)
        if not trajectories:
            logger.warning("No trajectories with reward > 0 — skipping training step.")
            return

        solved = sum(t.solved for t in trajectories)
        avg_reward = sum(t.reward for t in trajectories) / len(trajectories)
        logger.info(
            f"Training on {len(trajectories)} trajectories | "
            f"Solved: {solved} | Avg shaped reward: {avg_reward:.3f}"
        )

        dataset = self._to_hf_dataset(trajectories)
        trl_config = TRLGRPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            seed=self.config.seed,
        )

        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        trainer = TRLGRPOTrainer(
            model=model,
            args=trl_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        trainer.train()
        trainer.save_model(Path(self.config.output_dir) / "final")
        logger.info(f"Model saved to {self.config.output_dir}/final")

    def _to_hf_dataset(self, trajectories: list[Trajectory]) -> Any:
        from datasets import Dataset
        return Dataset.from_list([
            {
                "prompt": t.prompt,
                "response": t.response,
                "reward": [t.reward],   # TRL expects a list per trajectory
            }
            for t in trajectories
        ])
