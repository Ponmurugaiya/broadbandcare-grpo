"""Reward shaping functions for BroadbandCare GRPO environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class RewardConfig:
    step_cost: float = -0.1
    correct_step_bonus: float = 1.0
    wrong_step_penalty: float = -1.0
    repeat_penalty: float = -0.4
    resolve_success_bonus: float = 5.0
    resolve_incomplete_penalty: float = -3.0
    invalid_json_penalty: float = -2.0
    invalid_tool_penalty: float = -1.5


DEFAULT_REWARD_CONFIG = RewardConfig()


def compute_step_reward(
    tool_name: str,
    solution_path: List[str],
    called_tools: List[str],
    resolved_correctly: bool,
    is_resolve_action: bool,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    """Compute dense + sparse reward for one environment step.

    Reward is *order-aware*: only the next expected tool in `solution_path`
    receives the `correct_step_bonus`. This prevents reward hacking by repeating
    any single in-path tool.
    """
    reward = config.step_cost

    expected_next = next_expected_tool(solution_path, called_tools[:-1])
    if expected_next is None:
        # No remaining required steps; only resolve should be taken.
        reward += config.correct_step_bonus if tool_name == "resolve_ticket" else config.wrong_step_penalty
    else:
        reward += config.correct_step_bonus if tool_name == expected_next else config.wrong_step_penalty

    if len(called_tools) >= 2 and called_tools[-1] == called_tools[-2]:
        reward += config.repeat_penalty

    if is_resolve_action:
        reward += (
            config.resolve_success_bonus
            if resolved_correctly
            else config.resolve_incomplete_penalty
        )
    return round(reward, 4)


def next_expected_tool(solution_path: List[str], called_tools: List[str]) -> str | None:
    """
    Return the next required tool according to `solution_path` order.
    Ignores extra (off-path) calls; progresses when a required step is matched in order.
    """
    required = [t for t in solution_path if t != "resolve_ticket"]
    idx = 0
    for t in called_tools:
        if idx < len(required) and t == required[idx]:
            idx += 1
    if idx >= len(required):
        return None
    return required[idx]


def resolution_is_correct(solution_path: List[str], called_tools: List[str]) -> bool:
    """Resolution is correct if required tools occurred in-order before resolve."""
    required = [t for t in solution_path if t != "resolve_ticket"]
    idx = 0
    for t in called_tools:
        if t == "resolve_ticket":
            break
        if idx < len(required) and t == required[idx]:
            idx += 1
    return idx == len(required)


def compute_metrics(solution_path: List[str], called_tools: List[str], resolved: bool) -> Dict[str, float]:
    """Episode metrics for judge-facing evidence."""
    if not called_tools:
        return {"tool_accuracy": 0.0, "path_coverage": 0.0, "resolved": 0.0}

    solution_set = set(solution_path)
    matches = sum(1 for t in called_tools if t in solution_set)
    coverage = sum(1 for t in solution_set if t in set(called_tools)) / max(1, len(solution_set))
    return {
        "tool_accuracy": round(matches / len(called_tools), 4),
        "path_coverage": round(coverage, 4),
        "resolved": 1.0 if resolved else 0.0,
    }
