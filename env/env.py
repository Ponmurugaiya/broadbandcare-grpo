"""OpenEnv-compatible broadband support environment."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import BroadbandcareAction, BroadbandcareObservation
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models import BroadbandcareAction, BroadbandcareObservation

from .reward import compute_metrics, compute_step_reward, resolution_is_correct
from .tools import execute_tool, get_allowed_tools


ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "data" / "cases.json"
MAX_STEPS = 10

TOOL_ALIASES: Dict[str, str] = {
    # Common LLM synonyms / variations
    "reboot_router": "restart_router",
    "restart_modem": "restart_router",
    "reboot_modem": "restart_router",
    "reset_router": "restart_router",
    "dns_change": "change_dns_settings",
    "set_dns": "change_dns_settings",
    "update_dns": "change_dns_settings",
    "open_ticket": "create_support_ticket",
    "raise_ticket": "create_support_ticket",
    "escalate": "create_support_ticket",
    "close": "resolve_ticket",
    "resolve": "resolve_ticket",
    # Slightly different naming for read tools
    "speedtest": "run_speed_test",
    "speed_test": "run_speed_test",
    "ping": "get_ping_stats",
    "router_stats": "get_router_stats",
    "signal_strength": "get_router_stats",
    "check_signal_strength": "get_router_stats",
    "account_details": "get_account_details",
    "location": "get_user_broadband_location",
    "troubleshooting_docs": "search_troubleshooting_docs",

    # Extra common "metrics" names LLMs invent
    "check_packet_loss": "get_ping_stats",
    "packet_loss": "get_ping_stats",
    "analyze_signal_strength": "get_router_stats",
    "analyze_router_signal": "get_router_stats",
    "reset_wifi_network": "restart_router",
    "upgrade_firmware": "create_support_ticket",

    # Seen in your logs
    "apply_dns_optimization": "change_dns_settings",
    "optimize_dns": "change_dns_settings",
    "check_plan": "get_account_details",
    "docs": "search_troubleshooting_docs",
}


def _normalize_tool_name(tool_name: str) -> str:
    key = (tool_name or "").strip()
    key_lower = key.lower().replace("-", "_")
    mapped = TOOL_ALIASES.get(key_lower, key_lower)
    return mapped


def _load_cases() -> List[Dict[str, Any]]:
    with CASES_PATH.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list) or not cases:
        raise ValueError("data/cases.json must contain a non-empty list")
    return cases


class BroadbandSupportEnv(Environment):
    """Tool-calling support environment with partial observability."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._cases = _load_cases()
        self._allowed_tools = get_allowed_tools()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_case: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []
        self._tool_calls: List[str] = []
        self._episode_reward = 0.0
        self._done = False
        self._last_result: Dict[str, Any] = {}
        self._hidden_state: Dict[str, Any] = {}

    def reset(self) -> BroadbandcareObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_case = self._rng.choice(self._cases)
        self._history = []
        self._tool_calls = []
        self._episode_reward = 0.0
        self._done = False
        self._last_result = {"ok": True, "message": "Episode started."}
        self._hidden_state = {
            "dns_fixed": False,
            "router_restarted": False,
            "ticket_created": False,
            "resolved": False,
        }
        return self._get_observation(reward=0.0)

    def step(self, action: BroadbandcareAction) -> BroadbandcareObservation:  # type: ignore[override]
        if self._done:
            return self._get_observation(reward=0.0)

        self._state.step_count += 1
        reward = 0.0

        tool_name = _normalize_tool_name(action.tool)
        args = action.args or {}

        if tool_name not in self._allowed_tools:
            self._last_result = {"ok": False, "error": f"invalid tool '{tool_name}'"}
            reward = -1.5
        else:
            result, new_state = execute_tool(tool_name, self._current_case, self._hidden_state, args=args)
            self._hidden_state = new_state
            self._last_result = result
            self._tool_calls.append(tool_name)

            is_resolve = tool_name == "resolve_ticket"
            resolved_ok = resolution_is_correct(
                self._current_case["solution_path"],
                self._tool_calls,
            )
            reward = compute_step_reward(
                tool_name=tool_name,
                solution_path=self._current_case["solution_path"],
                called_tools=self._tool_calls,
                resolved_correctly=resolved_ok,
                is_resolve_action=is_resolve,
            )
            if is_resolve or self._state.step_count >= MAX_STEPS:
                self._done = True

        self._episode_reward += reward
        self._history.append({"tool": tool_name, "args": args, "result": self._last_result, "reward": reward})
        if self._state.step_count >= MAX_STEPS:
            self._done = True
        return self._get_observation(reward=reward)

    def _get_observation(self, reward: float) -> BroadbandcareObservation:
        metrics = compute_metrics(
            solution_path=self._current_case.get("solution_path", []),
            called_tools=self._tool_calls,
            resolved=self._hidden_state.get("resolved", False),
        )
        return BroadbandcareObservation(
            customer_message=self._current_case.get("customer_message", ""),
            history=self._history,
            last_result=self._last_result,
            available_tools=self._allowed_tools,
            step_count=self._state.step_count,
            max_steps=MAX_STEPS,
            episode_reward=round(self._episode_reward, 4),
            done=self._done,
            reward=round(reward, 4),
            metadata=metrics,
        )

    @property
    def state(self) -> State:
        return self._state
