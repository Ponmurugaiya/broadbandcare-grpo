"""Action and observation models for tool-calling BroadbandCare env."""

from typing import Any, Dict, List

from openenv.core.env_server.types import Action, Observation
from pydantic import Field

class BroadbandcareAction(Action):
    """Agent action in JSON-tool format."""

    tool: str = Field(..., description="Tool name to execute.")
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool argument payload.",
    )


class BroadbandcareObservation(Observation):
    """Partial observation for GRPO rollouts."""

    customer_message: str = Field(
        default="",
        description="Original customer issue statement.",
    )
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of prior tool calls and results.",
    )
    last_result: Dict[str, Any] = Field(
        default_factory=dict,
        description="Latest tool execution output.",
    )
    available_tools: List[str] = Field(
        default_factory=list,
        description="Allowed tool names in this environment.",
    )
    step_count: int = Field(default=0, description="Episode step count.")
    max_steps: int = Field(default=10, description="Maximum steps before forced termination.")
    episode_reward: float = Field(default=0.0, description="Cumulative reward in the episode.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary metrics such as tool accuracy and path coverage.",
    )
