"""BroadbandCare Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import BroadbandcareAction, BroadbandcareObservation
except ImportError:
    from models import BroadbandcareAction, BroadbandcareObservation


class BroadbandcareEnv(
    EnvClient[BroadbandcareAction, BroadbandcareObservation, State]
):
    """
    Client for the BroadbandCare Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own isolated environment session.

    Quick start (sync):
        with BroadbandcareEnv(base_url="http://localhost:8000").sync() as env:
            obs = env.reset()
            print(obs.customer_message)
            result = env.step(BroadbandcareAction(tool="get_account_details", args={}))
            print(result.observation.customer_message)

    Quick start (async):
        async with BroadbandcareEnv(base_url="http://localhost:8000") as env:
            obs = await env.reset()
            result = await env.step(BroadbandcareAction(tool="get_account_details", args={}))

    Docker launch:
        env = BroadbandcareEnv.from_docker_image("broadbandcare-env:latest")
        try:
            obs = env.reset()
        finally:
            env.close()
    """

    def _step_payload(self, action: BroadbandcareAction) -> Dict:
        """Convert BroadbandcareAction to JSON payload."""
        return {
            "tool": action.tool,
            "args": action.args,
        }

    def _parse_result(self, payload: Dict) -> StepResult[BroadbandcareObservation]:
        """Parse server JSON response into a typed StepResult."""
        obs_data = payload.get("observation", {})
        observation = BroadbandcareObservation(
            customer_message=obs_data.get("customer_message", ""),
            history=obs_data.get("history", []),
            last_result=obs_data.get("last_result", {}),
            available_tools=obs_data.get("available_tools", []),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 10),
            episode_reward=obs_data.get("episode_reward", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server state response."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
