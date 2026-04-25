"""FastAPI app for BroadbandCare tool-based RL environment."""

import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from e

try:
    from ..models import BroadbandcareAction, BroadbandcareObservation
    from ..env.env import BroadbandSupportEnv
except ImportError:
    import sys as _sys
    _here = os.path.dirname(os.path.abspath(__file__))          # .../server/
    _root = os.path.dirname(_here)                              # .../broadbandcare/
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from models import BroadbandcareAction, BroadbandcareObservation  # noqa
    from env.env import BroadbandSupportEnv  # noqa


# Create a single shared instance for HTTP mode (so state persists across requests)
# In a WebSocket or production MCP scenario, a factory can be used per-session.
_global_env = BroadbandSupportEnv()

app = create_app(
    lambda: _global_env,
    BroadbandcareAction,
    BroadbandcareObservation,
    env_name="broadbandcare",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution.

    Examples:
        uv run server
        python -m broadbandcare.server.app
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="BroadbandCare Environment Server")
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

