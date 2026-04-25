"""BroadbandCare baseline inference for tool-calling environment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import textwrap
import socket
from typing import List, Optional
import urllib.request
import re

def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

# Ensure we can import local modules even when running directly
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

API_KEY       = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/featherless-ai/v1")
MODEL_NAME    = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-1.5B-Instruct")
BENCHMARK     = "broadbandcare"
LOCAL_IMAGE   = os.getenv("LOCAL_IMAGE_NAME")
SERVER_URL    = os.getenv("BROADBANDCARE_URL")

MAX_STEPS     = 10
TEMPERATURE   = 0.3       # low temp for reproducibility
MAX_TOKENS    = 200
SUCCESS_THRESHOLD = 1.0
EPISODES = int(os.getenv("EPISODES", "3"))
NO_LLM = os.getenv("NO_LLM", "0") == "1"
MODEL_FALLBACKS = [
    s.strip()
    for s in os.getenv(
        "MODEL_FALLBACKS",
        "Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-72B-Instruct",
    ).split(",")
    if s.strip()
]

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Agent persona
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a broadband support troubleshooting agent.
    Pick exactly one tool call per turn.
    You MUST choose a tool from the provided Available tools list. Do NOT invent new tool names.
    If you are uncertain, prefer diagnostic tools first (account/location/speed/ping/router/docs),
    then take one corrective action (dns/restart/escalate), then call resolve_ticket.

    Respond ONLY with valid JSON:
    {"tool":"<tool_name>","args":{}}
""").strip()

_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _parse_tool_json(raw_text: str) -> dict:
    """
    Parse a tool-call JSON object from an LLM response.
    Tries strict json.loads first, then extracts the first {...} block.
    """
    text = (raw_text or "").strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()
            if text.startswith("json"):
                text = text[4:].strip()
    try:
        return json.loads(text)
    except Exception:
        m = _JSON_OBJECT_RE.search(text)
        if not m:
            raise
        return json.loads(m.group(0))


def _make_user_prompt(obs_data: dict) -> str:
    """Build the per-step user prompt from observation data."""
    lines = [
        f"Customer says: \"{obs_data.get('customer_message', '')}\"",
    ]
    if obs_data.get("last_result"):
        lines.append(f"Last tool result: {json.dumps(obs_data['last_result'])}")
    if obs_data.get("history"):
        lines.append(f"History: {json.dumps(obs_data['history'][-3:])}")
    lines.append(f"Available tools: {obs_data.get('available_tools', [])}")
    lines.append("")
    lines.append("Respond with JSON only: {\"tool\": \"...\", \"args\": {}}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SERVER LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────

_server_process: Optional[subprocess.Popen] = None


def _start_local_server() -> str:
    """Start the BroadbandCare FastAPI server as a subprocess and return its URL."""
    global _server_process
    if _server_process and _server_process.poll() is None:
        _server_process.terminate()
        time.sleep(1)

    env = os.environ.copy()

    port = _get_free_port()
    _server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "error"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
    )
    url = f"http://127.0.0.1:{port}"
    # Wait for server to be ready
    import urllib.request
    for _ in range(30):
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError(f"Server did not start in time on {url}.")
    return url


def _stop_local_server() -> None:
    global _server_process
    if _server_process:
        _server_process.terminate()
        _server_process = None


def _server_supports_tool_schema(base_url: str) -> bool:
    """Return True if server /schema exposes 'tool' action field."""
    try:
        with urllib.request.urlopen(f"{base_url}/schema", timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        action_props = (payload.get("action") or {}).get("properties") or {}
        return "tool" in action_props
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# EPISODE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

from client import BroadbandcareEnv
from models import BroadbandcareAction

def run_task_via_client(task_name: str, client: OpenAI, server_url: str) -> dict:
    """
    Run one task episode using the EnvClient over WebSocket.
    Returns a dict with: score, rewards, steps, success.
    """
    env_client = BroadbandcareEnv(base_url=server_url)
    
    with env_client.sync() as env:
        reset_resp = env.reset()
        def _to_dict(obj):
            if isinstance(obj, dict): return obj
            if getattr(obj, "model_dump", None): return obj.model_dump()
            if getattr(obj, "dict", None): return obj.dict()
            if hasattr(obj, "__dict__"): return obj.__dict__
            return {}

        reset_obs_raw = getattr(reset_resp, "observation", reset_resp)
        obs = _to_dict(reset_obs_raw)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({
            "role": "user",
            "content": (
                f"You are handling a support call. Task: {task_name.replace('_', ' ')}.\n\n"
                + _make_user_prompt(obs)
            ),
        })

        rewards: List[float] = []
        last_action_str = "null"
        last_error = "null"
        step_count = 0
        done = False

        print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

        while step_count < MAX_STEPS:
            step_count += 1

            # ── Tool selection (LLM or deterministic fallback) ──
            if NO_LLM:
                # Simple deterministic policy to validate environment wiring
                if step_count == 1:
                    action_type, payload = "get_account_details", {}
                elif step_count == 2:
                    action_type, payload = "run_speed_test", {}
                elif step_count == 3:
                    action_type, payload = "get_router_stats", {}
                elif step_count == 4:
                    action_type, payload = "restart_router", {}
                elif step_count == 5:
                    action_type, payload = "run_speed_test", {}
                else:
                    action_type, payload = "resolve_ticket", {}
                raw_text = json.dumps({"tool": action_type, "args": payload})
                last_action_str = f"{action_type}({json.dumps(payload)})"
                last_error = "null"
            else:
                for attempt in range(3):
                    try:
                        time.sleep(2)  # Pace requests out to avoid bursting API limits
                        last_exc = None
                        response = None
                        for model_try in [MODEL_NAME, *MODEL_FALLBACKS]:
                            try:
                                response = client.chat.completions.create(
                                    model=model_try,
                                    messages=messages,
                                    temperature=TEMPERATURE,
                                    max_tokens=MAX_TOKENS,
                                )
                                break
                            except Exception as e:
                                last_exc = e
                                msg = str(e)
                                if ("not supported" in msg) or ("model" in msg and "supported" in msg):
                                    continue
                                raise
                        if response is None and last_exc is not None:
                            raise last_exc
                        raw_text = response.choices[0].message.content.strip()
                        action_json = _parse_tool_json(raw_text)
                        action_type = action_json.get("tool", "get_account_details")
                        payload = action_json.get("args", {})
                        last_action_str = f"{action_type}({json.dumps(payload)})"
                        last_error = "null"
                        break # Success!
                    except Exception as exc:
                        last_error = str(exc).replace("\n", " ")[:120]
                        if attempt < 2 and "402" not in last_error:
                            time.sleep(5) # Retry on transient errors
                        else:
                            action_type = "get_account_details"
                            payload = {}
                            last_action_str = "get_account_details({})"
                            raw_text = '{"tool": "get_account_details", "args": {}}'
                            break # Give up after 3 attempts or on hard 402 error

            # ── env.step ──
            try:
                action = BroadbandcareAction(tool=action_type, args=payload)
                step_resp = env.step(action)
                obs_raw = getattr(step_resp, "observation", step_resp.get("observation", {}) if isinstance(step_resp, dict) else {})
                obs = _to_dict(obs_raw)
                reward = float(getattr(step_resp, "reward", step_resp.get("reward", 0.0) if isinstance(step_resp, dict) else 0.0))
                done = bool(getattr(step_resp, "done", step_resp.get("done", False) if isinstance(step_resp, dict) else False))
            except Exception as exc:
                reward = 0.0
                done = True
                obs = {}
                last_error = str(exc).replace("\n", " ")[:120]

            rewards.append(reward)

            # ── [STEP] log ──
            done_str = "true" if done else "false"
            print(
                f"[STEP] step={step_count} action={last_action_str} "
                f"reward={reward:.2f} done={done_str} error={last_error}"
            )

            if done:
                break

            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": _make_user_prompt(obs)})

        # ── Final score ──
        final_score = float(obs.get("episode_reward", 0.0)) if isinstance(obs, dict) else 0.0
        success = final_score >= SUCCESS_THRESHOLD
        success_str = "true" if success else "false"
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        print(
            f"[END] success={success_str} steps={step_count} "
            f"score={final_score:.2f} rewards={rewards_str}"
        )

        try:
            os.makedirs("logs", exist_ok=True)
            with open(f"logs/{task_name}_debug.json", "w") as f:
                json.dump(messages, f, indent=2)
        except Exception as log_exc:
            print(f"Warning: Failed to write debug log: {log_exc}", file=sys.stderr)

        return {
            "score": final_score,
            "rewards": rewards,
            "steps": step_count,
            "success": success,
        }

def main():
    if not NO_LLM and not API_KEY:
        print("ERROR: HF_TOKEN (or API_KEY / OPENAI_API_KEY) is not set.", file=sys.stderr)
        sys.exit(1)

    llm_client = OpenAI(api_key=API_KEY or "DUMMY", base_url=API_BASE_URL)

    task_results = {}
    external_url = os.getenv("BROADBANDCARE_URL")
    current_url = external_url or SERVER_URL
    if external_url and not _server_supports_tool_schema(external_url):
        print(
            "Warning: BROADBANDCARE_URL points to a server with legacy action schema. "
            "Starting local server instead.",
            file=sys.stderr,
        )
        external_url = None
        current_url = None

    if not external_url:
        try:
            current_url = _start_local_server()
        except RuntimeError as e:
            print(f"Fatal: {e}", file=sys.stderr)
            sys.exit(1)

    for i in range(EPISODES):
        task_name = f"episode_{i+1}"
        task_results[task_name] = run_task_via_client(task_name, llm_client, current_url)

    _stop_local_server()

    # Summary
    total_score = sum(r["score"] for r in task_results.values()) / len(task_results)
    print(f"\n{'='*50}")
    print(f"OVERALL SCORE: {total_score:.2f}")
    for task_name, result in task_results.items():
        status = "OK" if result["success"] else "FAIL"
        print(f"  {status} {task_name}: {result['score']:.2f} ({result['steps']} steps)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
