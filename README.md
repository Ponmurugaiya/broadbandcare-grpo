---
title: BroadbandCare OpenEnv GRPO
emoji: 📡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# BroadbandCare — Tool-Based RL Agent for Telecom Support

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NhnN-Sujq2aobozC8l8ovi09Vo_VVKxG?usp=sharing)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-orange?logo=huggingface)](https://huggingface.co/spaces/Ponmurugaiya72/broadbandcare-openenv-grpo)

BroadbandCare is an **OpenEnv-compatible reinforcement learning environment** where a small language model learns to diagnose and resolve broadband support cases by calling tools step-by-step under partial observability. The agent is trained using **GRPO (Group Relative Policy Optimization)** on top of `Qwen2.5-1.5B-Instruct` via Unsloth + TRL.

---

## How It Works

Each episode presents the agent with a real-world-style broadband support case. The agent observes the customer's message and call history, then selects tools one at a time to diagnose and resolve the issue. The environment tracks a hidden solution path and rewards the agent for making the right tool choices in the right order.

```
Customer Message → Agent observes partial state
       ↓
   Selects Tool (e.g. run_speed_test)
       ↓
   Receives result + shaped reward
       ↓
   Continues until resolve_ticket or step limit
```

---

## Environment Design

| Component | Details |
|---|---|
| **Observation** | `customer_message`, `history`, `last_tool_result` |
| **Hidden State** | `metrics`, `solution_path` (not visible to agent) |
| **Max Steps** | Configurable per case |
| **Task Difficulty** | Medium |
| **Runtime** | FastAPI (`server/app.py`) on port 8000 |

### Available Tools

**Diagnostic (Read-only)**
- `get_account_details` — Fetch customer account info
- `get_user_broadband_location` — Check service area and line type
- `run_speed_test` — Measure current download/upload speeds
- `get_ping_stats` — Check latency and packet loss
- `get_router_stats` — Inspect router connection status
- `search_troubleshooting_docs` — Query internal KB for known issues

**Action (State-changing)**
- `change_dns_settings` — Update DNS configuration
- `restart_router` — Remotely reboot the customer's router
- `create_support_ticket` — Escalate to human agent
- `resolve_ticket` — Mark episode as resolved *(terminal action)*

---

## Reward Function

| Condition | Reward |
|---|---|
| Each step taken | `-0.1` (step cost) |
| Tool matches hidden solution path | `+1.0` |
| Tool does not match solution path | `-1.0` |
| Correct `resolve_ticket` call | `+5.0` (terminal) |
| Incorrect or incomplete resolution | `-3.0` (terminal) |

Implemented in `env/reward.py`.

---

## Project Structure

```
broadbandcare-fresh-v1/
├── env/
│   ├── env.py              # BroadbandSupportEnv (OpenEnv-compatible)
│   ├── tools.py            # Tool definitions and handlers
│   └── reward.py           # Deterministic reward shaping logic
├── data/
│   └── cases.json          # Curated support case dataset (20 base cases)
├── scripts/
│   └── generate_cases.py   # Deterministic case generator (up to 100 cases)
├── server/
│   └── app.py              # FastAPI server (OpenEnv runtime)
├── training/
│   └── train.ipynb         # GRPO training notebook (run in Colab)
├── models.py               # Pydantic models (Action, Observation, etc.)
├── client.py               # Python client for interacting with the env
├── inference.py            # Baseline rollout script
├── openenv.yaml            # OpenEnv environment specification
├── Dockerfile              # Multi-stage Docker build for HF Space
└── requirements.txt        # Python dependencies
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the environment server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 3. Run baseline inference

```bash
HF_TOKEN=hf_xxx python inference.py
```

Or against an already running server:

```bash
BROADBANDCARE_URL=http://localhost:8000 HF_TOKEN=hf_xxx python inference.py
```

---

## Dataset

`data/cases.json` ships with **20 curated cases** covering:
- Slow speeds / intermittent dropouts
- Complete outages (fiber and copper)
- Router / modem hardware issues
- DNS misconfiguration
- Billing and account disputes

To expand to 50–100 cases deterministically:

```bash
python scripts/generate_cases.py --target-count 80 --seed 7
```

---

## Training (GRPO + Unsloth)

Open the training notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NhnN-Sujq2aobozC8l8ovi09Vo_VVKxG?usp=sharing)

The pipeline:

1. **SFT** — Supervised fine-tuning on curated rollouts to give the base model task awareness
2. **GRPO** — Group Relative Policy Optimization with environment reward signal to teach tool selection strategy

**Model:** `Qwen/Qwen2.5-1.5B-Instruct` (4-bit quantized via Unsloth)  
**Key packages:** `unsloth`, `trl>=0.11.0`, `transformers>=4.46.0`, `accelerate`, `bitsandbytes`

---

## Results

| Model Stage | Behaviour |
|---|---|
| **Base** | Conversational but unfocused; tends to apologize and deflect |
| **SFT** | More structured; follows format but still mechanical |
| **GRPO** | Actively reasons; selects tools based on context; handles ambiguity |

The reward curve shows steady improvement across training steps, with the GRPO agent achieving significantly higher episode rewards than both the base and SFT checkpoints.

---

## Validation

Syntax check all modules:

```bash
python -m compileall models.py client.py env server/app.py scripts/generate_cases.py inference.py
```

Smoke test:

```bash
python -c "
from env.env import BroadbandSupportEnv
from models import BroadbandcareAction
e = BroadbandSupportEnv(seed=1)
e.reset()
print(e.step(BroadbandcareAction(tool='get_account_details', args={})).reward)
"
```

---

## Read the Blog

For a full walkthrough of the design decisions, training challenges, and results:  
📖 [From Chatbot to Thinking Agent: How I Built BroadbandCare with RL](./blog.md)

---

*Built with Qwen2.5-1.5B, Unsloth, TRL, OpenEnv, and too much coffee. ☕*
