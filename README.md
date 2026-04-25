# BroadbandCare Agent (Tool-Based RL)

BroadbandCare is an OpenEnv-compatible environment where an agent learns to diagnose and resolve broadband issues by selecting tools step-by-step under partial observability.

## Project Layout

```text
broadbandcare-agent/
├── env/
│   ├── env.py
│   ├── tools.py
│   └── reward.py
├── data/
│   └── cases.json
├── scripts/
│   └── generate_cases.py
├── training/
│   └── train.ipynb
├── server/
│   └── app.py
├── models.py
├── client.py
├── openenv.yaml
└── requirements.txt
```

## Environment Design

- **Partial observation**: agent sees `customer_message`, `history`, and `last_result`.
- **Hidden state**: case `metrics` and `solution_path` stay internal to the environment.
- **Tool API**:
  - Read: `get_account_details`, `get_user_broadband_location`, `run_speed_test`, `get_ping_stats`, `get_router_stats`, `search_troubleshooting_docs`
  - Action: `change_dns_settings`, `restart_router`, `create_support_ticket`, `resolve_ticket`

## Reward Function

Per step:

- `-0.1` step cost
- `+1.0` if selected tool is in hidden solution path
- `-1.0` if selected tool is off path
- terminal `+5.0` for correct `resolve_ticket`
- terminal `-3.0` for incomplete/incorrect resolution

Implemented in `env/reward.py`.

## Quick Start

Install:

```bash
pip install -r requirements.txt
```

Run server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Run baseline inference:

```bash
HF_TOKEN=hf_xxx python inference.py
```

Run with existing server:

```bash
BROADBANDCARE_URL=http://localhost:8000 HF_TOKEN=hf_xxx python inference.py
```

## Dataset Workflow

Initial curated cases:

- `data/cases.json` ships with 20 cases.

Expand to 50-100 cases deterministically:

```bash
python scripts/generate_cases.py --target-count 80 --seed 7
```

## Training (GRPO + Unsloth)

Use `training/train.ipynb` in Colab:

1. Install dependencies (`unsloth`, `trl`, `transformers`, etc.).
2. Load `Qwen/Qwen2.5-1.5B-Instruct` in 4-bit.
3. Connect environment rollouts.
4. Configure GRPO trainer.
5. Log and plot training metrics.

## Judge-Facing Evidence Checklist

Capture these artifacts for submission:

- Loss curve over training steps/epochs
- Mean episode reward curve
- Tool accuracy (`tool_accuracy`) trend
- Resolution success rate before vs after training
- Qualitative episode comparison (early vs late policy)

## Validation Notes

- Python syntax check:

```bash
python -m compileall models.py client.py env server/app.py scripts/generate_cases.py inference.py
```

- Basic smoke test:

```bash
python -c "from env.env import BroadbandSupportEnv; from models import BroadbandcareAction; e=BroadbandSupportEnv(seed=1); e.reset(); print(e.step(BroadbandcareAction(tool='get_account_details', args={})).reward)"
```
