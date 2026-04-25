"""Deterministically expand BroadbandCare case dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "data" / "cases.json"

LOCATIONS = [
    "Chennai",
    "Mumbai",
    "Delhi",
    "Coimbatore",
    "Lucknow",
    "Hyderabad",
    "Jaipur",
    "Pune",
    "Mysuru",
    "Kolkata",
    "Nagpur",
    "Bhopal",
    "Patna",
    "Noida",
    "Indore",
]
PLAN_TYPES = ["fiber", "cable", "ftth"]
SIGNALS = ["strong", "medium", "weak"]
ISSUES = ["slow_speed", "no_internet", "intermittent", "high_latency", "wrong_plan"]


def _load_seed_cases() -> List[Dict[str, Any]]:
    with CASES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _deepcopy_json(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def _new_case_id(idx: int) -> str:
    return f"CASE-{idx:03d}"


def _mutate_case(seed_case: Dict[str, Any], idx: int, rng: random.Random) -> Dict[str, Any]:
    case = _deepcopy_json(seed_case)
    issue_type = rng.choice(ISSUES)
    location = rng.choice(LOCATIONS)
    plan_type = rng.choice(PLAN_TYPES)
    signal = rng.choice(SIGNALS)

    case["case_id"] = _new_case_id(idx)
    case["issue_type"] = issue_type
    case["location"] = location
    case["plan_type"] = plan_type

    plan_speed = rng.choice([60, 75, 100, 120, 150, 200, 250, 300])
    usage = rng.randint(20, 1250)
    throttled = usage > 1000

    metrics = case["metrics"]
    if issue_type == "no_internet":
        speed, upload, ping, loss = 0, 0, 0, 100.0
        solution = [
            "get_user_broadband_location",
            "get_ping_stats",
            "restart_router",
            "run_speed_test",
            "create_support_ticket",
            "resolve_ticket",
        ]
    elif issue_type == "slow_speed":
        speed = rng.randint(15, max(25, int(plan_speed * 0.35)))
        upload = max(5, int(speed * 0.5))
        ping = rng.randint(20, 60)
        loss = round(rng.uniform(0.0, 3.5), 1)
        solution = [
            "get_account_details",
            "run_speed_test",
            "get_router_stats",
            "restart_router",
            "run_speed_test",
            "resolve_ticket",
        ]
    elif issue_type == "high_latency":
        speed = rng.randint(int(plan_speed * 0.6), plan_speed)
        upload = rng.randint(max(15, int(speed * 0.3)), max(20, int(speed * 0.6)))
        ping = rng.randint(95, 170)
        loss = round(rng.uniform(0.1, 2.0), 1)
        solution = ["get_ping_stats", "change_dns_settings", "get_ping_stats", "resolve_ticket"]
    elif issue_type == "wrong_plan":
        speed = rng.randint(max(20, int(plan_speed * 0.8)), plan_speed)
        upload = rng.randint(max(10, int(speed * 0.3)), max(20, int(speed * 0.6)))
        ping = rng.randint(15, 35)
        loss = round(rng.uniform(0.0, 0.5), 1)
        solution = ["get_account_details", "create_support_ticket", "resolve_ticket"]
        plan_speed = rng.choice([100, 150, 200])
    else:
        speed = rng.randint(int(plan_speed * 0.45), int(plan_speed * 0.85))
        upload = rng.randint(max(10, int(speed * 0.3)), max(20, int(speed * 0.6)))
        ping = rng.randint(35, 80)
        loss = round(rng.uniform(3.0, 8.0), 1)
        solution = ["get_ping_stats", "get_router_stats", "restart_router", "get_ping_stats", "resolve_ticket"]

    case["customer_message"] = (
        f"Issue in {location}: {issue_type.replace('_', ' ')} on my {plan_type} broadband. "
        "Please diagnose and fix it."
    )
    case["account"] = {
        "plan_speed_mbps": plan_speed,
        "throttled": throttled,
        "usage_gb": usage,
    }
    metrics["speed_test_mbps"] = speed
    metrics["upload_mbps"] = upload
    metrics["ping_ms"] = ping
    metrics["packet_loss_pct"] = loss
    metrics["signal_strength"] = signal
    metrics["connected_devices"] = rng.randint(2, 25)
    case["metrics"] = metrics
    case["solution_path"] = solution
    return case


def expand_cases(target_count: int, seed: int = 7) -> List[Dict[str, Any]]:
    if target_count < 20:
        raise ValueError("target_count must be >= 20")
    base_cases = _load_seed_cases()
    rng = random.Random(seed)
    expanded = [*base_cases]

    idx = len(expanded) + 1
    while len(expanded) < target_count:
        source = rng.choice(base_cases)
        expanded.append(_mutate_case(source, idx, rng))
        idx += 1
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand BroadbandCare cases deterministically.")
    parser.add_argument("--target-count", type=int, default=60, help="Final number of cases.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for deterministic expansion.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(CASES_PATH),
        help="Output path for generated cases JSON.",
    )
    args = parser.parse_args()

    cases = expand_cases(target_count=args.target_count, seed=args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)
    print(f"Wrote {len(cases)} cases to {output_path}")


if __name__ == "__main__":
    main()
