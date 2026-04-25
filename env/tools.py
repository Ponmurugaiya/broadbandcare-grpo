"""Deterministic tool runtime for BroadbandCare support cases."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


READ_ONLY_TOOLS = [
    "get_account_details",
    "get_user_broadband_location",
    "run_speed_test",
    "get_ping_stats",
    "get_router_stats",
    "search_troubleshooting_docs",
]

ACTION_TOOLS = [
    "change_dns_settings",
    "restart_router",
    "create_support_ticket",
    "resolve_ticket",
]

ALL_TOOLS = READ_ONLY_TOOLS + ACTION_TOOLS


DOCS = {
    "slow_speed": "For slow speed: check plan, run speed test, inspect router load, restart router.",
    "no_internet": "For no internet: check outage/location and ping, then DNS/router actions, escalate if unresolved.",
    "intermittent": "For intermittent drops: inspect packet loss and signal strength, reboot router, retest stability.",
    "high_latency": "For high latency: verify ping and packet loss, apply DNS optimization, retest.",
    "wrong_plan": "For wrong plan mapping: verify account details and escalate for billing-plan sync.",
}


def _tool_error(message: str) -> Dict[str, Any]:
    return {"ok": False, "error": message}


def execute_tool(
    tool_name: str,
    case: Dict[str, Any],
    state: Dict[str, Any],
    args: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute one tool and return (result, updated_state)."""
    args = args or {}
    updated_state = dict(state)

    if tool_name not in ALL_TOOLS:
        return _tool_error(f"unknown tool '{tool_name}'"), updated_state

    account = case["account"]
    metrics = case["metrics"]
    issue_type = case["issue_type"]

    if tool_name == "get_account_details":
        return {
            "ok": True,
            "tool": tool_name,
            "plan_speed_mbps": account["plan_speed_mbps"],
            "throttled": account["throttled"],
            "usage_gb": account["usage_gb"],
        }, updated_state

    if tool_name == "get_user_broadband_location":
        return {"ok": True, "tool": tool_name, "location": case["location"]}, updated_state

    if tool_name == "run_speed_test":
        boost = 18 if updated_state.get("router_restarted") else 0
        boost += 8 if updated_state.get("dns_fixed") else 0
        speed = min(account["plan_speed_mbps"], metrics["speed_test_mbps"] + boost)
        upload = min(max(5, int(speed * 0.4)), metrics["upload_mbps"] + int(boost * 0.4))
        return {
            "ok": True,
            "tool": tool_name,
            "download_mbps": speed,
            "upload_mbps": upload,
            "latency_ms": max(1, metrics["ping_ms"] - (10 if updated_state.get("dns_fixed") else 0)),
        }, updated_state

    if tool_name == "get_ping_stats":
        ping = metrics["ping_ms"] - (12 if updated_state.get("dns_fixed") else 0)
        loss = max(0.0, metrics["packet_loss_pct"] - (2.0 if updated_state.get("router_restarted") else 0.0))
        return {
            "ok": True,
            "tool": tool_name,
            "avg_ping_ms": max(1, ping),
            "packet_loss_pct": round(loss, 1),
        }, updated_state

    if tool_name == "get_router_stats":
        signal = metrics["signal_strength"]
        if updated_state.get("router_restarted") and signal == "weak":
            signal = "medium"
        return {
            "ok": True,
            "tool": tool_name,
            "connected_devices": metrics["connected_devices"],
            "signal_strength": signal,
            "router_restarted": updated_state.get("router_restarted", False),
        }, updated_state

    if tool_name == "search_troubleshooting_docs":
        issue = args.get("issue", issue_type)
        snippet = DOCS.get(issue, DOCS[issue_type])
        return {"ok": True, "tool": tool_name, "issue": issue, "doc_snippet": snippet}, updated_state

    if tool_name == "change_dns_settings":
        updated_state["dns_fixed"] = True
        return {"ok": True, "tool": tool_name, "message": "DNS set to ISP recommended profile."}, updated_state

    if tool_name == "restart_router":
        updated_state["router_restarted"] = True
        return {"ok": True, "tool": tool_name, "message": "Router restart command completed."}, updated_state

    if tool_name == "create_support_ticket":
        updated_state["ticket_created"] = True
        ticket_no = updated_state.get("ticket_number", f"TKT-{case['case_id']}")
        updated_state["ticket_number"] = ticket_no
        return {"ok": True, "tool": tool_name, "ticket_number": ticket_no}, updated_state

    if tool_name == "resolve_ticket":
        updated_state["resolved"] = True
        return {"ok": True, "tool": tool_name, "message": "Resolution attempt recorded."}, updated_state

    return _tool_error("unreachable"), updated_state


def get_allowed_tools() -> List[str]:
    return list(ALL_TOOLS)
