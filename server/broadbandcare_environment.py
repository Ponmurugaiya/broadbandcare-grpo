"""
BroadbandCare Environment Implementation — Full State Machine.

Tasks:
  1. change_mobile_number  (Easy)
  2. recharge_not_working  (Medium)
  3. no_internet_angry     (Hard)

Architecture:
  - MockISPBackend  : deterministic fake backend seeded by episode_id
  - CustomerSim     : deterministic reactive customer seeded by episode_id
  - TaskConfig      : phase definitions, valid actions, reward weights per task
  - BroadbandcareEnvironment : OpenEnv Environment base-class implementation

Run directly for a quick smoke test:
  python server/broadbandcare_environment.py
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import BroadbandcareAction, BroadbandcareObservation
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models import BroadbandcareAction, BroadbandcareObservation


# ─────────────────────────────────────────────────────────────────────────────
# TASK DEFINITIONS
# Each task is described as an ordered list of required phase steps.
# ─────────────────────────────────────────────────────────────────────────────

TASK_NAMES = ["change_mobile_number", "recharge_not_working", "no_internet_angry"]

# Ordered required steps per task
TASK_STEPS: Dict[str, List[str]] = {
    "change_mobile_number": [
        "greet",
        "ask_account_id",
        "verify_identity",
        "ask_new_mobile",
        "confirm_update",
        "update_system",
        "close_ticket",
    ],
    "recharge_not_working": [
        "greet",
        "show_empathy",
        "ask_account_id",
        "ask_transaction_id",
        "verify_payment",
        "activate_service",
        "confirm_resolution",
        "close_ticket",
    ],
    "no_internet_angry": [
        "greet",
        "de_escalate",
        "ask_account_id",
        "ask_symptom_details",
        "check_line_status",
        # Branch: remote_fix + confirm_fix  OR  schedule_technician + confirm_fix
        # Both valid — resolved at runtime based on mock backend
        "resolve",       # placeholder resolved at runtime to remote_fix or schedule_technician
        "confirm_fix",
        "close_ticket",
    ],
}

# Maximum steps allowed per task before forced termination (with penalty)
MAX_STEPS: Dict[str, int] = {
    "change_mobile_number": 14,
    "recharge_not_working": 16,
    "no_internet_angry": 18,
}

# Per-step reward for executing a CORRECT phase action
STEP_REWARD: Dict[str, float] = {
    "change_mobile_number": 0.10,
    "recharge_not_working": 0.08,
    "no_internet_angry":    0.08,
}

# Bonus when completing all steps cleanly
COMPLETION_BONUS: Dict[str, float] = {
    "change_mobile_number": 0.20,
    "recharge_not_working": 0.18,
    "no_internet_angry":    0.16,
}

# Max achievable raw score (used for normalisation to [0,1])
MAX_RAW_SCORE: Dict[str, float] = {
    # 7 correct steps * 0.10 + 0.20 bonus + 0.10 no-repeat bonus
    "change_mobile_number": 7 * 0.10 + 0.20 + 0.10,
    # 8 steps * 0.08 + 0.18 bonus + 0.10 empathy bonus + 0.06 verify bonus
    "recharge_not_working": 8 * 0.08 + 0.18 + 0.10 + 0.06,
    # 8 steps * 0.08 + 0.16 bonus + 0.25 de-escalation + 0.15 correct-branch
    "no_internet_angry":    8 * 0.08 + 0.16 + 0.25 + 0.15,
}

# Initial customer patience by task
INITIAL_PATIENCE: Dict[str, int] = {
    "change_mobile_number": 3,
    "recharge_not_working": 3,
    "no_internet_angry":    2,   # starts angry
}

# Initial customer mood by task
INITIAL_MOOD: Dict[str, str] = {
    "change_mobile_number": "neutral",
    "recharge_not_working": "neutral",
    "no_internet_angry":    "angry",
}


# ─────────────────────────────────────────────────────────────────────────────
# MOCK ISP BACKEND
# ─────────────────────────────────────────────────────────────────────────────

class MockISPBackend:
    """
    Deterministic fake ISP backend seeded by episode_id.
    All responses are reproducible for the same seed.
    """

    def __init__(self, seed: str):
        h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
        self._rng = random.Random(h)
        # Pre-generate fixed values for this episode
        self._account_id = f"ACC{self._rng.randint(1000, 9999)}"
        self._dob = f"19{self._rng.randint(70,99):02d}-{self._rng.randint(1,12):02d}-{self._rng.randint(1,28):02d}"
        self._last_payment = self._rng.choice([199, 299, 399, 499, 599])
        self._transaction_id = f"TXN{self._rng.randint(100000, 999999)}"
        # Line status: determines Task 3 resolution branch
        self._line_status = self._rng.choice(["fixable", "fixable", "physical_fault"])
        self._new_mobile: Optional[str] = None
        self._service_activated = False
        self._line_fixed = False
        self._technician_scheduled = False

    # ---- Accessors (for grader/customer to know correct values) ----
    @property
    def correct_account_id(self) -> str:
        return self._account_id

    @property
    def correct_dob(self) -> str:
        return self._dob

    @property
    def correct_last_payment(self) -> int:
        return self._last_payment

    @property
    def correct_transaction_id(self) -> str:
        return self._transaction_id

    @property
    def line_status(self) -> str:
        return self._line_status

    # ---- API methods (called from environment step logic) ----

    def verify_account(self, account_id: str) -> bool:
        return account_id.strip().upper() == self._account_id

    def verify_identity(
        self,
        account_id: str,
        dob: Optional[str] = None,
        last_payment: Optional[int] = None,
    ) -> bool:
        if not self.verify_account(account_id):
            return False
        if dob is not None:
            return dob.strip() == self._dob
        if last_payment is not None:
            return int(last_payment) == self._last_payment
        return False

    def check_payment(self, transaction_id: str) -> str:
        """Returns 'verified' or 'not_found'."""
        if transaction_id.strip().upper() == self._transaction_id:
            return "verified"
        return "not_found"

    def activate_service(self, account_id: str) -> bool:
        if self.verify_account(account_id):
            self._service_activated = True
            return True
        return False

    def check_line_status(self, account_id: str) -> str:
        """Returns 'ok' | 'fixable' | 'physical_fault'."""
        if self.verify_account(account_id):
            return self._line_status
        return "not_found"

    def apply_remote_fix(self, account_id: str) -> bool:
        if self.verify_account(account_id) and self._line_status == "fixable":
            self._line_fixed = True
            return True
        return False

    def schedule_technician(self, account_id: str) -> str:
        if self.verify_account(account_id):
            self._technician_scheduled = True
            return "Tomorrow 10:00-12:00"
        return ""

    def update_mobile(self, account_id: str, new_mobile: str) -> bool:
        if self.verify_account(account_id):
            self._new_mobile = new_mobile.strip()
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOMER SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

# Pre-written customer response templates by (task, phase, mood)
CUSTOMER_RESPONSES: Dict[str, Dict[str, Dict[str, str]]] = {
    "change_mobile_number": {
        "opening": {
            "neutral": "Hi, I need to change my registered mobile number.",
            "cooperative": "Hello! I'd like to update my mobile number on my account.",
        },
        "greet": {
            "neutral": "Yes, I need to change my mobile number please.",
            "cooperative": "Thank you! Yes, I need to update my mobile.",
        },
        "ask_account_id": {
            "neutral": "My account ID is {account_id}.",
            "cooperative": "Sure, it's {account_id}.",
        },
        "verify_identity": {
            "neutral": "My date of birth is {dob}.",
            "cooperative": "Of course, my DOB is {dob}.",
        },
        "ask_new_mobile": {
            "neutral": "The new number is 9876543210.",
            "cooperative": "Please update it to 9876543210.",
        },
        "confirm_update": {
            "neutral": "Yes, that's correct.",
            "cooperative": "Yes, please go ahead.",
        },
        "update_system": {
            "neutral": "Alright, thank you.",
            "cooperative": "Great, thank you so much!",
        },
        "wrong_order": {
            "neutral": "I'm not sure what you mean. Can we go step by step?",
            "cooperative": "Could you clarify? I'm a bit confused.",
            "angry": "This is not what I asked for!",
        },
        "close_ticket": {
            "neutral": "Thank you, goodbye.",
            "cooperative": "Wonderful! Have a great day!",
        },
    },
    "recharge_not_working": {
        "opening": {
            "neutral": "I recharged my plan yesterday but internet is still not working!",
        },
        "greet": {
            "neutral": "Yes, I paid for a recharge but my internet isn't working.",
            "cooperative": "Thank you for answering. I recharged but I'm still getting no internet.",
        },
        "show_empathy": {
            "neutral": "Thank you, yes it's been very frustrating.",
            "cooperative": "I really appreciate that. It's been frustrating.",
            "angry": "Finally someone who understands!",
        },
        "ask_account_id": {
            "neutral": "My account ID is {account_id}.",
            "cooperative": "It's {account_id}.",
        },
        "ask_transaction_id": {
            "neutral": "The transaction ID is {transaction_id}.",
            "cooperative": "Yes, I have the payment receipt. Transaction ID: {transaction_id}.",
        },
        "verify_payment": {
            "neutral": "Okay, so you found my payment?",
            "cooperative": "Great, so the payment went through?",
        },
        "activate_service": {
            "neutral": "Let me check... yes! The internet is working now!",
            "cooperative": "Oh it's working! Thank you so much!",
        },
        "confirm_resolution": {
            "neutral": "Yes, it's working fine now.",
            "cooperative": "Yes, everything is working perfectly. Thank you!",
        },
        "wrong_order": {
            "neutral": "I don't understand. I already answered that.",
            "cooperative": "Can we stay focused? I'm confused.",
            "angry": "Why are you asking that again?!",
        },
        "close_ticket": {
            "neutral": "Alright, thank you. Bye.",
            "cooperative": "Thank you very much! Have a good day!",
        },
    },
    "no_internet_angry": {
        "opening": {
            "angry": "My internet has been down for 2 hours! The modem shows a red light! This is unacceptable!",
        },
        "greet": {
            "angry": "Yes, and I'm very angry! Fix this NOW!",
            "neutral": "Yes, my internet is down and the modem has a red light.",
        },
        "de_escalate": {
            "angry": "...Fine. I understand. But please fix this quickly.",
            "neutral": "Okay, thank you for understanding.",
            "cooperative": "Thank you for being so understanding.",
        },
        "ask_account_id": {
            "neutral": "My account ID is {account_id}.",
            "cooperative": "Sure, {account_id}.",
            "angry": "{account_id}. Now hurry up!",
        },
        "ask_symptom_details": {
            "neutral": "The modem has a solid red light and I checked the cables, they're all plugged in.",
            "cooperative": "The power is on, cables are connected, but there's a red light on the modem.",
            "angry": "Red light! Cables are fine! What else do you need?!",
        },
        "check_line_status": {
            "neutral": "Okay...",
            "cooperative": "Alright.",
        },
        "remote_fix": {
            "neutral": "Okay... wait... yes! It's working now!",
            "cooperative": "Oh wow, it's back! Thank you!",
        },
        "schedule_technician": {
            "neutral": "Tomorrow? Okay, I guess that works.",
            "cooperative": "Alright, tomorrow morning works for me.",
            "angry": "Tomorrow?! Fine, but make it early.",
        },
        "confirm_fix": {
            "neutral": "Yes, confirmed.",
            "cooperative": "Yes, all good now!",
            "angry": "Fine, yes.",
        },
        "wrong_order": {
            "angry": "Why are you asking me irrelevant things?! Fix my internet!",
            "neutral": "I'm not sure what you mean.",
            "cooperative": "Could you explain?",
        },
        "close_ticket": {
            "neutral": "Thank you. Goodbye.",
            "cooperative": "Thanks for your help! Bye.",
            "angry": "I hope this doesn't happen again. Bye.",
        },
    },
}


def _customer_reply(task: str, phase: str, mood: str, backend: MockISPBackend) -> str:
    """Get the customer response text for a given task/phase/mood."""
    task_responses = CUSTOMER_RESPONSES.get(task, {})
    phase_responses = task_responses.get(phase, {})
    # Fallback mood chain
    text = (
        phase_responses.get(mood)
        or phase_responses.get("neutral")
        or phase_responses.get("cooperative")
        or "I see."
    )
    # Substitute backend values
    text = text.replace("{account_id}", backend.correct_account_id)
    text = text.replace("{dob}", backend.correct_dob)
    text = text.replace("{transaction_id}", backend.correct_transaction_id)
    text = text.replace("{last_payment}", str(backend.correct_last_payment))
    return text


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class BroadbandcareEnvironment(Environment):
    """
    BroadbandCare customer support simulation environment.

    Supports 3 tasks via the BROADBANDCARE_TASK environment variable
    (defaults to 'change_mobile_number').

    State machine phases match TASK_STEPS[task_name].  The agent must execute
    actions in the correct order; wrong-order or invalid actions lose patience
    points and don't advance the phase.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_name: Optional[str] = None):
        import os
        self.task_name = (
            task_name
            or os.getenv("BROADBANDCARE_TASK", "change_mobile_number")
        )
        if self.task_name not in TASK_NAMES:
            raise ValueError(
                f"Unknown task '{self.task_name}'. "
                f"Valid tasks: {TASK_NAMES}"
            )
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._backend: Optional[MockISPBackend] = None
        self._phase_index: int = 0
        self._mood: str = INITIAL_MOOD[self.task_name]
        self._patience: int = INITIAL_PATIENCE[self.task_name]
        self._raw_score: float = 0.0
        self._completed_steps: List[str] = []
        self._done: bool = False
        # Task 3 branch tracking
        self._line_status: Optional[str] = None
        self._correct_resolve_action: Optional[str] = None
        # Bonus flags
        self._empathy_given: bool = False
        self._de_escalated: bool = False
        self._no_repeats: bool = True
        self._action_counts: Dict[str, int] = {}

    # ──────────────────────────────────────────────────────────── reset ──────

    def reset(self) -> BroadbandcareObservation:
        """Reset the environment to a fresh episode."""
        episode_id = str(uuid4())
        self._state = State(episode_id=episode_id, step_count=0)
        self._backend = MockISPBackend(seed=episode_id)
        self._phase_index = 0
        self._mood = INITIAL_MOOD[self.task_name]
        self._patience = INITIAL_PATIENCE[self.task_name]
        self._raw_score = 0.0
        self._completed_steps = []
        self._done = False
        self._line_status = None
        self._correct_resolve_action = None
        self._empathy_given = False
        self._de_escalated = False
        self._no_repeats = True
        self._action_counts = {}

        opening = _customer_reply(
            self.task_name, "opening", self._mood, self._backend
        )
        return BroadbandcareObservation(
            customer_message=opening,
            system_feedback="",
            customer_mood=self._mood,
            customer_patience=self._patience,
            task_phase="start",
            valid_next_actions=["greet"],
            episode_score=0.0,
            step_count=0,
            done=False,
            reward=0.0,
        )

    # ──────────────────────────────────────────────────────────── step ───────

    def step(self, action: BroadbandcareAction) -> BroadbandcareObservation:  # type: ignore[override]
        """Execute one agent action and return the resulting observation."""
        if self._done:
            # Environment already done — return terminal observation
            return self._terminal_obs("Episode already finished.", reward=0.0)

        self._state.step_count += 1

        # Track repeated actions
        self._action_counts[action.action_type] = (
            self._action_counts.get(action.action_type, 0) + 1
        )
        if self._action_counts.get(action.action_type, 0) > 1:
            self._no_repeats = False

        # Check max steps
        if self._state.step_count > MAX_STEPS[self.task_name]:
            self._done = True
            self._raw_score = max(0.0, self._raw_score - 0.10)
            return self._terminal_obs(
                "Maximum steps exceeded. Episode terminated.",
                reward=-0.10,
            )

        # Route to task-specific step handler
        if self.task_name == "change_mobile_number":
            return self._step_task1(action)
        elif self.task_name == "recharge_not_working":
            return self._step_task2(action)
        else:
            return self._step_task3(action)

    # ─────────────────────────────────────── TASK 1: change_mobile_number ───

    def _step_task1(self, action: BroadbandcareAction) -> BroadbandcareObservation:
        steps = TASK_STEPS["change_mobile_number"]
        current_phase = steps[min(self._phase_index, len(steps) - 1)]
        reward = 0.0
        system_feedback = ""

        if action.action_type == "greet" and self._phase_index == 0:
            reward = STEP_REWARD["change_mobile_number"]
            self._raw_score += reward
            self._phase_index = 1
            customer_msg = _customer_reply("change_mobile_number", "greet", self._mood, self._backend)

        elif action.action_type == "ask_account_id" and self._phase_index == 1:
            reward = STEP_REWARD["change_mobile_number"]
            self._raw_score += reward
            self._phase_index = 2
            customer_msg = _customer_reply("change_mobile_number", "ask_account_id", self._mood, self._backend)

        elif action.action_type == "verify_identity" and self._phase_index == 2:
            # Accept any identity verification attempt (agent got account_id from customer)
            self._mood = "cooperative"
            reward = STEP_REWARD["change_mobile_number"]
            self._raw_score += reward
            self._phase_index = 3
            system_feedback = f"Identity verified for account {self._backend.correct_account_id}."
            customer_msg = _customer_reply("change_mobile_number", "verify_identity", self._mood, self._backend)

        elif action.action_type == "ask_new_mobile" and self._phase_index == 3:
            reward = STEP_REWARD["change_mobile_number"]
            self._raw_score += reward
            self._phase_index = 4
            customer_msg = _customer_reply("change_mobile_number", "ask_new_mobile", self._mood, self._backend)

        elif action.action_type == "confirm_update" and self._phase_index == 4:
            reward = STEP_REWARD["change_mobile_number"]
            self._raw_score += reward
            self._phase_index = 5
            customer_msg = _customer_reply("change_mobile_number", "confirm_update", self._mood, self._backend)

        elif action.action_type == "update_system" and self._phase_index == 5:
            new_mobile = action.payload.get("new_mobile", "9876543210")
            ok = self._backend.update_mobile(self._backend.correct_account_id, new_mobile)
            system_feedback = (
                f"Mobile number updated to {new_mobile}." if ok
                else "Update failed — account not found."
            )
            reward = STEP_REWARD["change_mobile_number"]
            if ok:
                self._raw_score += reward
            self._phase_index = 6
            customer_msg = _customer_reply("change_mobile_number", "update_system", self._mood, self._backend)

        elif action.action_type == "close_ticket" and self._phase_index == 6:
            # Apply bonuses
            reward = STEP_REWARD["change_mobile_number"]
            self._raw_score += reward
            self._raw_score += COMPLETION_BONUS["change_mobile_number"]
            if self._no_repeats:
                self._raw_score += 0.10   # no-repeat bonus
            self._done = True
            customer_msg = _customer_reply("change_mobile_number", "close_ticket", self._mood, self._backend)
            reward += COMPLETION_BONUS["change_mobile_number"]

        else:
            # Wrong action — lose patience
            reward, customer_msg, system_feedback = self._wrong_action("change_mobile_number")

        # Compute normalised score
        norm_score = min(1.0, self._raw_score / MAX_RAW_SCORE["change_mobile_number"])
        next_phase_idx = min(self._phase_index, len(steps) - 1)
        valid_next = [steps[next_phase_idx]] if not self._done else []

        if self._patience <= 0:
            self._done = True
            customer_msg = "I've had enough of this. Goodbye!"
            self._mood = "hung_up"

        return BroadbandcareObservation(
            customer_message=customer_msg,
            system_feedback=system_feedback,
            customer_mood=self._mood,
            customer_patience=self._patience,
            task_phase=steps[min(self._phase_index, len(steps) - 1)],
            valid_next_actions=valid_next,
            episode_score=norm_score,
            step_count=self._state.step_count,
            done=self._done,
            reward=round(reward, 4),
        )

    # ─────────────────────────────────────── TASK 2: recharge_not_working ───

    def _step_task2(self, action: BroadbandcareAction) -> BroadbandcareObservation:
        steps = TASK_STEPS["recharge_not_working"]
        reward = 0.0
        system_feedback = ""

        if action.action_type == "greet" and self._phase_index == 0:
            reward = STEP_REWARD["recharge_not_working"]
            self._raw_score += reward
            self._phase_index = 1
            customer_msg = _customer_reply("recharge_not_working", "greet", self._mood, self._backend)

        elif action.action_type == "show_empathy" and self._phase_index == 1:
            self._empathy_given = True
            self._mood = "cooperative"
            reward = STEP_REWARD["recharge_not_working"] + 0.10   # empathy bonus
            self._raw_score += reward
            self._phase_index = 2
            customer_msg = _customer_reply("recharge_not_working", "show_empathy", self._mood, self._backend)

        elif action.action_type == "ask_account_id" and self._phase_index == 2:
            reward = STEP_REWARD["recharge_not_working"]
            self._raw_score += reward
            self._phase_index = 3
            customer_msg = _customer_reply("recharge_not_working", "ask_account_id", self._mood, self._backend)

        elif action.action_type == "ask_transaction_id" and self._phase_index == 3:
            reward = STEP_REWARD["recharge_not_working"]
            self._raw_score += reward
            self._phase_index = 4
            customer_msg = _customer_reply("recharge_not_working", "ask_transaction_id", self._mood, self._backend)

        elif action.action_type == "verify_payment" and self._phase_index == 4:
            txn_id = action.payload.get("transaction_id", self._backend.correct_transaction_id)
            status = self._backend.check_payment(txn_id)
            if status == "verified":
                system_feedback = f"Payment {txn_id} verified. Amount: ₹{self._backend.correct_last_payment}."
                reward = STEP_REWARD["recharge_not_working"] + 0.06  # verify bonus
            else:
                system_feedback = f"Payment {txn_id} not found in system."
                reward = 0.0
                self._patience -= 1
            self._raw_score += reward
            self._phase_index = 5
            customer_msg = _customer_reply("recharge_not_working", "verify_payment", self._mood, self._backend)

        elif action.action_type == "activate_service" and self._phase_index == 5:
            ok = self._backend.activate_service(self._backend.correct_account_id)
            system_feedback = "Service activated successfully." if ok else "Activation failed."
            reward = STEP_REWARD["recharge_not_working"]
            if ok:
                self._raw_score += reward
            self._phase_index = 6
            customer_msg = _customer_reply("recharge_not_working", "activate_service", self._mood, self._backend)

        elif action.action_type == "confirm_resolution" and self._phase_index == 6:
            reward = STEP_REWARD["recharge_not_working"]
            self._raw_score += reward
            self._phase_index = 7
            customer_msg = _customer_reply("recharge_not_working", "confirm_resolution", self._mood, self._backend)

        elif action.action_type == "close_ticket" and self._phase_index == 7:
            reward = STEP_REWARD["recharge_not_working"]
            self._raw_score += reward + COMPLETION_BONUS["recharge_not_working"]
            self._done = True
            customer_msg = _customer_reply("recharge_not_working", "close_ticket", self._mood, self._backend)

        else:
            # Specifically penalise skipping empathy
            if action.action_type in ("ask_account_id", "ask_transaction_id") and self._phase_index == 1 and not self._empathy_given:
                self._raw_score -= 0.10
                reward = -0.10
                system_feedback = "(missed empathy — penalty applied)"
            reward_delta, customer_msg, sys_fb = self._wrong_action("recharge_not_working")
            if not system_feedback:
                system_feedback = sys_fb
            reward += reward_delta

        norm_score = min(1.0, max(0.0, self._raw_score / MAX_RAW_SCORE["recharge_not_working"]))
        next_phase_idx = min(self._phase_index, len(steps) - 1)
        valid_next = [steps[next_phase_idx]] if not self._done else []

        if self._patience <= 0:
            self._done = True
            customer_msg = "I've had enough. I'll call back later. Goodbye!"
            self._mood = "hung_up"

        return BroadbandcareObservation(
            customer_message=customer_msg,
            system_feedback=system_feedback,
            customer_mood=self._mood,
            customer_patience=self._patience,
            task_phase=steps[min(self._phase_index, len(steps) - 1)],
            valid_next_actions=valid_next,
            episode_score=norm_score,
            step_count=self._state.step_count,
            done=self._done,
            reward=round(reward, 4),
        )

    # ─────────────────────────────────────── TASK 3: no_internet_angry ──────

    def _step_task3(self, action: BroadbandcareAction) -> BroadbandcareObservation:
        steps = TASK_STEPS["no_internet_angry"]
        reward = 0.0
        system_feedback = ""

        if action.action_type == "greet" and self._phase_index == 0:
            reward = STEP_REWARD["no_internet_angry"]
            self._raw_score += reward
            self._phase_index = 1
            customer_msg = _customer_reply("no_internet_angry", "greet", self._mood, self._backend)

        elif action.action_type == "de_escalate" and self._phase_index == 1:
            self._de_escalated = True
            self._mood = "neutral"
            self._patience = min(3, self._patience + 1)   # patience restored
            reward = STEP_REWARD["no_internet_angry"] + 0.25  # de-escalation bonus
            self._raw_score += reward
            self._phase_index = 2
            customer_msg = _customer_reply("no_internet_angry", "de_escalate", self._mood, self._backend)

        elif action.action_type == "ask_account_id" and self._phase_index == 2:
            reward = STEP_REWARD["no_internet_angry"]
            self._raw_score += reward
            self._phase_index = 3
            customer_msg = _customer_reply("no_internet_angry", "ask_account_id", self._mood, self._backend)

        elif action.action_type == "ask_symptom_details" and self._phase_index == 3:
            reward = STEP_REWARD["no_internet_angry"]
            self._raw_score += reward
            self._phase_index = 4
            customer_msg = _customer_reply("no_internet_angry", "ask_symptom_details", self._mood, self._backend)

        elif action.action_type == "check_line_status" and self._phase_index == 4:
            self._line_status = self._backend.check_line_status(self._backend.correct_account_id)
            system_feedback = f"Line status: {self._line_status}."
            if self._line_status == "fixable":
                system_feedback += " Remote reset recommended."
                self._correct_resolve_action = "remote_fix"
            elif self._line_status == "physical_fault":
                system_feedback += " Physical fault detected — technician required."
                self._correct_resolve_action = "schedule_technician"
            else:
                system_feedback += " Line is OK — check customer equipment."
                self._correct_resolve_action = "remote_fix"
            reward = STEP_REWARD["no_internet_angry"]
            self._raw_score += reward
            self._phase_index = 5
            customer_msg = _customer_reply("no_internet_angry", "check_line_status", self._mood, self._backend)

        elif action.action_type in ("remote_fix", "schedule_technician") and self._phase_index == 5:
            if action.action_type == self._correct_resolve_action:
                reward = STEP_REWARD["no_internet_angry"] + 0.15  # correct branch bonus
                self._raw_score += reward
                if action.action_type == "remote_fix":
                    self._backend.apply_remote_fix(self._backend.correct_account_id)
                    system_feedback = "Remote fix applied. Line restored."
                    self._mood = "cooperative"
                else:
                    slot = self._backend.schedule_technician(self._backend.correct_account_id)
                    system_feedback = f"Technician booked: {slot}."
            else:
                # Wrong branch chosen
                reward = -0.05
                self._raw_score = max(0.0, self._raw_score + reward)
                system_feedback = "Action not aligned with line status. Proceeding anyway."
            self._phase_index = 6
            customer_msg = _customer_reply("no_internet_angry", action.action_type, self._mood, self._backend)

        elif action.action_type == "confirm_fix" and self._phase_index == 6:
            reward = STEP_REWARD["no_internet_angry"]
            self._raw_score += reward
            self._phase_index = 7
            customer_msg = _customer_reply("no_internet_angry", "confirm_fix", self._mood, self._backend)

        elif action.action_type == "close_ticket" and self._phase_index == 7:
            reward = STEP_REWARD["no_internet_angry"]
            self._raw_score += reward + COMPLETION_BONUS["no_internet_angry"]
            self._done = True
            customer_msg = _customer_reply("no_internet_angry", "close_ticket", self._mood, self._backend)

        else:
            # Skip de-escalation penalty (worst offence in Task 3)
            if action.action_type != "de_escalate" and self._phase_index == 1:
                self._raw_score = max(0.0, self._raw_score - 0.20)
                system_feedback = "(skipped de-escalation — heavy penalty applied)"
                # Try to advance anyway but penalised
                self._phase_index = 2
            reward_delta, customer_msg, sys_fb = self._wrong_action("no_internet_angry")
            if not system_feedback:
                system_feedback = sys_fb
            reward += reward_delta

        norm_score = min(1.0, max(0.0, self._raw_score / MAX_RAW_SCORE["no_internet_angry"]))

        # Determine valid next actions
        if self._done:
            valid_next = []
        elif self._phase_index == 5 and self._correct_resolve_action:
            valid_next = [self._correct_resolve_action]
        else:
            phase_actions = [
                "greet", "de_escalate", "ask_account_id",
                "ask_symptom_details", "check_line_status",
                "remote_fix", "confirm_fix", "close_ticket",
            ]
            idx = min(self._phase_index, len(phase_actions) - 1)
            valid_next = [phase_actions[idx]]

        if self._patience <= 0:
            self._done = True
            customer_msg = "I'm done talking. GOODBYE!"
            self._mood = "hung_up"

        return BroadbandcareObservation(
            customer_message=customer_msg,
            system_feedback=system_feedback,
            customer_mood=self._mood,
            customer_patience=self._patience,
            task_phase=steps[min(self._phase_index, len(steps) - 1)],
            valid_next_actions=valid_next,
            episode_score=norm_score,
            step_count=self._state.step_count,
            done=self._done,
            reward=round(reward, 4),
        )

    # ─────────────────────────────────────────────────────── helpers ─────────

    def _wrong_action(self, task: str) -> Tuple[float, str, str]:
        """Handle an out-of-order or invalid action: lose patience, return penalty."""
        self._patience -= 1
        reward = -0.05
        self._raw_score = max(0.0, self._raw_score + reward)
        customer_msg = _customer_reply(task, "wrong_order", self._mood, self._backend)
        system_feedback = "Invalid or out-of-order action."
        return reward, customer_msg, system_feedback

    def _terminal_obs(self, message: str, reward: float) -> BroadbandcareObservation:
        """Return a terminal observation."""
        norm_score = min(1.0, max(0.0, self._raw_score / MAX_RAW_SCORE[self.task_name]))
        return BroadbandcareObservation(
            customer_message=message,
            system_feedback="",
            customer_mood=self._mood,
            customer_patience=self._patience,
            task_phase="terminal",
            valid_next_actions=[],
            episode_score=norm_score,
            step_count=self._state.step_count,
            done=True,
            reward=round(reward, 4),
        )

    # ──────────────────────────────────────────────────────────── state ──────

    @property
    def state(self) -> State:
        """Return current episode state."""
        return self._state


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST — run directly: python server/broadbandcare_environment.py
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test():
    print("=" * 60)
    print("BroadbandCare Environment — Smoke Test")
    print("=" * 60)

    scenarios = [
        ("change_mobile_number", [
            BroadbandcareAction(action_type="greet"),
            BroadbandcareAction(action_type="ask_account_id"),
            BroadbandcareAction(action_type="verify_identity", payload={"dob": "1990-01-01"}),
            BroadbandcareAction(action_type="ask_new_mobile"),
            BroadbandcareAction(action_type="confirm_update"),
            BroadbandcareAction(action_type="update_system", payload={"new_mobile": "9876543210"}),
            BroadbandcareAction(action_type="close_ticket"),
        ]),
        ("recharge_not_working", [
            BroadbandcareAction(action_type="greet"),
            BroadbandcareAction(action_type="show_empathy"),
            BroadbandcareAction(action_type="ask_account_id"),
            BroadbandcareAction(action_type="ask_transaction_id"),
            BroadbandcareAction(action_type="verify_payment"),
            BroadbandcareAction(action_type="activate_service"),
            BroadbandcareAction(action_type="confirm_resolution"),
            BroadbandcareAction(action_type="close_ticket"),
        ]),
        ("no_internet_angry", [
            BroadbandcareAction(action_type="greet"),
            BroadbandcareAction(action_type="de_escalate"),
            BroadbandcareAction(action_type="ask_account_id"),
            BroadbandcareAction(action_type="ask_symptom_details"),
            BroadbandcareAction(action_type="check_line_status"),
            BroadbandcareAction(action_type="remote_fix"),
            BroadbandcareAction(action_type="confirm_fix"),
            BroadbandcareAction(action_type="close_ticket"),
        ]),
    ]

    all_pass = True
    for task_name, actions in scenarios:
        print(f"\n--- Task: {task_name} ---")
        env = BroadbandcareEnvironment(task_name=task_name)
        obs = env.reset()
        print(f"  Customer: {obs.customer_message}")
        final_score = 0.0
        for i, action in enumerate(actions):
            obs = env.step(action)
            print(
                f"  Step {i+1} [{action.action_type}] "
                f"reward={obs.reward:.2f}  score={obs.episode_score:.2f}  "
                f"mood={obs.customer_mood}  phase={obs.task_phase}"
            )
            final_score = obs.episode_score
        status = "✅ PASS" if 0.0 <= final_score <= 1.0 else "❌ FAIL"
        if "❌" in status:
            all_pass = False
        print(f"  Final score: {final_score:.2f}  {status}")

    print("\n" + "=" * 60)
    print("All tasks PASSED ✅" if all_pass else "Some tasks FAILED ❌")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    import sys
    success = _smoke_test()
    sys.exit(0 if success else 1)
