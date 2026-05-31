"""Policy/action module.

Uses a trained next-action classifier when available, with deterministic guards
for states where only one action is valid.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.state import DialogueState

ACTIONS = ("ask_slot", "ask_optional", "retrieve", "confirm", "book", "done", "relax_constraints")

_MODEL_PATH = Path(__file__).parent.parent / "data" / "processed" / "policy_model.pkl"
_MODEL_CACHE: Dict[str, Any] = {}

_AFFIRMATIVE = frozenset({
  "yes", "yeah", "yep", "confirm", "book", "sure", "ok", "okay",
})

_AFFIRMATIVE_PHRASES = ("sounds good", "go ahead", "do it")

_BOOKING_SLOTS = frozenset({
  "origin", "destination", "depart_date", "return_date", "budget_usd", "num_travelers",
})

_PROJECT_SLOTS = (
  "origin",
  "destination",
  "depart_date",
  "return_date",
  "budget_usd",
  "num_travelers",
  "preferences",
)


def _load_policy_model() -> Optional[Dict[str, Any]]:
  """Load the trained policy bundle, returning None if unavailable."""
  if _MODEL_CACHE:
    return _MODEL_CACHE
  if not _MODEL_PATH.exists():
    return None

  try:
    with open(_MODEL_PATH, "rb") as f:
      model = pickle.load(f)
  except Exception:
    return None

  _MODEL_CACHE["vectorizer"] = model["vectorizer"]
  _MODEL_CACHE["classifier"] = model["classifier"]
  _MODEL_CACHE["action_vocab"] = model["action_vocab"]
  return _MODEL_CACHE


def _state_feature_text(state: DialogueState) -> str:
  """Serialize DialogueState using the same feature style as train_policy.py."""
  parts = []
  for slot in _PROJECT_SLOTS:
    value = getattr(state, slot)
    if value in (None, "", []):
      parts.append(f"missing_{slot}")
      continue
    if isinstance(value, list):
      value_text = "_".join(str(v).lower().replace(" ", "_") for v in value)
    else:
      value_text = str(value).lower().replace(" ", "_")
    parts.append(f"{slot}={value_text}")

  missing_required = state.missing_slots()
  parts.append("missing_required_count=" + str(len(missing_required)))
  for slot in missing_required:
    parts.append(f"needs_{slot}")
  return " ".join(parts)


def _user_intent_features(last_user_msg: str) -> str:
  msg = re.sub(r"\s+", " ", last_user_msg.strip().lower())
  features = []
  if _is_affirmative(msg):
    features.append("user_affirmative")
  if re.search(r"\b(no|not yet|wait|don't|do not|skip)\b", msg):
    features.append("user_negative")
  if re.search(r"\b(option|flight|hotel|first|second|third|\d+)\b", msg):
    features.append("user_selecting_option")
  if re.search(r"\b(change|actually|instead|meant|different)\b", msg):
    features.append("user_correction")
  if re.search(r"\b(budget|\$|pool|vegetarian|outdoor|luxury|cheap|affordable)\b", msg):
    features.append("user_optional_info")
  return " ".join(features)


def _history_parts(history: List[dict] | None, n_turns: int = 3) -> tuple[str, str]:
  if not history:
    return "", ""
  user_turns = [
    turn.get("content", "")
    for turn in history
    if turn.get("role") == "user"
  ]
  system_turns = [
    turn.get("content", "")
    for turn in history
    if turn.get("role") == "assistant"
  ]
  return " ".join(user_turns[-n_turns:]), " ".join(system_turns[-n_turns:])


def _feature_text(
  state: DialogueState,
  last_user_msg: str,
  last_action: str,
  history: List[dict] | None,
) -> str:
  recent_user, recent_system = _history_parts(history)
  return (
    f"previous_action={last_action or '<START>'} "
    f"state: {_state_feature_text(state)} "
    f"user_intent: {_user_intent_features(last_user_msg)} "
    f"recent_user: {recent_user} "
    f"recent_system: {recent_system}"
  )


def _predict_model_action(
  state: DialogueState,
  last_user_msg: str,
  last_action: str,
  history: List[dict] | None,
) -> Optional[str]:
  model = _load_policy_model()
  if model is None:
    return None

  features = model["vectorizer"].transform([
    _feature_text(state, last_user_msg, last_action, history)
  ])
  pred_idx = model["classifier"].predict(features)[0]
  return model["action_vocab"][pred_idx]


def _is_affirmative(msg_lower: str) -> bool:
  return (
    any(re.search(r'\b' + re.escape(word) + r'\b', msg_lower) for word in _AFFIRMATIVE)
    or any(p in msg_lower for p in _AFFIRMATIVE_PHRASES)
  )


def _rule_action(
  state: DialogueState,
  retrieval_results: dict,
  changed_slots: set[str],
  last_user_msg: str,
  last_action: str,
) -> str:
  """Deterministic fallback policy."""
  if last_action in ("book", "done"):
    return "done"

  if changed_slots and _BOOKING_SLOTS & changed_slots:
    if state.confirmed_flight is not None or state.confirmed_hotel is not None:
      state.confirmed_flight = None
      state.confirmed_hotel = None
      return "retrieve"

  if state.missing_slots():
    return "ask_slot"

  if not state.optional_info_prompted and state.budget_usd is None and not state.preferences:
    return "ask_optional"

  if state.confirmed_flight is None or state.confirmed_hotel is None:
    if retrieval_results:
      if not retrieval_results.get("flights") or not retrieval_results.get("hotels"):
        return "relax_constraints"
    return "retrieve"

  if not state.itinerary:
    return "confirm"

  if _is_affirmative(last_user_msg.lower()):
    return "book"

  return "confirm"


def _valid_actions(
  state: DialogueState,
  retrieval_results: dict,
  changed_slots: set[str],
  last_user_msg: str,
  last_action: str,
) -> set[str]:
  """Return actions that are possible from the current state."""
  if last_action in ("book", "done"):
    return {"done"}
  if state.missing_slots():
    return {"ask_slot"}
  if changed_slots and _BOOKING_SLOTS & changed_slots:
    return {"retrieve"}
  if retrieval_results and (not retrieval_results.get("flights") or not retrieval_results.get("hotels")):
    return {"relax_constraints"}
  if not state.optional_info_prompted and state.budget_usd is None and not state.preferences:
    return {"ask_optional"}
  if state.confirmed_flight is None or state.confirmed_hotel is None:
    return {"retrieve"}
  if not state.itinerary:
    return {"confirm"}
  if _is_affirmative(last_user_msg.lower()):
    return {"book"}
  return {"confirm"}


def decide_action(
  state: DialogueState,
  retrieval_results: dict | None = None,
  changed_slots: set[str] | None = None,
  last_user_msg: str = "",
  last_action: str = "",
  history: List[dict] | None = None,
) -> str:
  """Return the next action, using the trained policy model when valid."""
  if retrieval_results is None:
    retrieval_results = {}
  if changed_slots is None:
    changed_slots = set()

  fallback = _rule_action(state, retrieval_results, changed_slots, last_user_msg, last_action)
  valid = _valid_actions(state, retrieval_results, changed_slots, last_user_msg, last_action)
  model_action = _predict_model_action(state, last_user_msg, last_action, history)

  if model_action in valid:
    return model_action
  return fallback
