"""Evaluation framework.

Metrics:
  DST:       slot accuracy, joint goal accuracy
  Retrieval: recall@k
  Policy:    action accuracy
  End-to-end: task completion rate, avg turns to completion
"""

from typing import List, Optional

_DST_SLOTS = ("origin", "destination", "depart_date", "return_date", "budget_usd", "num_travelers", "preferences")


def evaluate_dst(test_dialogues: List[dict]) -> dict:
  """Evaluate DST slot accuracy and joint goal accuracy.

  Replays each dialogue turn-by-turn through update_state(), then compares
  the final predicted DialogueState to the gold_slots labels.

  Args:
    test_dialogues: List of dicts with "turns" and "gold_slots" keys.
      Each turn has "speaker" ("user"/"system") and "text" fields.

  Returns:
    {
      "slot_accuracy": float | None,       # avg per-slot accuracy across all slots/dialogues
      "joint_goal_accuracy": float | None, # % dialogues where ALL slots match gold
      "per_slot_accuracy": dict | None,    # per-slot breakdown
    }
  """
  if not test_dialogues:
    return {"slot_accuracy": None, "joint_goal_accuracy": None, "per_slot_accuracy": None}

  from modules.dst import update_state
  from modules.state import DialogueState

  slot_correct = {s: 0 for s in _DST_SLOTS}
  joint_correct = 0

  for dialogue in test_dialogues:
    state = DialogueState()
    history: List[dict] = []
    gold_slots = dialogue.get("gold_slots", {})

    for turn in dialogue.get("turns", []):
      speaker = turn.get("speaker", "")
      role = "user" if speaker == "user" else "assistant"
      history.append({"role": role, "content": turn.get("text", "")})
      if speaker == "user":
        state = update_state(state, history)

    all_match = True
    for slot in _DST_SLOTS:
      gold_val = gold_slots.get(slot)
      pred_val = getattr(state, slot, None)

      if slot == "preferences":
        match = set(pred_val or []) == set(gold_val or [])
      elif slot in ("budget_usd", "num_travelers"):
        try:
          gold_int = int(gold_val) if gold_val is not None else None
          pred_int = int(pred_val) if pred_val is not None else None
        except (TypeError, ValueError):
          gold_int, pred_int = gold_val, pred_val
        match = pred_int == gold_int
      else:
        match = pred_val == gold_val

      if match:
        slot_correct[slot] += 1
      else:
        all_match = False

    if all_match:
      joint_correct += 1

  n = len(test_dialogues)
  per_slot = {s: slot_correct[s] / n for s in _DST_SLOTS}
  slot_accuracy = sum(slot_correct.values()) / (n * len(_DST_SLOTS))
  joint_goal_accuracy = joint_correct / n

  return {
    "slot_accuracy": slot_accuracy,
    "joint_goal_accuracy": joint_goal_accuracy,
    "per_slot_accuracy": per_slot,
  }


def evaluate_retrieval(test_dialogues: List[dict], k: int = 5) -> dict:
  """Evaluate retrieval recall@k.

  For each dialogue with complete gold_slots, constructs a DialogueState,
  calls search_flights / search_hotels, and records whether >=1 result is returned.

  Args:
    test_dialogues: List of test dialogue dicts with a "gold_slots" key.
    k: Number of top results to consider.

  Returns:
    {"flight_recall_at_k": float | None, "hotel_recall_at_k": float | None, "k": int}
  """
  from modules.retrieval import search_flights, search_hotels
  from modules.state import DialogueState

  flight_hits = flight_total = hotel_hits = hotel_total = 0

  for dlg in test_dialogues:
    gold = dlg.get("gold_slots", {})
    state = DialogueState(
      origin=gold.get("origin"),
      destination=gold.get("destination"),
      depart_date=gold.get("depart_date"),
      return_date=gold.get("return_date"),
      budget_usd=gold.get("budget_usd"),
      num_travelers=gold.get("num_travelers"),
      preferences=gold.get("preferences", []),
    )

    if not state.missing_slots():
      flight_total += 1
      try:
        if search_flights(state, k=k):
          flight_hits += 1
      except Exception:
        pass

    if state.destination is not None:
      hotel_total += 1
      try:
        if search_hotels(state, k=k):
          hotel_hits += 1
      except Exception:
        pass

  return {
    "flight_recall_at_k": flight_hits / flight_total if flight_total else None,
    "hotel_recall_at_k": hotel_hits / hotel_total if hotel_total else None,
    "k": k,
  }


def evaluate_policy(test_dialogues: List[dict]) -> dict:
  """Evaluate policy action accuracy against hand-labeled dialogues.

  Replays each dialogue turn-by-turn through update_state(), then calls
  decide_action() before each system turn and compares to gold_actions.

  Args:
    test_dialogues: List of dicts with "turns" and "gold_actions" keys.

  Returns:
    {"action_accuracy": float | None}
  """
  if not test_dialogues:
    return {"action_accuracy": None}

  from modules.dst import update_state
  from modules.policy import decide_action
  from modules.state import DialogueState

  correct = 0
  total = 0

  for dialogue in test_dialogues:
    state = DialogueState()
    history: List[dict] = []
    gold_actions = dialogue.get("gold_actions", [])
    action_idx = 0
    last_user_msg = ""
    last_action = ""

    for turn in dialogue.get("turns", []):
      speaker = turn.get("speaker", "")
      role = "user" if speaker == "user" else "assistant"
      history.append({"role": role, "content": turn.get("text", "")})

      if speaker == "user":
        last_user_msg = turn.get("text", "")
        state = update_state(state, history)
      else:
        if action_idx < len(gold_actions):
          predicted = decide_action(state, last_user_msg=last_user_msg, last_action=last_action)
          last_action = predicted
          if predicted == gold_actions[action_idx]:
            correct += 1
          total += 1
          action_idx += 1

  return {"action_accuracy": correct / total if total else None}


def evaluate_end_to_end(test_dialogues: List[dict]) -> dict:
  """Evaluate full agent on task completion and efficiency.

  Args:
    test_dialogues: List of dicts representing full conversations.

  Returns:
    {"task_completion_rate": float | None, "avg_turns": float | None}
  """
  if not test_dialogues:
    return {"task_completion_rate": None, "avg_turns": None}

  from agent import run_agent

  completed = 0
  completed_turns: List[int] = []

  for dialogue in test_dialogues:
    user_turns = [
      turn["text"]
      for turn in dialogue.get("turns", [])
      if turn.get("speaker") == "user"
    ]
    try:
      outcome = run_agent(turns=user_turns)
    except Exception:
      continue

    if outcome["task_completed"]:
      completed += 1
      completed_turns.append(outcome["num_user_turns"])

  task_completion_rate = completed / len(test_dialogues)
  avg_turns = sum(completed_turns) / len(completed_turns) if completed_turns else None
  return {"task_completion_rate": task_completion_rate, "avg_turns": avg_turns}


if __name__ == "__main__":
  import json
  from pathlib import Path

  _TEST_PATH = Path(__file__).parent / "data" / "test_dialogues.json"
  test_data: List[dict] = json.loads(_TEST_PATH.read_text()) if _TEST_PATH.exists() else []

  results = {
    "dst": evaluate_dst(test_data),
    "retrieval": evaluate_retrieval(test_data),
    "policy": evaluate_policy(test_data),
    "end_to_end": evaluate_end_to_end(test_data),
  }
  for module, metrics in results.items():
    print(f"{module}: {metrics}")
