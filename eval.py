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

  Args:
    test_dialogues: List of dicts with "state" and "gold_results" keys.
    k: Number of top results to consider.

  Returns:
    {"recall_at_k": float | None}
  """
  # TODO: Run search_flights/search_hotels per dialogue state,
  # check whether gold results appear in top-k returned.
  return {"recall_at_k": None}


def evaluate_policy(test_dialogues: List[dict]) -> dict:
  """Evaluate policy action accuracy against hand-labeled dialogues.

  Args:
    test_dialogues: List of dicts with "state" and "gold_action" keys.

  Returns:
    {"action_accuracy": float | None}
  """
  # TODO: Run decide_action per state, compare to gold_action.
  return {"action_accuracy": None}


def evaluate_end_to_end(test_dialogues: List[dict]) -> dict:
  """Evaluate full agent on task completion and efficiency.

  Args:
    test_dialogues: List of dicts representing full conversations.

  Returns:
    {"task_completion_rate": float | None, "avg_turns": float | None}
  """
  # TODO: Run run_agent (non-interactive mode) on each dialogue,
  # check whether booking was completed and count turns.
  return {"task_completion_rate": None, "avg_turns": None}


if __name__ == "__main__":
  import json
  from pathlib import Path

  _TEST_PATH = Path(__file__).parent / "data" / "test_dialogues.json"
  test_data: List[dict] = json.loads(_TEST_PATH.read_text()) if _TEST_PATH.exists() else []

  results = {
    "dst": evaluate_dst(test_data),
    "retrieval": evaluate_retrieval([]),
    "policy": evaluate_policy([]),
    "end_to_end": evaluate_end_to_end([]),
  }
  for module, metrics in results.items():
    print(f"{module}: {metrics}")
