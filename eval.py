"""Evaluation framework.

Metrics:
  DST:       slot accuracy, joint goal accuracy
  Retrieval: recall@k
  Policy:    action accuracy
  End-to-end: task completion rate, avg turns to completion
"""

from typing import List, Optional


def evaluate_dst(test_dialogues: List[dict]) -> dict:
  """Evaluate DST slot accuracy and joint goal accuracy.

  Args:
    test_dialogues: List of dicts with "turns" and "gold_state" keys.

  Returns:
    {"slot_accuracy": float | None, "joint_goal_accuracy": float | None}
  """
  # TODO: Run update_state on each turn, compare predicted state to gold_state.
  return {"slot_accuracy": None, "joint_goal_accuracy": None}


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
  results = {
    "dst": evaluate_dst([]),
    "retrieval": evaluate_retrieval([]),
    "policy": evaluate_policy([]),
    "end_to_end": evaluate_end_to_end([]),
  }
  for module, metrics in results.items():
    print(f"{module}: {metrics}")
