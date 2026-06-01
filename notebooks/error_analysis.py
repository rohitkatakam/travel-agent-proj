"""Detailed error analysis across all 20 test dialogues.

Produces data/processed/error_analysis.csv with columns:
  id, description, module, failure_type, predicted, expected, details

Also prints a summary of the top failure categories.
"""

import csv
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.dst import update_state
from modules.policy import decide_action
from modules.state import DialogueState
from agent import run_agent

TEST_PATH = Path(__file__).parent.parent / "data" / "test_dialogues.json"
OUT_PATH  = Path(__file__).parent.parent / "data" / "processed" / "error_analysis.csv"

_DST_SLOTS = ("origin", "destination", "depart_date", "return_date", "budget_usd", "num_travelers", "preferences")


def _slots_match(pred_state: DialogueState, gold: dict) -> dict[str, bool]:
  results = {}
  for slot in _DST_SLOTS:
    gold_val = gold.get(slot)
    pred_val = getattr(pred_state, slot, None)
    if slot == "preferences":
      results[slot] = set(pred_val or []) == set(gold_val or [])
    elif slot in ("budget_usd", "num_travelers"):
      try:
        results[slot] = int(pred_val or 0) == int(gold_val or 0)
      except (TypeError, ValueError):
        results[slot] = pred_val == gold_val
    else:
      results[slot] = pred_val == gold_val
  return results


def analyze(dialogues: List[dict]) -> List[dict]:
  rows = []

  for dlg in dialogues:
    dlg_id   = dlg["id"]
    desc     = dlg["description"]
    gold     = dlg.get("gold_slots", {})
    gold_acts = dlg.get("gold_actions", [])
    turns    = dlg.get("turns", [])

    # --- DST analysis ---
    state  = DialogueState()
    history: List[dict] = []
    for turn in turns:
      role = "user" if turn["speaker"] == "user" else "assistant"
      history.append({"role": role, "content": turn["text"]})
      if turn["speaker"] == "user":
        state = update_state(state, history)

    slot_results = _slots_match(state, gold)
    for slot, ok in slot_results.items():
      if not ok:
        pred_val = getattr(state, slot, None)
        gold_val = gold.get(slot)
        rows.append({
          "id":          dlg_id,
          "description": desc,
          "module":      "DST",
          "failure_type": "wrong_slot",
          "predicted":   str(pred_val),
          "expected":    str(gold_val),
          "details":     f"slot={slot}",
        })

    # --- Policy analysis ---
    state2   = DialogueState()
    history2: List[dict] = []
    act_idx  = 0
    last_act = ""
    last_usr = ""

    for turn in turns:
      role = "user" if turn["speaker"] == "user" else "assistant"
      history2.append({"role": role, "content": turn["text"]})
      if turn["speaker"] == "user":
        last_usr = turn["text"]
        state2 = update_state(state2, history2)
      else:
        if act_idx < len(gold_acts):
          predicted_act = decide_action(state2, last_user_msg=last_usr, last_action=last_act)
          last_act = predicted_act
          if predicted_act != gold_acts[act_idx]:
            rows.append({
              "id":          dlg_id,
              "description": desc,
              "module":      "Policy",
              "failure_type": "wrong_action",
              "predicted":   predicted_act,
              "expected":    gold_acts[act_idx],
              "details":     f"action_idx={act_idx}",
            })
          act_idx += 1

    # --- End-to-end analysis ---
    user_turns = [t["text"] for t in turns if t["speaker"] == "user"]
    try:
      outcome = run_agent(turns=user_turns, prompt_optional=False)
      if not outcome["task_completed"]:
        rows.append({
          "id":          dlg_id,
          "description": desc,
          "module":      "End-to-End",
          "failure_type": "task_not_completed",
          "predicted":   str(outcome["action_sequence"]),
          "expected":    "book or done",
          "details":     f"final_action={outcome['action_sequence'][-1] if outcome['action_sequence'] else 'none'}",
        })
    except Exception as e:
      rows.append({
        "id":          dlg_id,
        "description": desc,
        "module":      "End-to-End",
        "failure_type": "exception",
        "predicted":   "",
        "expected":    "no exception",
        "details":     str(e)[:120],
      })

  return rows


def print_summary(rows: List[dict]) -> None:
  from collections import Counter
  by_type = Counter(r["failure_type"] for r in rows)
  by_module = Counter(r["module"] for r in rows)
  by_slot = Counter(r["details"] for r in rows if r["module"] == "DST")

  print(f"\n{'='*55}")
  print(f"Error Analysis Summary  ({len(rows)} total failures)")
  print(f"{'='*55}")
  print("\nBy module:")
  for k, v in by_module.most_common():
    print(f"  {k:<16} {v:>3} failures")
  print("\nBy failure type:")
  for k, v in by_type.most_common():
    print(f"  {k:<24} {v:>3}")
  if by_slot:
    print("\nDST — most-missed slots:")
    for k, v in by_slot.most_common(5):
      print(f"  {k:<24} {v:>3}")

  # Policy confusion: predicted vs expected
  policy_rows = [r for r in rows if r["module"] == "Policy"]
  if policy_rows:
    confusion = Counter((r["expected"], r["predicted"]) for r in policy_rows)
    print("\nPolicy — top misclassifications (expected → predicted):")
    for (exp, pred), cnt in confusion.most_common(8):
      print(f"  {exp:<20} → {pred:<20}  ×{cnt}")


def main() -> None:
  with open(TEST_PATH) as f:
    dialogues = json.load(f)

  rows = analyze(dialogues)

  fieldnames = ["id", "description", "module", "failure_type", "predicted", "expected", "details"]
  with open(OUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

  print(f"Saved {len(rows)} error rows to {OUT_PATH}")
  print_summary(rows)


if __name__ == "__main__":
  main()
