#!/usr/bin/env python3
"""Policy/action training script for the travel agent.

Trains a next-action classifier from local dialogue JSON files that follow the
same schema as data/test_dialogues.json:

  {
    "turns": [
      {"speaker": "user", "text": "..."},
      {"speaker": "system", "text": "...", "action": "retrieve"}
    ],
    "gold_slots": {...},
    "gold_actions": ["retrieve", ...]
  }

The default training source is data/synthetic_travel_dialogues_200.json.

Usage:
  python3 notebooks/train_policy.py
  python3 notebooks/train_policy.py --input data/synthetic_travel_dialogues_200.json
"""

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


ACTIONS = (
  "ask_slot",
  "ask_optional",
  "retrieve",
  "confirm",
  "book",
  "done",
  "relax_constraints",
)

PROJECT_SLOTS = (
  "origin",
  "destination",
  "depart_date",
  "return_date",
  "budget_usd",
  "num_travelers",
  "preferences",
)

REQUIRED_SLOTS = (
  "origin",
  "destination",
  "depart_date",
  "num_travelers",
)

DEFAULT_INPUT = "data/synthetic_travel_dialogues_200.json"
DEFAULT_OUTPUT = "data/processed/policy_model.pkl"


def _norm(text: str) -> str:
  return re.sub(r"\s+", " ", text.strip().lower())


def _action_from_turn(turn: Dict[str, Any]) -> Optional[str]:
  action = turn.get("action")
  if action is None:
    return None
  if action not in ACTIONS:
    raise ValueError(f"Unknown action label: {action}")
  return action


def validate_dialogues(dialogues: List[dict]) -> None:
  """Validate that system-turn actions match gold_actions."""
  for dialogue in dialogues:
    dialogue_id = dialogue.get("id", "<unknown>")
    system_actions = [
      _action_from_turn(turn)
      for turn in dialogue.get("turns", [])
      if turn.get("speaker") == "system" and _action_from_turn(turn) is not None
    ]
    gold_actions = dialogue.get("gold_actions", [])
    if system_actions != gold_actions:
      raise ValueError(
        f"{dialogue_id}: system actions do not match gold_actions. "
        f"system={system_actions}, gold={gold_actions}"
      )


def _infer_state_for_action(gold_slots: Dict[str, Any], action: str) -> Dict[str, Any]:
  """Approximate slot-state features from final gold slots and target action.

  The synthetic dialogues label each system turn but only provide final slots.
  For action prediction, exact slot values are less important than whether the
  target action occurs before or after required info has been collected.
  """
  state: Dict[str, Any] = {}
  if action == "ask_slot":
    return state

  for slot in REQUIRED_SLOTS:
    value = gold_slots.get(slot)
    if value is not None:
      state[slot] = value

  if action in {"ask_optional", "retrieve", "confirm", "book", "done", "relax_constraints"}:
    for slot in ("return_date", "budget_usd", "preferences"):
      value = gold_slots.get(slot)
      if value not in (None, [], ""):
        state[slot] = value

  return state


def _state_feature_text(state: Dict[str, Any]) -> str:
  parts = []
  for slot in PROJECT_SLOTS:
    value = state.get(slot)
    if value in (None, "", []):
      parts.append(f"missing_{slot}")
      continue
    if isinstance(value, list):
      value_text = "_".join(str(v).lower().replace(" ", "_") for v in value)
    else:
      value_text = str(value).lower().replace(" ", "_")
    parts.append(f"{slot}={value_text}")

  missing_required = [slot for slot in REQUIRED_SLOTS if slot not in state]
  parts.append("missing_required_count=" + str(len(missing_required)))
  for slot in missing_required:
    parts.append(f"needs_{slot}")
  return " ".join(parts)


def _user_intent_features(last_user_msg: str) -> str:
  msg = _norm(last_user_msg)
  features = []
  if re.search(r"\b(yes|yeah|yep|book|confirm|go ahead|do it|sure|ok|okay)\b", msg):
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


def _feature_text(
  user_history: List[str],
  system_history: List[str],
  state: Dict[str, Any],
  previous_action: str,
  n_turns: int = 3,
) -> str:
  recent_user = " ".join(user_history[-n_turns:])
  recent_system = " ".join(system_history[-n_turns:])
  last_user = user_history[-1] if user_history else ""
  return (
    f"previous_action={previous_action or '<START>'} "
    f"state: {_state_feature_text(state)} "
    f"user_intent: {_user_intent_features(last_user)} "
    f"recent_user: {recent_user} "
    f"recent_system: {recent_system}"
  )


def build_examples(dialogues: List[dict]) -> Tuple[List[str], List[str]]:
  """Build one policy example for each labeled system turn."""
  texts: List[str] = []
  labels: List[str] = []

  for dialogue in dialogues:
    gold_slots = dialogue.get("gold_slots", {})
    user_history: List[str] = []
    system_history: List[str] = []
    previous_action = ""

    for turn in dialogue.get("turns", []):
      speaker = turn.get("speaker")
      text = turn.get("text", "")

      if speaker == "user":
        user_history.append(text)
        continue

      if speaker != "system":
        continue

      action = _action_from_turn(turn)
      if action is None:
        continue

      state = _infer_state_for_action(gold_slots, action)
      texts.append(_feature_text(
        user_history=user_history,
        system_history=system_history,
        state=state,
        previous_action=previous_action,
      ))
      labels.append(action)

      system_history.append(text)
      previous_action = action

  return texts, labels


def load_dialogues(path: str) -> List[dict]:
  with open(path) as f:
    data = json.load(f)
  if not isinstance(data, list):
    raise ValueError(f"Expected top-level list in {path}")
  validate_dialogues(data)
  return data


def _can_stratify(labels: List[str], test_size: float) -> bool:
  counts = Counter(labels)
  if min(counts.values()) < 2:
    return False
  n_test = int(round(len(labels) * test_size))
  return n_test >= len(counts)


def train(
  texts: List[str],
  labels: List[str],
  test_size: float = 0.2,
  random_state: int = 42,
) -> Tuple[TfidfVectorizer, LogisticRegression, List[str]]:
  """Fit a TF-IDF + LogisticRegression next-action classifier."""
  action_vocab = sorted(set(labels))
  y = [action_vocab.index(label) for label in labels]
  stratify = y if _can_stratify(labels, test_size) else None

  X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=stratify,
  )

  print(f"Fitting TF-IDF on {len(X_train_text)} training examples...")
  vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1, 2),
    min_df=1,
    strip_accents="unicode",
    lowercase=True,
  )
  X_train = vectorizer.fit_transform(X_train_text)
  X_test = vectorizer.transform(X_test_text)

  print(f"Training policy classifier ({len(action_vocab)} actions)...")
  clf = LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced")
  clf.fit(X_train, y_train)

  print(f"  train acc: {clf.score(X_train, y_train):.3f}")
  print(f"  test acc:  {clf.score(X_test, y_test):.3f}")
  print()
  print(classification_report(
    y_test,
    clf.predict(X_test),
    target_names=action_vocab,
    zero_division=0,
  ))

  return vectorizer, clf, action_vocab


def save_model(
  path: str,
  vectorizer: TfidfVectorizer,
  classifier: LogisticRegression,
  action_vocab: List[str],
  label_counts: Counter,
  input_path: str,
) -> None:
  """Serialize the policy model bundle to disk."""
  output_path = Path(path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  model = {
    "vectorizer": vectorizer,
    "classifier": classifier,
    "action_vocab": action_vocab,
    "label_counts": dict(label_counts),
    "actions": ACTIONS,
    "training_source": input_path,
  }
  with open(output_path, "wb") as f:
    pickle.dump(model, f)
  print(f"Saved model to {output_path}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Train policy model on local travel dialogues")
  parser.add_argument(
    "--input",
    default=DEFAULT_INPUT,
    help="Dialogue JSON file to train on",
  )
  parser.add_argument(
    "--output",
    default=DEFAULT_OUTPUT,
    help="Path to write the trained policy model pickle",
  )
  parser.add_argument(
    "--test-size",
    type=float,
    default=0.2,
    help="Held-out test fraction for reporting metrics",
  )
  args = parser.parse_args()

  print(f"Loading policy dialogues from {args.input}...")
  dialogues = load_dialogues(args.input)

  print("Building policy examples...")
  texts, labels = build_examples(dialogues)
  label_counts = Counter(labels)
  print(f"  {len(texts)} examples extracted")
  for label, count in sorted(label_counts.items()):
    print(f"  {label}: {count}")

  vectorizer, classifier, action_vocab = train(
    texts,
    labels,
    test_size=args.test_size,
  )
  save_model(
    args.output,
    vectorizer,
    classifier,
    action_vocab,
    label_counts,
    args.input,
  )


if __name__ == "__main__":
  main()
