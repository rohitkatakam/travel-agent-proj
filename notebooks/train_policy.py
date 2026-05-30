#!/usr/bin/env python3
"""Policy/action training script for the travel agent.

Trains a next-action classifier on MultiWOZ 2.2 system turns, mapping
MultiWOZ-style system utterances into this project's policy action space:

  ask_slot, retrieve, confirm, book, done, relax_constraints

The labels are weakly supervised from system utterance patterns. This is meant
to provide a learned policy component similar in spirit to train_dst.py, not a
drop-in replacement for the current rule policy.

Usage:
  python3 notebooks/train_policy.py --output data/processed/policy_model.pkl
"""

import argparse
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


ACTIONS = (
  "ask_slot",
  "retrieve",
  "confirm",
  "book",
  "done",
  "relax_constraints",
)

MULTIWOZ_DATASET_CANDIDATES = (
  ("multi_woz_v22", {}),
  ("pfb30/multi_woz_v22", {"data_dir": "v2.2"}),
)

TARGET_DOMAINS = {"hotel", "restaurant", "attraction", "train"}

PROJECT_SLOTS = (
  "origin",
  "destination",
  "depart_date",
  "budget_usd",
  "num_travelers",
  "preferences",
)

SLOT_SOURCES: Dict[str, List[Tuple[str, str]]] = {
  "origin":        [("train", "train-departure")],
  "destination":   [("train", "train-destination")],
  "depart_date":   [("train", "train-day")],
  "budget_usd":    [("hotel", "hotel-pricerange"), ("restaurant", "restaurant-pricerange")],
  "num_travelers": [("train", "train-bookpeople"), ("hotel", "hotel-bookpeople"), ("restaurant", "restaurant-bookpeople")],
  "preferences":   [("hotel", "hotel-type"), ("restaurant", "restaurant-food"), ("attraction", "attraction-type")],
}

PRICE_RANGE_TO_USD = {
  "cheap": "100",
  "moderate": "200",
  "expensive": "400",
}


def _iter_frame_services(frames: Dict) -> List[str]:
  """Return lowercase service/domain names from a MultiWOZ frames object."""
  return [str(s).lower() for s in frames.get("service", [])]


def _in_target_domain(frames: Dict) -> bool:
  """Return True if a turn has frames in a domain relevant to this project."""
  return bool(set(_iter_frame_services(frames)) & TARGET_DOMAINS)


def _get_slot_value(frames: Dict, slot: str) -> Optional[str]:
  """Extract a project slot value from MultiWOZ frame state if present."""
  services: List[str] = frames.get("service", [])
  states: List = frames.get("state", [])

  for domain, key in SLOT_SOURCES.get(slot, []):
    for j, service in enumerate(services):
      if str(service).lower() != domain:
        continue
      if j >= len(states):
        continue
      slot_values = states[j].get("slots_values", {})
      names: List[str] = slot_values.get("slots_values_name", [])
      values: List[List[str]] = slot_values.get("slots_values_list", [])
      for k, name in enumerate(names):
        if name == key and k < len(values) and values[k]:
          raw = values[k][0].strip().lower()
          if slot == "budget_usd":
            return PRICE_RANGE_TO_USD.get(raw)
          return raw
  return None


def _update_state_from_frames(state: Dict[str, str], frames: Dict) -> None:
  """Merge non-empty slot values from a user turn into a simple state dict."""
  for slot in PROJECT_SLOTS:
    value = _get_slot_value(frames, slot)
    if value:
      state[slot] = value


def _state_feature_text(state: Dict[str, str]) -> str:
  """Serialize the current slot state into text features for the classifier."""
  parts = []
  for slot in PROJECT_SLOTS:
    value = state.get(slot)
    if value:
      parts.append(f"{slot}={value}")
    else:
      parts.append(f"missing_{slot}")
  return " ".join(parts)


def _map_system_utterance_to_action(utterance: str) -> str:
  """Map a MultiWOZ system utterance into this project's action labels.

  MultiWOZ does not directly use our exact action names, so this function
  creates weak labels from wording patterns. Keep higher-specificity patterns
  before broader request/recommendation patterns.
  """
  text = utterance.lower()

  if re.search(r'\b(goodbye|bye|you(?:\'re| are) welcome|have a (?:nice|great) day)\b', text):
    return "done"

  if re.search(
    r'\b(reference|ref(?:erence)? number|booked|booking (?:was )?(?:successful|complete|confirmed)|'
    r'reservation (?:has been |is )?(?:made|confirmed|booked))\b',
    text,
  ):
    return "book"

  if re.search(
    r'\b(would you like (?:me )?to book|shall i book|should i book|'
    r'would you like to reserve|confirm(?: this)?|does that (?:work|sound good)|'
    r'is that (?:all|correct))\b',
    text,
  ):
    return "confirm"

  if re.search(
    r'\b(no (?:matching |available )?(?:option|options|result|results|train|hotel|restaurant|'
    r'attraction|booking)|not (?:available|found)|unable to find|fully booked|unfortunately)\b',
    text,
  ):
    return "relax_constraints"

  if (
    "?" in text
    or re.search(
      r'\b(what|where|when|how many|could you (?:please )?(?:tell|provide)|'
      r'can you (?:please )?(?:tell|provide)|do you have|would you prefer|'
      r'what (?:area|price|type|day|time))\b',
      text,
    )
  ):
    return "ask_slot"

  return "retrieve"


def _feature_text(
  user_history: List[str],
  system_history: List[str],
  state: Dict[str, str],
  previous_action: str,
  n_turns: int = 3,
) -> str:
  """Build classifier input from recent dialogue context and state."""
  recent_user = " ".join(user_history[-n_turns:])
  recent_system = " ".join(system_history[-n_turns:])
  return (
    f"previous_action={previous_action or '<START>'} "
    f"state: {_state_feature_text(state)} "
    f"recent_user: {recent_user} "
    f"recent_system: {recent_system}"
  )


def build_examples(dialogues) -> Tuple[List[str], List[str]]:
  """Extract policy training examples from MultiWOZ system turns.

  Returns:
    texts: Feature strings built from previous dialogue context and state.
    labels: One policy action label per feature string.
  """
  texts: List[str] = []
  labels: List[str] = []

  for dialogue in dialogues:
    turns = dialogue["turns"]
    user_history: List[str] = []
    system_history: List[str] = []
    state: Dict[str, str] = {}
    previous_action = ""

    for i, utterance in enumerate(turns["utterance"]):
      speaker = turns["speaker"][i]
      frames = turns["frames"][i]

      if speaker == 0:
        user_history.append(utterance)
        if _in_target_domain(frames):
          _update_state_from_frames(state, frames)
        continue

      if not user_history:
        continue
      if not _in_target_domain(frames) and not state:
        continue

      label = _map_system_utterance_to_action(utterance)
      texts.append(_feature_text(user_history, system_history, state, previous_action))
      labels.append(label)

      system_history.append(utterance)
      previous_action = label

  return texts, labels


def load_multiwoz_dataset():
  """Load MultiWOZ 2.2 from the first available Hugging Face source."""
  errors = []
  for dataset_name, kwargs in MULTIWOZ_DATASET_CANDIDATES:
    try:
      print(f"Trying dataset source: {dataset_name}")
      return load_dataset(dataset_name, trust_remote_code=True, **kwargs)
    except Exception as exc:
      errors.append(f"{dataset_name}: {exc}")

  raise RuntimeError(
    "Could not load MultiWOZ 2.2 from any configured source:\n"
    + "\n".join(errors)
  )


def train(
  texts: List[str],
  labels: List[str],
  test_size: float = 0.2,
  random_state: int = 42,
) -> Tuple[TfidfVectorizer, LogisticRegression, List[str]]:
  """Fit a TF-IDF + LogisticRegression next-action classifier."""
  action_vocab = sorted(set(labels))
  y = [action_vocab.index(label) for label in labels]

  X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y,
  )

  print(f"Fitting TF-IDF on {len(X_train_text)} training examples...")
  vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    strip_accents="unicode",
    lowercase=True,
  )
  X_train = vectorizer.fit_transform(X_train_text)
  X_test = vectorizer.transform(X_test_text)

  print(f"Training policy classifier ({len(action_vocab)} actions)...")
  clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
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
  }
  with open(output_path, "wb") as f:
    pickle.dump(model, f)
  print(f"Saved model to {output_path}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Train policy model on MultiWOZ 2.2")
  parser.add_argument(
    "--output",
    default="data/processed/policy_model.pkl",
    help="Path to write the trained policy model pickle",
  )
  parser.add_argument(
    "--test-size",
    type=float,
    default=0.2,
    help="Held-out test fraction for reporting metrics",
  )
  args = parser.parse_args()

  print("Loading MultiWOZ 2.2...")
  dataset = load_multiwoz_dataset()

  print("Building policy examples...")
  texts, labels = build_examples(dataset["train"])
  label_counts = Counter(labels)
  print(f"  {len(texts)} examples extracted")
  for label, count in sorted(label_counts.items()):
    print(f"  {label}: {count}")

  vectorizer, classifier, action_vocab = train(
    texts,
    labels,
    test_size=args.test_size,
  )
  save_model(args.output, vectorizer, classifier, action_vocab, label_counts)


if __name__ == "__main__":
  main()
