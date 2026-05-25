#!/usr/bin/env python3
"""DST training script for the travel agent.

Trains one LogisticRegression classifier per slot on MultiWOZ 2.2,
then saves a model bundle pickle compatible with modules/dst.py.

Usage:
  python3 notebooks/train_dst.py --output data/processed/dst_model.pkl
"""

import argparse
import pickle
from typing import Dict, List, Tuple

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SLOT_NAMES = (
  "origin",
  "destination",
  "depart_date",
  "budget_usd",
  "num_travelers",
  "preferences",
)

NONE_TOKEN = "<NONE>"

# (MultiWOZ domain, full slot key) sources per our slot, in priority order
SLOT_SOURCES: Dict[str, List[Tuple[str, str]]] = {
  "origin":        [("train", "train-departure")],
  "destination":   [("train", "train-destination")],
  "depart_date":   [("train", "train-day")],
  "budget_usd":    [("hotel", "hotel-pricerange"), ("restaurant", "restaurant-pricerange")],
  "num_travelers": [("train", "train-people"), ("hotel", "hotel-people"), ("restaurant", "restaurant-people")],
  "preferences":   [("hotel", "hotel-type"), ("restaurant", "restaurant-food"), ("attraction", "attraction-type")],
}

# Map MultiWOZ price range strings to int-parseable USD approximations
PRICE_RANGE_TO_USD: Dict[str, str] = {
  "cheap":     "100",
  "moderate":  "200",
  "expensive": "400",
}


def _get_slot_value(frames: List[Dict], slot: str) -> str:
  """Extract the first non-empty value for a slot from a turn's frames.

  Tries each source domain/key in priority order; returns NONE_TOKEN if
  nothing is found.
  """
  for domain, key in SLOT_SOURCES.get(slot, []):
    for frame in frames:
      if frame.get("service", "").lower() != domain:
        continue
      slot_values = frame.get("state", {}).get("slot_values", {})
      values = slot_values.get(key, [])
      if values:
        raw = values[0].strip().lower()
        if slot == "budget_usd":
          return PRICE_RANGE_TO_USD.get(raw, NONE_TOKEN)
        return raw
  return NONE_TOKEN


def build_examples(dialogues) -> Tuple[List[str], Dict[str, List[str]]]:
  """Extract (window_text, slot_labels) pairs from MultiWOZ dialogues.

  For each user turn, concatenates the last 3 user utterances as the
  feature string and reads the cumulative belief state for each slot.

  Returns:
    texts: List of feature strings.
    labels: Dict mapping slot_name -> list of label strings (parallel to texts).
  """
  texts: List[str] = []
  labels: Dict[str, List[str]] = {slot: [] for slot in SLOT_NAMES}

  for dialogue in dialogues:
    turns = dialogue["turns"]
    user_history: List[str] = []

    for i in range(len(turns["utterance"])):
      if turns["speaker"][i] != 0:  # skip system turns
        continue

      user_history.append(turns["utterance"][i])
      texts.append(" ".join(user_history[-3:]))

      frames = turns["frames"][i]
      for slot in SLOT_NAMES:
        labels[slot].append(_get_slot_value(frames, slot))

  return texts, labels


def build_vocabularies(labels: Dict[str, List[str]]) -> Dict[str, List[str]]:
  """Build sorted vocabulary list (including NONE_TOKEN) per slot."""
  return {slot: sorted(set(vals)) for slot, vals in labels.items()}


def train(
  texts: List[str],
  labels: Dict[str, List[str]],
  vocabularies: Dict[str, List[str]],
) -> Tuple[TfidfVectorizer, Dict[str, LogisticRegression]]:
  """Fit a TF-IDF vectorizer and one LogisticRegression per slot.

  Returns:
    vectorizer: Fitted TfidfVectorizer.
    classifiers: Dict of slot_name -> fitted LogisticRegression.
  """
  print(f"Fitting TF-IDF on {len(texts)} examples...")
  vectorizer = TfidfVectorizer(max_features=10000)
  X = vectorizer.fit_transform(texts)

  classifiers: Dict[str, LogisticRegression] = {}
  for slot in SLOT_NAMES:
    vocab = vocabularies[slot]
    y = [vocab.index(label) for label in labels[slot]]
    print(f"Training '{slot}' classifier ({len(vocab)} classes)...")
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, y)
    print(f"  train acc: {clf.score(X, y):.3f}")
    classifiers[slot] = clf

  return vectorizer, classifiers


def save_model(
  path: str,
  vectorizer: TfidfVectorizer,
  classifiers: Dict[str, LogisticRegression],
  vocabularies: Dict[str, List[str]],
) -> None:
  """Serialize the model bundle to a pickle file at path."""
  model = {
    "vectorizer":   vectorizer,
    "classifiers":  classifiers,
    "vocabularies": vocabularies,
  }
  with open(path, "wb") as f:
    pickle.dump(model, f)
  print(f"Saved model to {path}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Train DST model on MultiWOZ 2.2")
  parser.add_argument(
    "--output",
    default="data/processed/dst_model.pkl",
    help="Path to write the trained model pickle",
  )
  args = parser.parse_args()

  print("Loading MultiWOZ 2.2...")
  dataset = load_dataset("multi_woz_v22", trust_remote_code=True)

  print("Building training examples...")
  texts, labels = build_examples(dataset["train"])
  print(f"  {len(texts)} examples extracted")

  vocabularies = build_vocabularies(labels)
  for slot, vocab in vocabularies.items():
    print(f"  {slot}: {len(vocab)} unique values")

  vectorizer, classifiers = train(texts, labels, vocabularies)
  save_model(args.output, vectorizer, classifiers, vocabularies)


if __name__ == "__main__":
  main()
