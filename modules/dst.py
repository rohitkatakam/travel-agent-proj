"""Dialogue State Tracker (DST).

Approach: Trained scikit-learn classifier for slot extraction.

Training runs on Colab using MultiWOZ 2.2 data (hotel, restaurant, attraction,
train domains only). The trained model is saved as a pickle at
`data/processed/dst_model.pkl` and loaded locally at inference time.

No external API calls or LLM prompts are used.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.state import DialogueState

_MODEL_PATH = Path(__file__).parent.parent / "data" / "processed" / "dst_model.pkl"

_MODEL_CACHE: Dict[str, Any] = {}

_SLOT_NAMES = (
  "origin",
  "destination",
  "depart_date",
  "budget_usd",
  "num_travelers",
  "preferences",
)

_NONE_TOKEN = "<NONE>"


def _load_model() -> Dict[str, Any]:
  """Load the trained model bundle from disk, caching after first call.

  Returns:
    Dict with keys "vectorizer", "classifiers", "vocabularies".

  Raises:
    FileNotFoundError: If the model pickle does not exist yet.
  """
  if _MODEL_CACHE:
    return _MODEL_CACHE

  if not _MODEL_PATH.exists():
    raise FileNotFoundError(
      f"DST model not found at {_MODEL_PATH}. "
      "Run the Colab training notebook first to produce the model pickle."
    )

  with open(_MODEL_PATH, "rb") as f:
    model = pickle.load(f)

  _MODEL_CACHE["vectorizer"] = model["vectorizer"]
  _MODEL_CACHE["classifiers"] = model["classifiers"]
  _MODEL_CACHE["vocabularies"] = model["vocabularies"]
  return _MODEL_CACHE


def _extract_features(
  conversation_history: List[dict],
  vectorizer: Any,
  n_turns: int = 3,
) -> Any:
  """Concatenate the last n user turns and TF-IDF transform them.

  Args:
    conversation_history: List of {"role": str, "content": str} dicts.
    vectorizer: Fitted TfidfVectorizer from the trained model.
    n_turns: Number of most recent user turns to include.

  Returns:
    Sparse TF-IDF feature matrix (1 x n_features).
  """
  user_turns = [
    turn["content"]
    for turn in conversation_history
    if turn.get("role") == "user"
  ]
  text = " ".join(user_turns[-n_turns:]) if user_turns else ""
  return vectorizer.transform([text])


def _predict_slots(
  features: Any,
  classifiers: Dict[str, Any],
  vocabularies: Dict[str, List[str]],
) -> Dict[str, Optional[str]]:
  """Run each slot classifier and map predicted indices back to values.

  Args:
    features: Sparse TF-IDF feature matrix (1 x n_features).
    classifiers: Dict of slot_name -> LogisticRegression.
    vocabularies: Dict of slot_name -> list of possible values.

  Returns:
    Dict of slot_name -> predicted value (or None if predicted <NONE>).
  """
  predictions: Dict[str, Optional[str]] = {}

  for slot_name in _SLOT_NAMES:
    clf = classifiers[slot_name]
    vocab = vocabularies[slot_name]
    pred_idx = clf.predict(features)[0]
    pred_value = vocab[pred_idx]
    predictions[slot_name] = None if pred_value == _NONE_TOKEN else pred_value

  return predictions


def _merge_predictions(
  state: DialogueState,
  predictions: Dict[str, Optional[str]],
) -> DialogueState:
  """Merge non-None predicted slot values into the dialogue state.

  Args:
    state: Current dialogue state.
    predictions: Dict of slot_name -> predicted value or None.

  Returns:
    Updated DialogueState.
  """
  for slot_name, value in predictions.items():
    if value is not None and hasattr(state, slot_name):
      if slot_name == "num_travelers":
        try:
          value = int(value)
        except ValueError:
          continue
      if slot_name == "budget_usd":
        try:
          value = int(value)
        except ValueError:
          continue
      if slot_name == "preferences":
        if value not in state.preferences:
          state.preferences.append(value)
        continue
      setattr(state, slot_name, value)
  return state


def update_state(
  state: DialogueState,
  conversation_history: List[dict],
) -> DialogueState:
  """Extract slots from conversation history and merge into state.

  Loads the trained classifier from `data/processed/dst_model.pkl`,
  extracts features from the conversation text, predicts slot values,
  and merges any non-null predictions into the current DialogueState.

  Args:
    state: Current dialogue state.
    conversation_history: List of {"role": str, "content": str} dicts.

  Returns:
    Updated DialogueState with any newly extracted slots filled in.
  """
  model = _load_model()
  features = _extract_features(conversation_history, model["vectorizer"])
  predictions = _predict_slots(
    features,
    model["classifiers"],
    model["vocabularies"],
  )
  return _merge_predictions(state, predictions)
