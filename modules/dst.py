"""Dialogue State Tracker (DST).

Approach: Trained scikit-learn classifier for slot extraction.

Training runs on Colab using MultiWOZ 2.2 data (hotel, restaurant, attraction,
train domains only). The trained model is saved as a pickle at
`data/processed/dst_model.pkl` and loaded locally at inference time.

No external API calls or LLM prompts are used.
"""

import pickle
from pathlib import Path
from typing import List

from modules.state import DialogueState

_MODEL_PATH = Path(__file__).parent.parent / "data" / "processed" / "dst_model.pkl"


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
  # TODO: implement — skeleton below shows the intended call pattern.
  #
  # with open(_MODEL_PATH, "rb") as f:
  #   model = pickle.load(f)
  # features = _extract_features(conversation_history)
  # slots = model.predict(features)
  # for key, val in slots.items():
  #   if val is not None and hasattr(state, key):
  #     setattr(state, key, val)
  return state
