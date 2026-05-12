"""Dialogue State Tracker (DST) — Alex Shen.

Approach: Option A (prompt-based LLM slot extraction).

Requires env vars (set in .env):
  LLM_API_KEY   — your OpenRouter or NVIDIA API key
  LLM_BASE_URL  — "https://openrouter.ai/api/v1"
                  or "https://integrate.api.nvidia.com/v1"
  LLM_MODEL     — e.g. "meta-llama/llama-3.3-8b-instruct:free" (OpenRouter)
                      or "meta/llama-3.3-8b-instruct" (NVIDIA)
"""

import json
import os
from typing import List

from modules.state import DialogueState

_SLOT_EXTRACTION_SYSTEM = """You are a travel booking assistant.
Extract travel slots from the conversation and return ONLY a JSON object.
Include only slots that are explicitly mentioned. Use null for unknown slots.
Schema:
{
  "origin": string or null,
  "destination": string or null,
  "depart_date": string (YYYY-MM-DD) or null,
  "return_date": string (YYYY-MM-DD) or null,
  "budget_usd": integer or null,
  "num_travelers": integer or null,
  "preferences": list of strings
}"""


def update_state(
  state: DialogueState,
  conversation_history: List[dict],
) -> DialogueState:
  """Extract slots from conversation history and merge into state.

  Args:
    state: Current dialogue state.
    conversation_history: List of {"role": str, "content": str} dicts.

  Returns:
    Updated DialogueState with any newly extracted slots filled in.
  """
  # TODO: implement — skeleton below shows the intended call pattern.
  #
  # from openai import OpenAI
  # client = OpenAI(
  #   api_key=os.environ["LLM_API_KEY"],
  #   base_url=os.environ["LLM_BASE_URL"],  # OpenRouter or NVIDIA endpoint
  # )
  # messages = [{"role": "system", "content": _SLOT_EXTRACTION_SYSTEM}] + conversation_history
  # response = client.chat.completions.create(
  #   model=os.environ["LLM_MODEL"],
  #   messages=messages,
  #   response_format={"type": "json_object"},
  # )
  # slots = json.loads(response.choices[0].message.content)
  # for key, val in slots.items():
  #   if val is not None and hasattr(state, key):
  #     setattr(state, key, val)
  return state
