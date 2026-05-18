"""Response generator.

Converts a (state, action, retrieval results) tuple into a natural-language reply.
"""

from modules.state import DialogueState


def generate_response(
  state: DialogueState,
  action: str,
  results: dict,
) -> str:
  """Generate a natural-language response for the user.

  Args:
    state: Current dialogue state.
    action: Action decided by policy (e.g. "ask_slot", "retrieve").
    results: Dict containing retrieval results keyed by type
             (e.g. {"flights": [...], "hotels": [...]}).

  Returns:
    A string response to show the user.
  """
  # TODO: Implement response generation (template-based or LLM-based).
  return f"[stub response | action={action}]"
