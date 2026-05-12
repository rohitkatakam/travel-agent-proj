"""Policy/action module — Alan Wang.

Decides the next system action given the current dialogue state.
"""

from modules.state import DialogueState

ACTIONS = ("ask_slot", "retrieve", "confirm", "book", "done")


def decide_action(state: DialogueState) -> str:
  """Return the next action the agent should take.

  Returns one of: "ask_slot", "retrieve", "confirm", "book", "done".
  """
  # TODO: Implement full policy logic (rule-based or learned).
  if state.missing_slots():
    return "ask_slot"
  if state.confirmed_flight is None or state.confirmed_hotel is None:
    return "retrieve"
  if not state.itinerary:
    return "confirm"
  return "done"
