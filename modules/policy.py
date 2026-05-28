"""Policy/action module.

Decides the next system action given the current dialogue state.
"""

from __future__ import annotations

from modules.state import DialogueState

ACTIONS = ("ask_slot", "retrieve", "confirm", "book", "done", "relax_constraints")

_AFFIRMATIVE = frozenset({
  "yes", "yeah", "yep", "confirm", "book", "sure", "ok", "okay",
})

_AFFIRMATIVE_PHRASES = ("sounds good", "go ahead", "do it")

_BOOKING_SLOTS = frozenset({
  "origin", "destination", "depart_date", "return_date", "budget_usd", "num_travelers",
})


def decide_action(
  state: DialogueState,
  retrieval_results: dict | None = None,
  changed_slots: set[str] | None = None,
  last_user_msg: str = "",
  last_action: str = "",
) -> str:
  """Return the next action the agent should take.

  Returns one of: "ask_slot", "retrieve", "confirm", "book", "done", "relax_constraints".

  Args:
    state: Current dialogue state.
    retrieval_results: Results from the most recent retrieval call this turn,
      or None/{} if retrieval hasn't run yet.
    changed_slots: Slot names whose values changed this turn (user correction).
    last_user_msg: Raw user message for this turn (used for affirmative detection).
    last_action: The action returned last turn (used to advance past "book").
  """
  if retrieval_results is None:
    retrieval_results = {}

  # After a successful booking, wrap up.
  if last_action == "book":
    return "done"

  # User corrected a key booking slot after options were already confirmed — re-retrieve.
  if changed_slots and _BOOKING_SLOTS & changed_slots:
    if state.confirmed_flight is not None or state.confirmed_hotel is not None:
      state.confirmed_flight = None
      state.confirmed_hotel = None
      return "retrieve"

  if state.missing_slots():
    return "ask_slot"

  if state.confirmed_flight is None or state.confirmed_hotel is None:
    # Retrieval ran this turn but returned nothing — ask user to relax constraints.
    if retrieval_results:
      if not retrieval_results.get("flights") and not retrieval_results.get("hotels"):
        return "relax_constraints"
    return "retrieve"

  # Itinerary not yet built — show the confirm/summary prompt.
  if not state.itinerary:
    return "confirm"

  # Itinerary built — require explicit user confirmation before booking.
  msg_lower = last_user_msg.lower()
  words = set(msg_lower.split())
  if words & _AFFIRMATIVE or any(p in msg_lower for p in _AFFIRMATIVE_PHRASES):
    return "book"

  # Re-present the summary until the user confirms.
  return "confirm"
