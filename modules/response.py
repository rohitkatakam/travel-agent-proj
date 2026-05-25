"""Response generator.

Converts a (state, action, retrieval results) tuple into a natural-language reply.
"""

from modules.state import DialogueState

_SLOT_QUESTIONS = {
  "origin": "Where will you be departing from?",
  "destination": "Where would you like to travel to?",
  "depart_date": "What date would you like to depart? (YYYY-MM-DD)",
  "return_date": "When do you plan to return? (YYYY-MM-DD)",
  "budget_usd": "What is your total budget in USD?",
  "num_travelers": "How many travelers will be on this trip?",
}


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
  if action == "ask_slot":
    return _ask_slot(state)
  if action == "retrieve":
    return _present_options(results)
  if action == "confirm":
    return _confirm_trip(state)
  if action == "book":
    return _booking_done(results)
  if action == "done":
    return _wrap_up(state)
  return "I'm not sure how to help with that. Could you clarify?"


def _ask_slot(state: DialogueState) -> str:
  missing = state.missing_slots()
  if not missing:
    return "It looks like I have everything I need. Let me find some options for you."
  slot = missing[0]
  return _SLOT_QUESTIONS.get(slot, f"Could you tell me your {slot.replace('_', ' ')}?")


def _present_options(results: dict) -> str:
  lines = []

  flights = results.get("flights", [])
  if flights:
    lines.append("Here are some flight options:")
    for i, f in enumerate(flights[:3], 1):
      lines.append(f"  {i}. {f.get('airline', 'N/A')} — ${f.get('price_usd', 0):.0f}, departs {f.get('depart_date', 'N/A')}")
  else:
    lines.append("No flights found matching your criteria.")

  hotels = results.get("hotels", [])
  if hotels:
    lines.append("\nHere are some hotel options:")
    for i, h in enumerate(hotels[:3], 1):
      nights = h.get("num_nights", "?")
      lines.append(
        f"  {i}. {h.get('name', 'N/A')} — ${h.get('price_per_night_usd', 0):.0f}/night "
        f"({nights} nights), rated {h.get('rating', 'N/A')}"
      )
  else:
    lines.append("\nNo hotels found matching your criteria.")

  lines.append("\nWhich of these would you like to go with?")
  return "\n".join(lines)


def _confirm_trip(state: DialogueState) -> str:
  lines = ["Here's your trip summary:"]

  if state.confirmed_flight:
    f = state.confirmed_flight
    lines.append(
      f"  Flight: {f.get('airline', 'N/A')} — "
      f"{f.get('origin', '')} → {f.get('destination', '')}, "
      f"{f.get('depart_date', 'N/A')}, ${f.get('price_usd', 0):.0f}"
    )

  if state.confirmed_hotel:
    h = state.confirmed_hotel
    lines.append(
      f"  Hotel: {h.get('name', 'N/A')} in {h.get('city', 'N/A')}, "
      f"{h.get('num_nights', '?')} nights at ${h.get('price_per_night_usd', 0):.0f}/night"
    )

  lines.append("\nShall I go ahead and book this trip?")
  return "\n".join(lines)


def _booking_done(results: dict) -> str:
  conf = results.get("confirmation", {})
  booking_id = conf.get("booking_id", "N/A")
  return f"Your trip is booked! Confirmation ID: {booking_id}. Have a great trip!"


def _wrap_up(state: DialogueState) -> str:
  dest = state.destination or "your destination"
  return f"All done — enjoy your trip to {dest}!"
