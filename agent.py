"""Orchestration loop — changes require team sign-off.

Pipeline per turn:
  user input -> DST -> policy -> retrieval -> itinerary -> response
"""

import re
from typing import List, Optional

from modules.dst import update_state
from modules.itinerary import build_itinerary, simulate_booking
from modules.policy import decide_action
from modules.response import generate_response
from modules.retrieval import search_flights, search_hotels
from modules.state import DialogueState


_ORDINALS = {
  "first": 1,
  "second": 2,
  "third": 3,
  "fourth": 4,
  "fifth": 5,
}


def _choice_to_index(choice: str) -> Optional[int]:
  choice = choice.lower()
  if choice.isdigit():
    return int(choice) - 1
  if choice in _ORDINALS:
    return _ORDINALS[choice] - 1
  return None


def _find_option_choice(text: str, item_type: str) -> Optional[int]:
  token = r'(\d+|first|second|third|fourth|fifth)'
  patterns = (
    rf'\b(?:option\s+)?{token}\s+(?:for\s+)?(?:the\s+)?{item_type}\b',
    rf'\b{item_type}\s*(?:option\s+)?{token}\b',
    rf'\b{item_type}.*?\b(?:option\s+)?{token}\b',
  )
  for pattern in patterns:
    match = re.search(pattern, text, re.I)
    if match:
      return _choice_to_index(match.group(1))
  return None


def _find_name_choice(text: str, options: List[dict], name_fields: tuple[str, ...]) -> Optional[int]:
  text_lower = text.lower()
  text_tokens = set(re.findall(r'\b[a-z0-9]+\b', text_lower))
  for i, option in enumerate(options):
    for field in name_fields:
      value = str(option.get(field, "")).strip().lower()
      if value and value in text_lower:
        return i
      value_tokens = set(re.findall(r'\b[a-z0-9]+\b', value))
      if len(value_tokens & text_tokens) >= 2:
        return i
  return None


def _apply_user_selection(state: DialogueState, user_input: str, results: dict) -> None:
  flights = results.get("flights", [])
  hotels = results.get("hotels", [])
  if not flights and not hotels:
    return

  text = user_input.lower()
  token = r'(\d+|first|second|third|fourth|fifth)'
  both_match = re.search(rf'\b(?:option\s+)?{token}\s+for\s+both\b', text, re.I)
  both_idx = _choice_to_index(both_match.group(1)) if both_match else None

  flight_idx = _find_name_choice(text, flights, ("airline", "id"))
  if flight_idx is None:
    flight_idx = _find_option_choice(text, "flight")
  if flight_idx is None and both_idx is not None:
    flight_idx = both_idx

  hotel_idx = _find_name_choice(text, hotels, ("name", "id"))
  if hotel_idx is None:
    hotel_idx = _find_option_choice(text, "hotel")
  if hotel_idx is None and both_idx is not None:
    hotel_idx = both_idx

  if flight_idx is not None and flights:
    state.confirmed_flight = flights[flight_idx] if 0 <= flight_idx < len(flights) else flights[0]
  if hotel_idx is not None and hotels:
    state.confirmed_hotel = hotels[hotel_idx] if 0 <= hotel_idx < len(hotels) else hotels[0]


def run_agent(turns: Optional[List[str]] = None, prompt_optional: bool = True) -> Optional[dict]:
  """Run the travel planning agent interactively or in batch mode.

  Args:
    turns: If None, run interactively reading from stdin (current behavior,
      returns None). If a list of strings, feed them in sequence without user
      interaction and return a result dict.
    prompt_optional: If True, ask one optional follow-up for budget/preferences
      before retrieval.

  Returns:
    None in interactive mode. In batch mode:
      {
        "final_state": DialogueState,
        "transcript": list[dict],    # full [{role, content}, ...] history
        "action_sequence": list[str], # one action per user turn
        "task_completed": bool,       # True if final action == "done"
        "num_user_turns": int,
      }
  """
  state = DialogueState()
  state.optional_info_prompted = not prompt_optional
  history: List[dict] = []
  action_sequence: List[str] = []
  last_results: dict = {}

  def _run_turn(user_input: str) -> str:
    """Execute one pipeline turn and return the agent reply."""
    nonlocal state, last_results
    results: dict = {}

    history.append({"role": "user", "content": user_input})

    # Snapshot filled slots before DST update to detect user corrections.
    _tracked = ("origin", "destination", "depart_date", "return_date", "budget_usd", "num_travelers")
    prev_slots = {s: getattr(state, s) for s in _tracked}
    state = update_state(state, history)
    changed_slots = {s for s, v in prev_slots.items() if v is not None and getattr(state, s) != v}
    if changed_slots:
      state.confirmed_flight = None
      state.confirmed_hotel = None
      state.itinerary = []
      last_results = {}

    _apply_user_selection(state, user_input, last_results)

    last_action = action_sequence[-1] if action_sequence else ""
    action = decide_action(state, {}, changed_slots, user_input, last_action, history)
    if action == "ask_optional":
      state.optional_info_prompted = True

    if action == "retrieve":
      try:
        results["flights"] = search_flights(state)
        results["hotels"] = search_hotels(state)
      except NotImplementedError:
        results = {}
      last_results = results
      # Re-decide with retrieval results so no-results recovery can trigger.
      action = decide_action(state, results, changed_slots, user_input, last_action, history)

    action_sequence.append(action)

    if action == "confirm":
      try:
        state.itinerary = build_itinerary(
          state,
          [state.confirmed_flight] if state.confirmed_flight else [],
          [state.confirmed_hotel] if state.confirmed_hotel else [],
          [],
        )
      except NotImplementedError:
        pass

    if action == "book":
      try:
        results["confirmation"] = simulate_booking(state)
      except (NotImplementedError, ValueError):
        pass

    reply = generate_response(state, action, results)
    history.append({"role": "assistant", "content": reply})
    return reply

  if turns is None:
    # Interactive mode
    print("Travel Agent: Hi! I can help you plan a trip. Where will you be departing from?")
    while True:
      user_input = input("You: ").strip()
      if user_input.lower() in ("exit", "quit", "bye"):
        print("Travel Agent: Goodbye! Safe travels.")
        break
      if not user_input:
        continue
      reply = _run_turn(user_input)
      print(f"Travel Agent: {reply}")
    return None

  # Batch mode
  for user_input in turns:
    user_input = user_input.strip()
    if not user_input:
      continue
    _run_turn(user_input)
    if action_sequence and action_sequence[-1] == "done":
      break

  final_action = action_sequence[-1] if action_sequence else ""
  return {
    "final_state": state,
    "transcript": list(history),
    "action_sequence": list(action_sequence),
    "task_completed": final_action in ("book", "done"),
    "num_user_turns": sum(1 for t in history if t["role"] == "user"),
  }


if __name__ == "__main__":
  run_agent()
