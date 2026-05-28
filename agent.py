"""Orchestration loop — changes require team sign-off.

Pipeline per turn:
  user input -> DST -> policy -> retrieval -> itinerary -> response
"""

from typing import List, Optional

from modules.dst import update_state
from modules.itinerary import build_itinerary, simulate_booking
from modules.policy import decide_action
from modules.response import generate_response
from modules.retrieval import search_flights, search_hotels
from modules.state import DialogueState


def run_agent(turns: Optional[List[str]] = None) -> Optional[dict]:
  """Run the travel planning agent interactively or in batch mode.

  Args:
    turns: If None, run interactively reading from stdin (current behavior,
      returns None). If a list of strings, feed them in sequence without user
      interaction and return a result dict.

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
  history: List[dict] = []
  action_sequence: List[str] = []

  def _run_turn(user_input: str) -> str:
    """Execute one pipeline turn and return the agent reply."""
    nonlocal state
    results: dict = {}

    history.append({"role": "user", "content": user_input})

    # Snapshot filled slots before DST update to detect user corrections.
    _tracked = ("origin", "destination", "depart_date", "return_date", "budget_usd", "num_travelers")
    prev_slots = {s: getattr(state, s) for s in _tracked}
    state = update_state(state, history)
    changed_slots = {s for s, v in prev_slots.items() if v is not None and getattr(state, s) != v}

    last_action = action_sequence[-1] if action_sequence else ""
    action = decide_action(state, {}, changed_slots, user_input, last_action)

    if action == "retrieve":
      try:
        results["flights"] = search_flights(state)
        results["hotels"] = search_hotels(state)
      except NotImplementedError:
        results = {}
      # Re-decide with retrieval results so no-results recovery can trigger.
      action = decide_action(state, results, changed_slots, user_input, last_action)

    action_sequence.append(action)

    if action == "confirm":
      try:
        state.itinerary = build_itinerary(
          state,
          results.get("flights", []),
          results.get("hotels", []),
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
    print("Travel Agent: Hi! I can help you plan a trip. Where would you like to go?")
    while True:
      user_input = input("You: ").strip()
      if user_input.lower() in ("exit", "quit", "bye"):
        print("Travel Agent: Goodbye! Safe travels.")
        break
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
    "task_completed": final_action == "done",
    "num_user_turns": sum(1 for t in history if t["role"] == "user"),
  }


if __name__ == "__main__":
  run_agent()
