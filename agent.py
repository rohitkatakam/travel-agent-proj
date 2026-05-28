"""Orchestration loop — changes require team sign-off.

Pipeline per turn:
  user input -> DST -> policy -> retrieval -> itinerary -> response
"""

from typing import List

from modules.dst import update_state
from modules.itinerary import build_itinerary, simulate_booking
from modules.policy import decide_action
from modules.response import generate_response
from modules.retrieval import search_flights, search_hotels
from modules.state import DialogueState


def run_agent() -> None:
  """Run the interactive travel planning agent."""
  state = DialogueState()
  history: List[dict] = []

  print("Travel Agent: Hi! I can help you plan a trip. Where would you like to go?")

  while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("exit", "quit", "bye"):
      print("Travel Agent: Goodbye! Safe travels.")
      break

    history.append({"role": "user", "content": user_input})

    # 1. DST: extract slots from conversation history
    state = update_state(state, history)

    # 2. Policy: decide next action
    action = decide_action(state)

    # 3. Retrieval: fetch options if needed
    results: dict = {}
    if action == "retrieve":
      # TODO: handle NotImplementedError until retrieval is implemented
      try:
        results["flights"] = search_flights(state)
        results["hotels"] = search_hotels(state)
      except NotImplementedError:
        results = {}

    # 4. Itinerary: build if confirming
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

    # 5. Booking: simulate if done
    if action == "book":
      confirmation = simulate_booking(state)
      results["confirmation"] = confirmation

    # 6. Response generation
    reply = generate_response(state, action, results)
    history.append({"role": "assistant", "content": reply})
    print(f"Travel Agent: {reply}")


def run_agent_batch(user_turns: List[str]) -> dict:
  """Run the agent pipeline on pre-supplied user turns (non-interactive).

  Mirrors run_agent() logic exactly, replacing stdin with the given turns.
  Stops when policy returns "done" or all turns are exhausted.

  Args:
    user_turns: Ordered list of user utterance strings.

  Returns:
    {"final_action": str, "num_user_turns": int}
  """
  state = DialogueState()
  history: List[dict] = []
  results: dict = {}
  final_action: str = ""

  for user_input in user_turns:
    user_input = user_input.strip()
    if not user_input:
      continue

    history.append({"role": "user", "content": user_input})
    state = update_state(state, history)
    action = decide_action(state)
    final_action = action

    if action == "retrieve":
      try:
        results["flights"] = search_flights(state)
        results["hotels"] = search_hotels(state)
      except NotImplementedError:
        results = {}

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

    if action == "done":
      break

  num_user_turns = sum(1 for t in history if t["role"] == "user")
  return {"final_action": final_action, "num_user_turns": num_user_turns}


if __name__ == "__main__":
  run_agent()
