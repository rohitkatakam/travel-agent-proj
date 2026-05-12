"""Itinerary construction and booking simulation — Rohit Katakam."""

import uuid
from typing import List

from modules.state import DialogueState


def build_itinerary(
  state: DialogueState,
  flights: List[dict],
  hotels: List[dict],
  activities: List[dict],
) -> List[dict]:
  """Assemble a day-by-day itinerary from confirmed options.

  Args:
    state: Current dialogue state (used for dates and preferences).
    flights: List of flight dicts (from retrieval).
    hotels: List of hotel dicts (from retrieval).
    activities: List of activity dicts (from retrieval).

  Returns:
    Ordered list of itinerary item dicts.
  """
  # TODO: Map flights/hotels/activities onto calendar days.
  # Each item: {"day": int, "type": str, "details": dict}
  raise NotImplementedError


def simulate_booking(state: DialogueState) -> dict:
  """Simulate confirming all items in state.itinerary.

  Returns a booking confirmation dict (no real external calls).
  """
  # TODO: Validate that state.itinerary is non-empty and all required slots are filled.
  booking_id = str(uuid.uuid4())
  return {
    "booking_id": booking_id,
    "status": "confirmed",
    "itinerary": state.itinerary,
  }
