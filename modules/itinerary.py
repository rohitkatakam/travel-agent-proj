"""Itinerary construction and booking simulation."""

import datetime
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

  Day 1: outbound flight + hotel check-in.
  Middle days: hotel stay per day + activities distributed round-robin.
  Last day (only when return_date set and trip >= 2 days): hotel check-out + return flight.

  Args:
    state: Current dialogue state (provides depart_date and return_date).
    flights: Candidate flight dicts; only flights[0] is used.
    hotels: Candidate hotel dicts; only hotels[0] is used.
    activities: Activity dicts to distribute across middle days.

  Returns:
    Ordered list of {"day": int, "type": str, "details": dict} entries,
    sorted by day. Returns [] if state.depart_date is None.
  """
  if state.depart_date is None:
    return []

  depart = datetime.date.fromisoformat(state.depart_date)
  has_return = state.return_date is not None
  if has_return:
    ret = datetime.date.fromisoformat(state.return_date)
    trip_days = max((ret - depart).days + 1, 1)
  else:
    trip_days = 1

  flight = flights[0] if flights else None
  hotel = hotels[0] if hotels else None
  items: List[dict] = []

  if flight:
    items.append({"day": 1, "type": "flight", "details": flight})
  if hotel:
    items.append({"day": 1, "type": "hotel_checkin", "details": hotel})

  middle_days = list(range(2, trip_days))
  for day_num in middle_days:
    if hotel:
      items.append({"day": day_num, "type": "hotel_stay", "details": hotel})
  for i, act in enumerate(activities):
    if middle_days:
      items.append({"day": middle_days[i % len(middle_days)], "type": "activity", "details": act})

  if has_return and trip_days >= 2:
    if hotel:
      items.append({"day": trip_days, "type": "hotel_checkout", "details": hotel})
    if flight:
      items.append({"day": trip_days, "type": "return_flight", "details": flight})

  items.sort(key=lambda x: x["day"])
  return items


def simulate_booking(state: DialogueState) -> dict:
  """Simulate confirming all items in state.itinerary.

  Returns a booking confirmation dict (no real external calls).

  Raises:
    ValueError: If state.itinerary is empty or required slots are missing.
  """
  if not state.itinerary:
    raise ValueError("Cannot book: itinerary is empty.")
  missing = state.missing_slots()
  if missing:
    raise ValueError(f"Cannot book: missing required slots: {missing}")
  booking_id = str(uuid.uuid4())
  return {
    "booking_id": booking_id,
    "status": "confirmed",
    "itinerary": state.itinerary,
  }
