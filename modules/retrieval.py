"""Retrieval module — Rohit Katakam.

All retrieval is grounded in local CSV data under data/travel_db/.
No hallucinated results, no external API calls.
"""

from typing import List

from modules.state import DialogueState


def search_flights(state: DialogueState) -> List[dict]:
  """Return flights matching state.origin, state.destination, state.depart_date.

  Filters by budget_usd and num_travelers if provided.
  """
  # TODO: Load data/travel_db/flights.csv with pandas.
  # Filter rows matching origin, destination, depart_date.
  # Optionally filter by price_usd <= budget_usd and seats_available >= num_travelers.
  raise NotImplementedError


def search_hotels(state: DialogueState) -> List[dict]:
  """Return hotels in state.destination for the trip date range."""
  # TODO: Load data/travel_db/hotels.csv with pandas.
  # Filter by city == destination, check_in/check_out overlap with trip dates.
  # Optionally filter by price_per_night_usd within budget.
  raise NotImplementedError


def search_restaurants(state: DialogueState, city: str) -> List[dict]:
  """Return restaurants in city, optionally filtered by preferences."""
  # TODO: Load data/travel_db/restaurants.csv with pandas.
  # Filter by city. Optionally match cuisine against state.preferences.
  raise NotImplementedError


def search_activities(state: DialogueState, city: str) -> List[dict]:
  """Return activities in city, optionally filtered by preferences."""
  # TODO: Load data/travel_db/activities.csv with pandas.
  # Filter by city. Optionally match category against state.preferences.
  raise NotImplementedError
