"""Retrieval module — Rohit Katakam.

All retrieval is grounded in local CSV data under data/travel_db/.
No hallucinated results, no external API calls.
"""

from pathlib import Path
from typing import List

import pandas as pd

from modules.state import DialogueState

_DATA_DIR = Path(__file__).parent.parent / "data" / "travel_db"

_DBS: dict = {}


def _load_db() -> None:
  """Load all travel CSVs into module-level DataFrames on first call."""
  global _DBS
  if _DBS:
    return
  _DBS["flights"] = pd.read_csv(_DATA_DIR / "flights.csv")
  _DBS["hotels"] = pd.read_csv(_DATA_DIR / "hotels.csv")
  _DBS["restaurants"] = pd.read_csv(_DATA_DIR / "restaurants.csv")
  _DBS["activities"] = pd.read_csv(_DATA_DIR / "activities.csv")
  for name, df in _DBS.items():
    for col in df.select_dtypes("object").columns:
      _DBS[name][col] = df[col].str.strip()


def search_flights(state: DialogueState, k: int = 5) -> List[dict]:
  """Return flights matching state.origin, state.destination, state.depart_date.

  Hard filters: origin, destination, seats_available >= num_travelers.
  Date: exact match on depart_date; expands to ±3 days if no exact results.
  Budget applied as a soft ranking signal (within_budget ranked first).
  Returns up to k results sorted by budget compliance, date proximity, price.
  """
  if any(v is None for v in (state.origin, state.destination, state.depart_date, state.num_travelers)):
    raise ValueError("search_flights requires origin, destination, depart_date, and num_travelers")

  _load_db()
  df = _DBS["flights"].copy()

  df = df[df["origin"].str.lower() == state.origin.lower()]
  df = df[df["destination"].str.lower() == state.destination.lower()]
  df = df[df["seats_available"] >= state.num_travelers]

  target = pd.to_datetime(state.depart_date)
  df["_depart_dt"] = pd.to_datetime(df["depart_date"])
  df["_days_diff"] = (df["_depart_dt"] - target).dt.days.abs()

  exact = df[df["_days_diff"] == 0]
  candidates = exact if not exact.empty else df[df["_days_diff"] <= 3]

  if state.budget_usd is not None:
    candidates = candidates.copy()
    candidates["_within_budget"] = candidates["price_usd"] <= state.budget_usd * 0.5
  else:
    candidates = candidates.copy()
    candidates["_within_budget"] = True

  candidates = candidates.sort_values(
    ["_within_budget", "_days_diff", "price_usd"],
    ascending=[False, True, True],
  )

  candidates = candidates.drop(columns=["_depart_dt", "_days_diff", "_within_budget"])
  return candidates.head(k).to_dict("records")


def search_hotels(state: DialogueState, k: int = 5) -> List[dict]:
  """Return hotels in state.destination for the trip date range.

  Hard filter: city == destination.
  Date columns in CSV are empty; check_in/check_out are injected from state.
  Budget applied as a soft ranking signal. Ranked by budget compliance then rating.
  Returns up to k results with injected check_in, check_out, num_nights, total_price_usd.
  """
  if state.destination is None:
    raise ValueError("search_hotels requires destination")

  _load_db()
  df = _DBS["hotels"].copy()
  df = df[df["city"].str.lower() == state.destination.lower()]

  if state.return_date and state.depart_date:
    num_nights = max(
      (pd.to_datetime(state.return_date) - pd.to_datetime(state.depart_date)).days,
      1,
    )
  else:
    num_nights = 7

  if state.budget_usd is not None:
    nightly_cap = (state.budget_usd * 0.6) / num_nights
    df["_within_budget"] = df["price_per_night_usd"] <= nightly_cap
  else:
    df["_within_budget"] = True

  df = df.sort_values(["_within_budget", "rating"], ascending=[False, False])
  df = df.drop(columns=["_within_budget"])

  results = df.head(k).to_dict("records")
  for r in results:
    r["check_in"] = state.depart_date
    r["check_out"] = state.return_date
    r["num_nights"] = num_nights
    r["total_price_usd"] = round(r["price_per_night_usd"] * num_nights, 2)
  return results


def search_restaurants(state: DialogueState, city: str, k: int = 5) -> List[dict]:
  """Return restaurants in city, ranked by preference match then rating.

  Hard filter: city match.
  Preferences matched against cuisine via case-insensitive substring.
  Returns up to k results.
  """
  _load_db()
  df = _DBS["restaurants"].copy()
  df = df[df["city"].str.lower() == city.lower()]

  def _cuisine_score(cuisine: str) -> int:
    return sum(
      1 for p in state.preferences
      if p.lower() in cuisine.lower() or cuisine.lower() in p.lower()
    )

  _budget_keywords = {"budget", "cheap", "affordable", "inexpensive"}
  _luxury_keywords = {"luxury", "fine dining", "upscale", "expensive", "fancy"}
  pref_lower = {p.lower() for p in state.preferences}
  wants_cheap = bool(pref_lower & _budget_keywords)
  wants_luxury = bool(pref_lower & _luxury_keywords)

  def _price_pref(price_range: str) -> bool:
    if wants_cheap:
      return price_range in {"$", "$$"}
    if wants_luxury:
      return price_range in {"$$$", "$$$$"}
    return True

  df["_pref_score"] = df["cuisine"].apply(_cuisine_score)
  df["_price_pref"] = df["price_range"].apply(_price_pref)

  df = df.sort_values(["_pref_score", "_price_pref", "rating"], ascending=[False, False, False])
  df = df.drop(columns=["_pref_score", "_price_pref"])
  return df.head(k).to_dict("records")


def search_activities(state: DialogueState, city: str, k: int = 5) -> List[dict]:
  """Return activities in city, ranked by preference match then price.

  Hard filter: city match.
  Optional hard budget filter (10% of total budget per activity); falls back
  to no filter if fewer than k results survive.
  Preferences matched against category via case-insensitive substring.
  Returns up to k results.
  """
  _load_db()
  df = _DBS["activities"].copy()
  df = df[df["city"].str.lower() == city.lower()]

  if state.budget_usd is not None and state.num_travelers is not None:
    per_activity_cap = (state.budget_usd * 0.1) / state.num_travelers
    budget_filtered = df[df["price_usd"] <= per_activity_cap]
    if len(budget_filtered) >= k:
      df = budget_filtered

  def _cat_score(category: str) -> int:
    return sum(
      1 for p in state.preferences
      if p.lower() in category.lower() or category.lower() in p.lower()
    )

  df = df.copy()
  df["_pref_score"] = df["category"].apply(_cat_score)
  df = df.sort_values(["_pref_score", "price_usd"], ascending=[False, True])
  df = df.drop(columns=["_pref_score"])
  return df.head(k).to_dict("records")
