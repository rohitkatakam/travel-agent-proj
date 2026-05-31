"""Dialogue State Tracker (DST).

Approach: Hybrid extraction — trained scikit-learn classifier for preferences,
regex rules for structured slots (cities, dates, traveler count, budget).

Training runs on Colab using MultiWOZ 2.2 data (hotel, restaurant, attraction,
train domains only). The trained model is saved as a pickle at
`data/processed/dst_model.pkl` and loaded locally at inference time.

No external API calls or LLM prompts are used.
"""

import pickle
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.state import DialogueState

_ROOT_DIR = Path(__file__).parent.parent
_MODEL_PATHS = (
  _ROOT_DIR / "data" / "processed" / "dst_model.pkl",
  _ROOT_DIR / "dst_model_496.pkl",
)

_MODEL_CACHE: Dict[str, Any] = {}

_SLOT_NAMES = (
  "origin",
  "destination",
  "depart_date",
  "budget_usd",
  "num_travelers",
  "preferences",
)

_NONE_TOKEN = "<NONE>"

# Cities present in data/travel_db/flights.csv — longest first to avoid
# multi-word prefix collisions (e.g. "New" matching before "New York").
_KNOWN_CITIES = [
  "San Francisco", "New Orleans", "Los Angeles",
  "Las Vegas", "New York",
  "Austin", "Boston", "Chicago", "Denver",
  "Miami", "Nashville", "Seattle",
]

_WORD_NUMS: Dict[str, int] = {
  "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
  "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_MONTH_NUMS: Dict[str, int] = {
  "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
  "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
  "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9,
  "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

_WEEKDAY_NUMS: Dict[str, int] = {
  "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
  "friday": 4, "saturday": 5, "sunday": 6,
}

_PREFERENCE_KEYWORDS = (
  "vegetarian", "vegan", "pool", "wifi", "parking", "outdoor", "culture",
  "sightseeing", "food", "entertainment", "sports", "budget", "cheap",
  "affordable", "luxury", "fine dining", "upscale",
)

# Words that suggest a date applies to the return leg, not departure.
_RETURN_KEYWORDS = re.compile(
  r'\b(return(?:ing)?|come?\s+back|coming\s+home|back\s+(?:on|by)|arrive\s+back|'
  r'heading\s+back|fly(?:ing)?\s+back)\b',
  re.I,
)


def _load_model() -> Dict[str, Any]:
  """Load the trained model bundle from disk, caching after first call.

  Returns:
    Dict with keys "vectorizer", "classifiers", "vocabularies".

  Raises:
    FileNotFoundError: If the model pickle does not exist yet.
  """
  if _MODEL_CACHE:
    return _MODEL_CACHE

  model_path = next((p for p in _MODEL_PATHS if p.exists()), None)
  if model_path is None:
    raise FileNotFoundError(
      f"DST model not found at any expected path: {_MODEL_PATHS}. "
      "Run the Colab training notebook first to produce the model pickle."
    )

  with open(model_path, "rb") as f:
    model = pickle.load(f)

  _MODEL_CACHE["vectorizer"] = model["vectorizer"]
  _MODEL_CACHE["classifiers"] = model["classifiers"]
  _MODEL_CACHE["vocabularies"] = model["vocabularies"]
  return _MODEL_CACHE


def _extract_features(
  conversation_history: List[dict],
  vectorizer: Any,
  n_turns: int = 3,
) -> Any:
  """Concatenate the last n user turns and TF-IDF transform them.

  Args:
    conversation_history: List of {"role": str, "content": str} dicts.
    vectorizer: Fitted TfidfVectorizer from the trained model.
    n_turns: Number of most recent user turns to include.

  Returns:
    Sparse TF-IDF feature matrix (1 x n_features).
  """
  user_turns = [
    turn["content"]
    for turn in conversation_history
    if turn.get("role") == "user"
  ]
  text = " ".join(user_turns[-n_turns:]) if user_turns else ""
  return vectorizer.transform([text])


def _predict_slots(
  features: Any,
  classifiers: Dict[str, Any],
  vocabularies: Dict[str, List[str]],
) -> Dict[str, Optional[str]]:
  """Run each slot classifier and map predicted indices back to values.

  Args:
    features: Sparse TF-IDF feature matrix (1 x n_features).
    classifiers: Dict of slot_name -> LogisticRegression.
    vocabularies: Dict of slot_name -> list of possible values.

  Returns:
    Dict of slot_name -> predicted value (or None if predicted <NONE>).
  """
  predictions: Dict[str, Optional[str]] = {}

  for slot_name in _SLOT_NAMES:
    clf = classifiers[slot_name]
    vocab = vocabularies[slot_name]
    pred_idx = clf.predict(features)[0]
    pred_value = vocab[pred_idx]
    predictions[slot_name] = None if pred_value == _NONE_TOKEN else pred_value

  return predictions


def _find_dates_in_text(text: str) -> List[Tuple[int, str]]:
  """Find all date-like patterns and return (position, YYYY-MM-DD) pairs.

  Handles ISO dates, month-day phrases, and weekday names. Pairs are
  sorted by their position in the text.

  Args:
    text: Raw user text (lowercased).

  Returns:
    List of (char_position, "YYYY-MM-DD") sorted ascending by position.
  """
  today = date.today()
  found: List[Tuple[int, str]] = []

  # ISO date: 2026-06-07
  for m in re.finditer(r'\b(\d{4}-\d{2}-\d{2})\b', text):
    found.append((m.start(), m.group(1)))

  # Month Day[st/nd/rd/th]: "june 7", "june 7th"
  month_alt = "|".join(_MONTH_NUMS.keys())
  for m in re.finditer(
    rf'\b({month_alt})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b', text, re.I
  ):
    month = _MONTH_NUMS[m.group(1).lower()]
    day = int(m.group(2))
    try:
      candidate = date(today.year, month, day)
      year = today.year if candidate >= today else today.year + 1
      found.append((m.start(), f"{year}-{month:02d}-{day:02d}"))
    except ValueError:
      pass

  # Weekday names: "friday", "next friday"
  for day_name, day_num in _WEEKDAY_NUMS.items():
    for m in re.finditer(rf'\b(next\s+)?{day_name}\b', text, re.I):
      days_ahead = (day_num - today.weekday() + 7) % 7
      if m.group(1) or days_ahead == 0:
        days_ahead += 7
      resolved = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
      found.append((m.start(), resolved))

  # Sort by position; deduplicate keeping first occurrence of each date value.
  found.sort(key=lambda x: x[0])
  seen: set = set()
  unique: List[Tuple[int, str]] = []
  for pos, d in found:
    if d not in seen:
      seen.add(d)
      unique.append((pos, d))
  return unique


def _extract_slots_regex(
  conversation_history: List[dict],
  n_turns: int = 3,
) -> Dict[str, Any]:
  """Rule-based slot extraction to supplement the ML classifier.

  Handles origin/destination (matched against known DB cities), dates
  (resolved to YYYY-MM-DD), number of travelers, and budget.

  Args:
    conversation_history: List of {"role": str, "content": str} dicts.
    n_turns: Number of most recent user turns to include.

  Returns:
    Dict of slot_name -> extracted value (str); only populated keys found.
  """
  user_turns = [
    t["content"] for t in conversation_history if t.get("role") == "user"
  ]
  text = " ".join(user_turns[-n_turns:]) if user_turns else ""
  recent_turns = user_turns[-n_turns:]
  tl = text.lower()
  results: Dict[str, Any] = {}

  # --- Cities ---
  origin: Optional[str] = None
  destination: Optional[str] = None

  # Try "from X to Y" pattern first; check captured groups against known cities.
  from_to = re.search(r'\bfrom\s+(.+?)\s+to\s+(\S[\w\s]*?)(?=\s*(?:\bon\b|\bfor\b|,|\.|$))', tl)
  if from_to:
    g1, g2 = from_to.group(1), from_to.group(2)
    for city in _KNOWN_CITIES:
      cl = city.lower()
      if cl in g1 and origin is None:
        origin = city
      if cl in g2 and destination is None:
        destination = city

  if origin is None or destination is None:
    for from_city in _KNOWN_CITIES:
      for to_city in _KNOWN_CITIES:
        if from_city == to_city:
          continue
        if re.search(r'\b' + re.escape(from_city.lower()) + r'\s+to\s+' + re.escape(to_city.lower()) + r'\b', tl):
          origin = origin or from_city
          destination = destination or to_city
          break
      if origin and destination:
        break

  # Standalone origin patterns.
  if origin is None:
    for city in _KNOWN_CITIES:
      cl = city.lower()
      if (
        re.search(r'\bfrom\s+' + re.escape(cl), tl)
        or re.search(r'\bdepart(?:ing|ure)?\s+(?:from\s+)?' + re.escape(cl), tl)
        or re.search(r'\bleaving\s+' + re.escape(cl), tl)
        or re.search(r'\bcoming\s+from\s+' + re.escape(cl), tl)
      ):
        origin = city
        break

  # Standalone destination patterns.
  if destination is None:
    for city in _KNOWN_CITIES:
      cl = city.lower()
      if (
        re.search(r'\bto\s+' + re.escape(cl), tl)
        or re.search(r'\bin\s+' + re.escape(cl), tl)
        or re.search(r'\bvisit(?:ing)?\s+' + re.escape(cl), tl)
        or re.search(r'\bgoing\s+to\s+' + re.escape(cl), tl)
        or re.search(r'\bheaded?\s+to\s+' + re.escape(cl), tl)
      ):
        destination = city
        break

  if origin:
    results["origin"] = origin
  if destination:
    results["destination"] = destination

  # --- Dates ---
  date_text = tl
  for turn in reversed(recent_turns):
    if _find_dates_in_text(turn.lower()):
      date_text = turn.lower()
      break

  dated_pairs = _find_dates_in_text(date_text)
  if dated_pairs:
    return_idx: Optional[int] = None
    depart_idx: Optional[int] = None

    for i, (pos, _) in enumerate(dated_pairs):
      # Look at a short window of text before the date for return context.
      window = date_text[max(0, pos - 60):pos]
      if _RETURN_KEYWORDS.search(window):
        if return_idx is None:
          return_idx = i
      else:
        if depart_idx is None:
          depart_idx = i

    if depart_idx is not None:
      results["depart_date"] = dated_pairs[depart_idx][1]
    if return_idx is not None:
      results["return_date"] = dated_pairs[return_idx][1]
    elif len(dated_pairs) >= 2 and depart_idx == 0:
      # Two dates, no return keyword — treat second as return.
      results["return_date"] = dated_pairs[1][1]

  # --- Num travelers ---
  latest = recent_turns[-1].lower() if recent_turns else ""
  last_prompt = ""
  seen_latest_user = False
  for turn in reversed(conversation_history):
    if not seen_latest_user and turn.get("role") == "user":
      seen_latest_user = True
      continue
    if seen_latest_user and turn.get("role") == "assistant":
      last_prompt = turn.get("content", "").lower()
      break

  word_alt = "|".join(_WORD_NUMS.keys())
  num_travel_pat = re.compile(
    rf'\b(?:(\d+)|({word_alt}))\s+'
    r'(?:people|person|persons|passenger|passengers|traveler|travelers|adult|adults|ticket|tickets)\b',
    re.I,
  )
  if "how many travelers" in last_prompt:
    bare_num = re.fullmatch(r'\s*(\d+|' + word_alt + r')\s*', latest, re.I)
    if bare_num:
      raw = bare_num.group(1).lower()
      results["num_travelers"] = str(_WORD_NUMS.get(raw, int(raw) if raw.isdigit() else 0))

  if "num_travelers" not in results:
    m = num_travel_pat.search(tl)
    if m:
      results["num_travelers"] = str(int(m.group(1)) if m.group(1) else _WORD_NUMS[m.group(2).lower()])
    else:
      party_pat = re.compile(
        rf'\bparty\s+of\s+(?:(\d+)|({word_alt}))\b', re.I
      )
      m = party_pat.search(tl)
      if m:
        results["num_travelers"] = str(int(m.group(1)) if m.group(1) else _WORD_NUMS[m.group(2).lower()])
      elif re.search(r'\b(just\s+me|only\s+me|solo|by\s+myself)\b', tl):
        results["num_travelers"] = "1"
      else:
        us_pat = re.compile(
          rf'\b(?:(\d+)|({word_alt}))\s+of\s+us\b', re.I
        )
        m = us_pat.search(tl)
        if m:
          results["num_travelers"] = str(int(m.group(1)) if m.group(1) else _WORD_NUMS[m.group(2).lower()])

  # --- Budget ---
  for turn in reversed(recent_turns):
    turn_l = turn.lower()
    budget_pat = re.compile(r'\$\s*(\d[\d,]*)', re.I)
    m = budget_pat.search(turn_l)
    if m:
      results["budget_usd"] = m.group(1).replace(",", "")
      break

    unit_pat = re.compile(r'(\d[\d,]*)\s*(?:dollars?|usd)\b', re.I)
    m = unit_pat.search(turn_l)
    if m:
      results["budget_usd"] = m.group(1).replace(",", "")
      break

    ctx_pat = re.compile(
      r'\b(?:budget|spend(?:ing)?|afford|up\s+to|max(?:imum)?|around|about)\s+\$?\s*(\d[\d,]+)',
      re.I,
    )
    m = ctx_pat.search(turn_l)
    if m:
      results["budget_usd"] = m.group(1).replace(",", "")
      break

  # Destination-only corrections such as "change the destination to Chicago".
  for turn in reversed(recent_turns):
    turn_l = turn.lower()
    found_correction = False
    for city in _KNOWN_CITIES:
      cl = city.lower()
      if re.search(r'\b(?:change|switch|instead|make)\b.*\b(?:destination|trip)?\s*(?:to\s+)?' + re.escape(cl), turn_l):
        results["destination"] = city
        found_correction = True
        break
    if found_correction:
      break

  latest_exact_city: Optional[str] = None
  for city in _KNOWN_CITIES:
    if latest.strip() == city.lower():
      latest_exact_city = city
      break

  if latest_exact_city:
    if "where would you like to travel to" in last_prompt or "destination" in last_prompt:
      results["destination"] = latest_exact_city
    elif "where will you be departing from" in last_prompt or "departing from" in last_prompt:
      results["origin"] = latest_exact_city
    elif not last_prompt and "origin" not in results and "destination" not in results:
      results["origin"] = latest_exact_city

  preferences = [
    pref for pref in _PREFERENCE_KEYWORDS
    if re.search(r'\b' + re.escape(pref) + r'\b', tl)
  ]
  if preferences:
    results["preferences"] = preferences

  return results


def _merge_predictions(
  state: DialogueState,
  predictions: Dict[str, Optional[str]],
) -> DialogueState:
  """Merge non-None predicted slot values into the dialogue state.

  Args:
    state: Current dialogue state.
    predictions: Dict of slot_name -> predicted value or None.

  Returns:
    Updated DialogueState.
  """
  for slot_name, value in predictions.items():
    if value is not None and hasattr(state, slot_name):
      if slot_name == "num_travelers":
        try:
          value = int(value)
        except ValueError:
          continue
      if slot_name == "budget_usd":
        try:
          value = int(value)
        except ValueError:
          continue
      if slot_name == "preferences":
        values = value if isinstance(value, list) else [value]
        for pref in values:
          if pref not in state.preferences:
            state.preferences.append(pref)
        continue
      setattr(state, slot_name, value)
  if state.depart_date and state.return_date:
    try:
      if date.fromisoformat(state.return_date) < date.fromisoformat(state.depart_date):
        state.return_date = None
    except ValueError:
      pass
  return state


def update_state(
  state: DialogueState,
  conversation_history: List[dict],
) -> DialogueState:
  """Extract slots from conversation history and merge into state.

  Uses a hybrid approach: the trained scikit-learn classifier handles
  open-ended preferences (food/activity types); regex rules handle
  structured slots (cities, dates, traveler count, budget). Regex
  results take priority when non-null.

  Args:
    state: Current dialogue state.
    conversation_history: List of {"role": str, "content": str} dicts.

  Returns:
    Updated DialogueState with any newly extracted slots filled in.
  """
  try:
    model = _load_model()
    features = _extract_features(conversation_history, model["vectorizer"])
    clf_predictions = _predict_slots(features, model["classifiers"], model["vocabularies"])
  except Exception:
    clf_predictions = {}

  regex_predictions = _extract_slots_regex(conversation_history)

  # Regex wins for every slot it confidently found.
  merged = {**clf_predictions, **{k: v for k, v in regex_predictions.items() if v is not None}}

  return _merge_predictions(state, merged)
