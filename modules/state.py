"""Shared DialogueState contract. Do not modify without team sign-off."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DialogueState:
  """Tracks all dialogue slots across a conversation turn."""

  origin: Optional[str] = None
  destination: Optional[str] = None
  depart_date: Optional[str] = None
  return_date: Optional[str] = None
  budget_usd: Optional[int] = None
  num_travelers: Optional[int] = None
  preferences: List[str] = field(default_factory=list)
  confirmed_flight: Optional[dict] = None
  confirmed_hotel: Optional[dict] = None
  itinerary: List[dict] = field(default_factory=list)

  # Required slots that must be filled before retrieval can proceed.
  _REQUIRED = ("origin", "destination", "depart_date", "num_travelers")

  def missing_slots(self) -> List[str]:
    """Return names of required slots that have not yet been filled."""
    return [s for s in self._REQUIRED if getattr(self, s) is None]
