"""Data contracts shared by ingestion, the NLP engine and downstream modules."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Document:
    """A raw piece of unstructured text from any source."""
    id: str
    source: str                 # "news" | "social"
    channel: str                # e.g. "reddit/worldnews", "twitter", "google-news"
    timestamp: datetime
    text: str
    tickers: list[str] = field(default_factory=list)   # tickers already known from the source
    meta: dict[str, Any] = field(default_factory=dict)  # followers, verified, rank, url ...


@dataclass
class RiskSignal:
    """Structured, machine-readable output of the engine (one per document x entity)."""
    doc_id: str
    timestamp: str
    source: str
    channel: str
    entity: str                 # ticker or "MARKET"
    entity_type: str            # "company" | "sector" | "market"
    relevance: float            # 0-1, how directly the text is about the entity
    sentiment: float            # -1.0 (very negative) .. +1.0 (very positive)
    sentiment_label: str        # negative | neutral | positive
    event_type: str             # see config.EVENT_TYPES
    event_confidence: float     # 0-1
    impact: float               # 1-10 predicted market-impact severity
    text: str
    explain: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
