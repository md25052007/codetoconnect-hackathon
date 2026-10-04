"""Entity linking: which index stocks (or the whole market) is a text about?"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from src.config import MARKET, SECTOR_KEYWORDS, UNIVERSE
from src.engine.nlp.text import cashtags


@dataclass
class EntityMatch:
    entity: str
    entity_type: str   # company | sector | market
    relevance: float
    via: str


def _compile(words: list[str]) -> re.Pattern:
    return re.compile(r"(?<![\w$])(" + "|".join(re.escape(w.strip()) for w in words) + r")(?![\w])", re.IGNORECASE)


_ALIAS_RE = {t: _compile(v["aliases"]) for t, v in UNIVERSE.items()}
_SECTOR_RE = {s: _compile(w) for s, w in SECTOR_KEYWORDS.items()}
_SECTOR_MEMBERS: dict[str, list[str]] = {}
for _t, _v in UNIVERSE.items():
    _SECTOR_MEMBERS.setdefault(_v["sector"], []).append(_t)

MARKET_WIDE_EVENTS = {"Geopolitical", "Macroeconomic", "Credit Event"}


def link_entities(text: str, known_tickers: list[str] | None = None, event_type: str | None = None) -> list[EntityMatch]:
    """Return the entities a text refers to, with a relevance weight.

    Priority: explicit cashtags / source tickers (1.0) > company aliases (0.9) >
    sector keywords (0.5, broadcast to sector members) > market-wide (macro events).
    Tweets that spray many cashtags are usually spam lists, so relevance is damped
    by 1/sqrt(n_tags).
    """
    found: dict[str, EntityMatch] = {}
    tags = [t for t in cashtags(text) if t in UNIVERSE]
    n_tags = max(len(set(cashtags(text))), 1)
    damp = 1.0 / math.sqrt(n_tags) if n_tags > 2 else 1.0

    for t in (known_tickers or []):
        if t in UNIVERSE:
            found[t] = EntityMatch(t, "company", round(damp, 3), "source")
    for t in tags:
        if t not in found:
            found[t] = EntityMatch(t, "company", round(damp, 3), "cashtag")
    for t, rx in _ALIAS_RE.items():
        if t not in found and rx.search(text):
            found[t] = EntityMatch(t, "company", 0.9, "alias")
    if not found:
        for sector, rx in _SECTOR_RE.items():
            if rx.search(text):
                for t in _SECTOR_MEMBERS.get(sector, []):
                    found.setdefault(t, EntityMatch(t, "sector", 0.5, f"sector:{sector}"))

    matches = list(found.values())
    if event_type in MARKET_WIDE_EVENTS and not any(m.entity_type == "company" for m in matches):
        matches.append(EntityMatch(MARKET, "market", 1.0, "macro-event"))
    return matches
