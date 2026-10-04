"""Impact score (1-10): how much could this piece of information move markets?

A transparent, explainable scoring model (every factor is returned in ``explain``):

    raw = severity(event) x intensity(|sentiment|) x credibility(source) x reach(entity)
          + urgency + magnitude + popularity
    impact = 1 + 9 x clip(raw, 0, 1)

* severity    - base severity of the event class (config.EVENT_SEVERITY)
* intensity   - 0.6 + 0.4*|sentiment|; strongly worded news matters more
* credibility - news 1.0; social 0.6-1.0 rising with author reach (log followers, verified)
* reach       - market-wide 1.15 if the story is systemic (major economies, energy, finance,
                great powers) else 0.7; single company 1.0; sector broadcast 0.9
* urgency     - shock words ("breaking", "crash", "emergency", "collapse" ...)
* magnitude   - big numbers: moves >= 5 %, "billion", casualty counts
* popularity  - top-voted news headlines (Reddit rank) get a small boost

The weights were set from domain reasoning, then checked against realised market
moves (see ``scripts/evaluate.py``: high-impact days show larger absolute returns).
"""
from __future__ import annotations

import math
import re

from src.config import EVENT_SEVERITY, SOURCE_CREDIBILITY

URGENCY_RE = re.compile(
    r"\bbreaking\b|\burgent\b|\bemergency\b|\bcrash(?:es|ed)?\b|\bcollapse[sd]?\b|\bplunge[sd]?\b|"
    r"\bplummet(?:s|ed)?\b|\bpanic\b|\bturmoil\b|\bcrisis\b|\bhalt(?:s|ed)?\b|\bsuspend(?:s|ed)?\b|"
    r"\bunexpected(?:ly)?\b|\bshock\b|\brecord\b|\bworst\b|\bbiggest\b|\bhistoric\b|\bescalat",
    re.IGNORECASE,
)
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s?(?:%|percent|per cent)", re.IGNORECASE)
BIG_MONEY_RE = re.compile(r"\b(?:billion|bn|trillion)\b|\$\d{2,}\s?b\b", re.IGNORECASE)
CASUALTY_RE = re.compile(r"(\d[\d,]*)\s+(?:people\s+)?(?:killed|dead|deaths|died|wounded)", re.IGNORECASE)
REACH = {"market": 1.15, "company": 1.0, "sector": 0.9}
LOCAL_MARKET_REACH = 0.7
# Cues that a market-wide story is systemic (major economies, energy, finance, great-power
# conflict). A tragic but local event moves global markets far less than, say, sanctions
# on Russia or an OPEC decision.
SYSTEMIC_RE = re.compile(
    r"\b(?:u\.?s\.?|united states|america|american|washington|obama|china|chinese|beijing|russia[n]?|moscow|putin|"
    r"eu|europe(?:an)?|eurozone|ecb|germany|japan|uk|britain|saudi|iran|opec|nato|g7|g20|imf|world bank|"
    r"fed|federal reserve|central bank|oil|crude|gas|energy|market[s]?|stocks?|bonds?|economy|economic|"
    r"trade|tariffs?|sanctions?|currency|dollar|euro|yuan|ruble|rouble|banks?|debt|default|recession|"
    r"inflation|nuclear|global|world)\b", re.IGNORECASE)


def credibility(source: str, meta: dict) -> float:
    base = SOURCE_CREDIBILITY.get(source, 0.8)
    if source != "social":
        return base
    if meta.get("verified"):
        return 1.0
    followers = float(meta.get("followers") or 0)
    return round(min(1.0, 0.6 + 0.08 * math.log10(1 + followers)), 3)


def impact_score(text: str, event_type: str, sentiment: float, source: str,
                 entity_type: str, meta: dict | None = None) -> tuple[float, dict]:
    meta = meta or {}
    sev = EVENT_SEVERITY.get(event_type, 0.2)
    intensity = 0.6 + 0.4 * min(abs(sentiment), 1.0)
    cred = credibility(source, meta)
    reach = REACH.get(entity_type, 1.0)
    systemic = None
    if entity_type == "market":
        systemic = bool(SYSTEMIC_RE.search(text))
        reach = REACH["market"] if systemic else LOCAL_MARKET_REACH

    urgency = min(0.15, 0.06 * len(URGENCY_RE.findall(text)))
    magnitude = 0.0
    pcts = [float(p) for p in PCT_RE.findall(text)]
    if pcts and max(pcts) >= 5:
        magnitude += 0.05 if max(pcts) < 10 else 0.1
    if BIG_MONEY_RE.search(text):
        magnitude += 0.05
    cas = [int(c.replace(",", "")) for c in CASUALTY_RE.findall(text) if c.replace(",", "").isdigit()]
    if cas and max(cas) >= 10:
        magnitude += 0.05 if max(cas) < 100 else 0.1
    magnitude = min(magnitude, 0.15)
    rank = meta.get("rank")
    popularity = 0.05 if (rank is not None and rank < 3) else 0.0

    raw = sev * intensity * cred * reach + urgency + magnitude + popularity
    score = round(1 + 9 * max(0.0, min(1.0, raw)), 1)
    explain = {"severity": sev, "intensity": round(intensity, 3), "credibility": cred, "reach": reach,
               "urgency": round(urgency, 3), "magnitude": round(magnitude, 3), "popularity": popularity,
               "systemic": systemic}
    return score, explain
