"""Light text normalisation shared by all NLP components."""
from __future__ import annotations

import html
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
RT_RE = re.compile(r"^RT\s+@\w+:?\s*", re.IGNORECASE)
CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5}(?:[.-][A-Za-z])?)\b")
WS_RE = re.compile(r"\s+")


def cashtags(text: str) -> list[str]:
    return [m.upper() for m in CASHTAG_RE.findall(text)]


def clean(text: str, keep_cashtags: bool = False) -> str:
    """Remove URLs, retweet prefixes, @mentions and HTML entities."""
    t = html.unescape(str(text))
    t = RT_RE.sub("", t)
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    if not keep_cashtags:
        t = CASHTAG_RE.sub(" ", t)
    t = t.replace("#", " ")
    return WS_RE.sub(" ", t).strip()


def is_retweet(text: str) -> bool:
    return bool(RT_RE.match(str(text)))
