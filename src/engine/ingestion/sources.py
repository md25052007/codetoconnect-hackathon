"""Data ingestion: every source yields ``Document`` objects with a common schema.

Historical sources (bundled in data/raw, always available, used for the demo):
    * StockNetTweets      - social media (Twitter), 59k real tweets, 2014-2016
    * RedditWorldNews     - news feed, 20k top r/worldnews headlines, 2014-2016

Live sources (need internet; used with `python main.py live`):
    * GoogleNewsRSS       - Google News RSS search per company + macro queries
    * RedditLive          - newest posts from r/stocks, r/investing, r/wallstreetbets
"""
from __future__ import annotations

import heapq
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Iterable, Iterator
from urllib.parse import quote_plus

import pandas as pd

from src.config import RAW, UNIVERSE
from src.engine.schema import Document


class Source(ABC):
    name: str = "source"

    @abstractmethod
    def documents(self, start: str | None = None, end: str | None = None) -> Iterator[Document]:
        ...


def _window(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df.timestamp >= pd.Timestamp(start)]
    if end:
        df = df[df.timestamp < pd.Timestamp(end) + pd.Timedelta(days=1)]
    return df


class StockNetTweets(Source):
    name = "twitter"

    def __init__(self, path=RAW / "tweets_stocknet.csv.gz") -> None:
        self.path = path

    def frame(self, start=None, end=None) -> pd.DataFrame:
        df = pd.read_csv(self.path, parse_dates=["timestamp"], dtype={"id": str})
        return _window(df, start, end)

    def documents(self, start=None, end=None) -> Iterator[Document]:
        for r in self.frame(start, end).itertuples(index=False):
            yield Document(id=f"tw-{r.id}", source="social", channel="twitter", timestamp=r.timestamp,
                           text=r.text, tickers=[r.ticker],
                           meta={"followers": int(r.followers), "verified": bool(r.verified), "retweets": int(r.retweets)})


class RedditWorldNews(Source):
    name = "reddit/worldnews"

    def __init__(self, path=RAW / "news_reddit_worldnews.csv.gz") -> None:
        self.path = path

    def frame(self, start=None, end=None) -> pd.DataFrame:
        df = pd.read_csv(self.path, parse_dates=["timestamp"])
        return _window(df, start, end)

    def documents(self, start=None, end=None) -> Iterator[Document]:
        for r in self.frame(start, end).itertuples(index=False):
            yield Document(id=r.id, source="news", channel="reddit/worldnews", timestamp=r.timestamp,
                           text=r.text, meta={"rank": int(r.rank)})


# ---------------------------------------------------------------------------
# Live sources
# ---------------------------------------------------------------------------
class GoogleNewsRSS(Source):
    """Live financial news via Google News RSS (no API key)."""
    name = "google-news"
    MACRO_QUERIES = ["federal reserve interest rates", "inflation economy", "geopolitical tensions markets",
                     "credit rating downgrade default", "oil prices OPEC"]

    def __init__(self, tickers: list[str] | None = None, per_query: int = 10) -> None:
        self.tickers = tickers or list(UNIVERSE)
        self.per_query = per_query

    def _queries(self) -> list[tuple[str, list[str]]]:
        qs = [(f'"{UNIVERSE[t]["name"].split(" (")[0]}" stock', [t]) for t in self.tickers]
        return qs + [(q, []) for q in self.MACRO_QUERIES]

    def documents(self, start=None, end=None) -> Iterator[Document]:
        import feedparser  # local import: only needed in live mode
        for q, tickers in self._queries():
            url = f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            for e in feed.entries[: self.per_query]:
                ts = datetime(*e.published_parsed[:6]) if getattr(e, "published_parsed", None) else datetime.utcnow()
                yield Document(id=f"gn-{abs(hash(e.get('link', e.title)))}", source="news", channel="google-news",
                               timestamp=ts, text=e.title, tickers=tickers, meta={"url": e.get("link")})


class RedditLive(Source):
    """Live social posts from finance subreddits via Reddit's public JSON endpoint."""
    name = "reddit-live"
    SUBS = ["stocks", "investing", "wallstreetbets", "economics"]

    def __init__(self, limit: int = 50) -> None:
        self.limit = limit

    def documents(self, start=None, end=None) -> Iterator[Document]:
        import requests
        headers = {"User-Agent": "risk-engine-hackathon/1.0"}
        for sub in self.SUBS:
            try:
                r = requests.get(f"https://www.reddit.com/r/{sub}/new.json?limit={self.limit}", headers=headers, timeout=15)
                r.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                print(f"[reddit-live] r/{sub} unavailable: {exc}")
                continue
            for child in r.json().get("data", {}).get("children", []):
                p = child["data"]
                text = (p.get("title", "") + ". " + (p.get("selftext") or "")[:400]).strip()
                yield Document(id=f"rd-{p['id']}", source="social", channel=f"reddit/{sub}",
                               timestamp=datetime.fromtimestamp(p["created_utc"], tz=timezone.utc).replace(tzinfo=None),
                               text=text, meta={"followers": p.get("score", 0), "url": "https://reddit.com" + p.get("permalink", "")})


# ---------------------------------------------------------------------------
# Stream merger / replay
# ---------------------------------------------------------------------------
def merged_stream(sources: Iterable[Source], start=None, end=None) -> Iterator[Document]:
    """Merge several time-ordered sources into one chronological stream."""
    return heapq.merge(*(s.documents(start, end) for s in sources), key=lambda d: d.timestamp)


def replay(sources: Iterable[Source], start=None, end=None, speed: float = 3600.0) -> Iterator[Document]:
    """Replay historical documents in (accelerated) real time: speed=3600 -> 1 hour per second."""
    prev = None
    for doc in merged_stream(sources, start, end):
        if prev is not None and speed > 0:
            gap = (doc.timestamp - prev).total_seconds() / speed
            if gap > 0:
                time.sleep(min(gap, 2.0))
        prev = doc.timestamp
        yield doc


def historical_sources() -> list[Source]:
    return [RedditWorldNews(), StockNetTweets()]


def live_sources() -> list[Source]:
    return [GoogleNewsRSS(), RedditLive()]
