"""Aggregate document-level signals into one daily signal per entity.

Timing convention (prevents look-ahead bias in the back-test): anything published after
the US market close (21:00 UTC) is assigned to the *next* day's signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLOSE_UTC_HOUR = 21
SHRINK_K = 3.0  # Bayesian-style shrinkage: few documents -> score pulled towards 0


def signal_date(ts: pd.Series) -> pd.Series:
    return (ts + pd.Timedelta(hours=24 - CLOSE_UTC_HOUR)).dt.normalize()


def daily_signals(sig: pd.DataFrame) -> pd.DataFrame:
    df = sig.copy()
    df["date"] = signal_date(df["timestamp"])
    cred = df["explain"].map(lambda e: e.get("credibility", 1.0) if isinstance(e, dict) else 1.0)
    df["w"] = df["relevance"] * cred * (df["impact"] / 10.0)
    df["ws"] = df["w"] * df["sentiment"]
    df["imp_w"] = df["impact"] * df["relevance"]

    g = df.groupby(["entity", "date"])
    out = g.agg(
        n_docs=("doc_id", "nunique"),
        n_news=("source", lambda s: int((s == "news").sum())),
        n_social=("source", lambda s: int((s == "social").sum())),
        sentiment_mean=("sentiment", "mean"),
        ws=("ws", "sum"),
        w=("w", "sum"),
        max_impact=("impact", "max"),
        n_high=("impact", lambda s: int((s >= 7).sum())),
        mean_impact=("impact", "mean"),
        pos_share=("sentiment_label", lambda s: float((s == "positive").mean())),
        neg_share=("sentiment_label", lambda s: float((s == "negative").mean())),
    ).reset_index()
    out["sentiment_weighted"] = np.where(out.w > 0, out.ws / out.w.replace(0, np.nan), 0.0)
    conf = out.n_docs / (out.n_docs + SHRINK_K)
    out["score"] = (out["sentiment_weighted"] * conf).round(4)

    # dominant event = class with the largest impact mass that day
    ev = df.groupby(["entity", "date", "event_type"])["imp_w"].sum().reset_index()
    ev = ev[ev.event_type != "Other"].sort_values("imp_w").groupby(["entity", "date"]).tail(1)
    out = out.merge(ev[["entity", "date", "event_type"]].rename(columns={"event_type": "dominant_event"}),
                    on=["entity", "date"], how="left")
    out["dominant_event"] = out["dominant_event"].fillna("Other")

    # buzz: today's document count vs the entity's trailing 30-day average
    out = out.sort_values(["entity", "date"])
    roll = out.groupby("entity")["n_docs"].transform(lambda s: s.shift(1).rolling(30, min_periods=5).mean())
    out["buzz"] = (out["n_docs"] / roll).round(2).fillna(1.0)
    cols = ["entity", "date", "n_docs", "n_news", "n_social", "score", "sentiment_weighted", "sentiment_mean",
            "max_impact", "mean_impact", "n_high", "dominant_event", "pos_share", "neg_share", "buzz"]
    return out[cols].round(4).reset_index(drop=True)
