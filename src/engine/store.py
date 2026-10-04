"""Signal store: the engine publishes signals to (1) gzipped JSON-lines, (2) SQLite and
(3) an aggregated daily CSV. Downstream modules and the API read from here."""
from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import DAILY_SIGNALS_CSV, SIGNALS_DB, SIGNALS_JSONL

COLUMNS = ["doc_id", "timestamp", "source", "channel", "entity", "entity_type", "relevance", "sentiment",
           "sentiment_label", "event_type", "event_confidence", "impact", "text", "explain"]


def write_jsonl(signals: Iterable[dict], path: Path = SIGNALS_JSONL) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for s in signals:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_signals(path: Path = SIGNALS_JSONL) -> pd.DataFrame:
    df = pd.read_json(path, lines=True, compression="gzip", dtype={"doc_id": str})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def to_sqlite(df: pd.DataFrame, path: Path = SIGNALS_DB) -> None:
    out = df.copy()
    out["timestamp"] = out["timestamp"].astype(str)
    out["explain"] = out["explain"].map(json.dumps)
    with sqlite3.connect(path) as con:
        out[COLUMNS].to_sql("signals", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_entity_ts ON signals(entity, timestamp)")
        con.execute("CREATE INDEX IF NOT EXISTS ix_event ON signals(event_type, impact)")


def query_sqlite(sql: str, params: tuple = (), path: Path = SIGNALS_DB) -> list[dict]:
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(sql, params).fetchall()]
    for r in rows:
        if "explain" in r and isinstance(r["explain"], str):
            r["explain"] = json.loads(r["explain"])
    return rows


def read_daily(path: Path = DAILY_SIGNALS_CSV) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])
