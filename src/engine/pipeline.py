"""Batch pipeline: ingest all historical sources -> NLP engine -> signal store."""
from __future__ import annotations

import time

from src.config import DAILY_SIGNALS_CSV, END_DATE, SIGNALS_JSONL, START_DATE
from src.engine.aggregate import daily_signals
from src.engine.engine import RiskEngine
from src.engine.ingestion.sources import historical_sources, merged_stream
from src.engine.store import read_signals, to_sqlite, write_jsonl


def run(start: str = START_DATE, end: str = END_DATE) -> None:
    t0 = time.time()
    engine = RiskEngine()
    print(f"[pipeline] back-ends: {engine.backends}")
    stream = merged_stream(historical_sources(), start, end)
    n = write_jsonl((s.to_dict() for s in engine.process_stream(stream)), SIGNALS_JSONL)
    print(f"[pipeline] {n:,} signals -> {SIGNALS_JSONL.name} ({time.time() - t0:.0f}s)")
    sig = read_signals()
    to_sqlite(sig)
    daily = daily_signals(sig)
    daily.to_csv(DAILY_SIGNALS_CSV, index=False)
    print(f"[pipeline] {len(daily):,} daily entity signals -> {DAILY_SIGNALS_CSV.name}")
    print(sig.groupby("event_type").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    run()
