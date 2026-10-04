"""REST API exposing the risk engine's signals and both downstream modules.

Run:  uvicorn src.api.main:app --port 8000      (docs at http://localhost:8000/docs)
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import EVENT_TYPES, PROCESSED, SIGNALS_DB, STRESS_TRIGGERS, TICKERS
from src.engine.engine import RiskEngine
from src.engine.store import query_sqlite, read_signals, to_sqlite
from src.modules.rebalancer import IndexRebalancer
from src.modules.stress.engine import is_trigger, run_stress

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not SIGNALS_DB.exists():
        to_sqlite(read_signals())
    STATE["engine"] = RiskEngine()
    STATE["rebalancer"] = IndexRebalancer()
    yield


app = FastAPI(title="AI/NLP Financial Risk Engine", version="1.0",
              description="Structured risk signals (sentiment, event type, impact) from news and social media, "
                          "plus a sentiment-driven index rebalancer and an event-driven stress tester.",
              lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., examples=["Russia invades Crimea as Ukraine crisis escalates, markets plunge"])
    source: str = Field("news", pattern="^(news|social)$")
    tickers: list[str] = Field(default_factory=list)


class StressRequest(BaseModel):
    event_type: Optional[str] = Field(None, examples=["Geopolitical"])
    impact: Optional[float] = Field(None, ge=1, le=10, examples=[8.5])
    text: Optional[str] = Field(None, description="If given, the engine classifies it first")
    force: bool = Field(False, description="Run even if the trigger threshold is not met")


class RebalanceRequest(BaseModel):
    scores: dict[str, float] = Field(..., examples=[{"AAPL": 0.4, "XOM": -0.3}])
    n_docs: Optional[dict[str, int]] = None


@app.get("/")
def root():
    return {"name": "AI/NLP Financial Risk Engine", "docs": "/docs",
            "endpoints": ["/health", "/signals", "/signals/latest", "/signals/daily", "/analyze", "/stream",
                          "/modules/index/weights", "/modules/index/backtest", "/modules/index/rebalance",
                          "/modules/stress/triggers", "/modules/stress/run"]}


@app.get("/health")
def health():
    n = query_sqlite("SELECT COUNT(*) AS n, MIN(timestamp) AS first, MAX(timestamp) AS last FROM signals")[0]
    return {"status": "ok", "backends": STATE["engine"].backends, "signals": n}


@app.get("/signals")
def signals(entity: Optional[str] = None, event_type: Optional[str] = None, source: Optional[str] = None,
            min_impact: float = 0, start: Optional[str] = None, end: Optional[str] = None,
            limit: int = Query(100, le=5000), offset: int = 0):
    """Query document-level signals (most recent first)."""
    sql, p = "SELECT * FROM signals WHERE impact >= ?", [min_impact]
    for col, val in (("entity", entity), ("event_type", event_type), ("source", source)):
        if val:
            sql += f" AND {col} = ?"
            p.append(val.upper() if col == "entity" else val)
    if start:
        sql += " AND timestamp >= ?"
        p.append(start)
    if end:
        sql += " AND timestamp <= ?"
        p.append(end + "T23:59:59")
    sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    p += [limit, offset]
    return query_sqlite(sql, tuple(p))


@app.get("/signals/latest")
def latest(n: int = Query(20, le=500)):
    return query_sqlite("SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", (n,))


@app.get("/signals/daily")
def daily(entity: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None):
    df = pd.read_csv(PROCESSED / "daily_signals.csv")
    if entity:
        df = df[df.entity == entity.upper()]
    if start:
        df = df[df.date >= start]
    if end:
        df = df[df.date <= end]
    return json.loads(df.to_json(orient="records"))


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Run the NLP engine on any text and return structured signals (one per linked entity)."""
    sigs = STATE["engine"].analyze_text(req.text, req.source, [t.upper() for t in req.tickers])
    return [s.to_dict() for s in sigs]


@app.get("/stream")
async def stream(start: str = "2014-03-01", end: str = "2014-03-07", speed: float = 20.0,
                 min_impact: float = 0):
    """Server-Sent Events replay of historical signals in time order (`speed` signals/second)."""
    rows = query_sqlite("SELECT * FROM signals WHERE timestamp >= ? AND timestamp <= ? AND impact >= ? "
                        "ORDER BY timestamp", (start, end + "T23:59:59", min_impact))

    async def gen():
        for r in rows:
            yield f"data: {json.dumps(r)}\n\n"
            await asyncio.sleep(1.0 / max(speed, 0.1))
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/modules/index/weights")
def index_weights(date: Optional[str] = None):
    w = pd.read_csv(PROCESSED / "index_weights.csv", parse_dates=["date"]).set_index("date")
    row = w.loc[:pd.Timestamp(date)].iloc[-1] if date else w.iloc[-1]
    return {"date": str(row.name.date()), "weights": row.round(5).to_dict()}


@app.get("/modules/index/backtest")
def index_backtest():
    m = json.loads((PROCESSED.parent.parent / "models" / "metrics.json").read_text())
    return m.get("module_a_backtest", {})


@app.post("/modules/index/rebalance")
def index_rebalance(req: RebalanceRequest):
    """Live mode: push one day of sentiment scores, get the new index weights."""
    rb: IndexRebalancer = STATE["rebalancer"]
    rb.update({k.upper(): v for k, v in req.scores.items()}, req.n_docs)
    turnover = rb.rebalance()
    return {"weights": rb.snapshot(), "turnover": round(turnover, 4),
            "smoothed_sentiment": dict(zip(TICKERS, rb.ewma.round(4).tolist()))}


@app.get("/modules/stress/triggers")
def stress_triggers(limit: int = 50):
    rows = query_sqlite("SELECT timestamp, event_type, impact, sentiment, text FROM signals WHERE entity='MARKET' "
                        "ORDER BY impact DESC, timestamp DESC LIMIT 2000")
    out = [r for r in rows if is_trigger(r["event_type"], r["impact"], "market", r["sentiment"])]
    return {"thresholds": STRESS_TRIGGERS, "events": out[:limit]}


@app.post("/modules/stress/run")
def stress_run(req: StressRequest):
    trigger_text = req.text or ""
    if req.text:
        sigs = STATE["engine"].analyze_text(req.text, "news")
        top = max(sigs, key=lambda s: s.impact)
        event_type, impact, sentiment = top.event_type, top.impact, top.sentiment
    elif req.event_type and req.impact is not None:
        event_type, impact, sentiment = req.event_type, req.impact, -1.0
    else:
        raise HTTPException(422, "Provide either `text` or both `event_type` and `impact`.")
    if event_type not in EVENT_TYPES:
        raise HTTPException(422, f"event_type must be one of {EVENT_TYPES}")
    triggered = is_trigger(event_type, impact, "market", sentiment)
    if not triggered and not req.force:
        return {"triggered": False, "event_type": event_type, "impact": impact,
                "threshold": STRESS_TRIGGERS.get(event_type), "message": "Below trigger threshold (use force=true)."}
    res = run_stress(event_type, impact, trigger_text)
    return {"triggered": triggered, **res.summary()}
