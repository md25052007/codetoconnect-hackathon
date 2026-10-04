"""Fast unit tests: python -m pytest -q"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import TICKERS
from src.engine.engine import RiskEngine
from src.engine.nlp.entities import link_entities
from src.engine.nlp.events import RuleTagger
from src.engine.nlp.impact import impact_score
from src.modules.rebalancer import IndexRebalancer, _cap_normalise
from src.modules.stress.engine import corroborated, is_trigger, load_portfolio, run_stress
from src.modules.stress.scenarios import SCENARIOS, severity_factor


@pytest.fixture(scope="module")
def engine():
    return RiskEngine()


def test_signal_schema_and_ranges(engine):
    out = engine.analyze_text("Greece defaults on IMF loan as bailout talks collapse")
    assert out, "engine returned no signals"
    for s in out:
        assert -1.0 <= s.sentiment <= 1.0
        assert 1.0 <= s.impact <= 10.0
        assert s.sentiment_label in {"negative", "neutral", "positive"}
    assert any(s.event_type == "Credit Event" for s in out)
    assert any(s.entity == "MARKET" for s in out)


def test_sentiment_polarity(engine):
    pos = engine.analyze_text("Apple posts record quarterly profit, shares soar", "news")[0]
    neg = engine.analyze_text("Exxon profit plunges 50% as oil prices crash", "news")[0]
    assert pos.sentiment > 0 > neg.sentiment


def test_entity_linking():
    ents = {e.entity for e in link_entities("$AAPL and Microsoft rally", [])}
    assert {"AAPL", "MSFT"} <= ents
    sector = {e.entity for e in link_entities("Oil prices slump after OPEC meeting", [])}
    assert {"XOM", "CVX"} <= sector


@pytest.mark.parametrize("text,label", [
    ("Russia invades Crimea", "Geopolitical"),
    ("Fed raises interest rates by 25bp", "Macroeconomic"),
    ("Brazil downgraded to junk status by Moody's", "Credit Event"),
    ("Kraft agrees to merge with Heinz", "Merger/Acquisition"),
    ("Apple unveils new iPhone at keynote", "Product Launch"),
    ("Intel quarterly earnings beat estimates", "Earnings"),
    ("Apple sued over iPhone storage, class action filed", "Regulatory/Legal"),
    ("How to change the default image viewer on Mac", "Other"),
    ("Israel launches operation in Gaza", "Geopolitical"),
])
def test_rules(text, label):
    assert RuleTagger().predict_one(text)[0] == label


def test_impact_monotone_in_severity():
    hi, _ = impact_score("war breaks out", "Geopolitical", -0.8, "news", "market")
    lo, _ = impact_score("new phone colour", "Product Launch", 0.1, "social", "company", {"followers": 10})
    assert hi > lo


def test_cap_normalise():
    w = _cap_normalise(np.array([10, 1, 1, 1, 1.0]), 0.05, 0.4)
    assert abs(w.sum() - 1) < 1e-9 and w.max() <= 0.4 + 1e-9 and w.min() >= 0.05 - 1e-9


def test_rebalancer_direction():
    rb = IndexRebalancer()
    for _ in range(5):
        rb.update({"AAPL": 0.6, "XOM": -0.6}, {"AAPL": 10, "XOM": 10})
        rb.rebalance()
    w = rb.snapshot()
    assert w["AAPL"] > 1 / len(TICKERS) > w["XOM"]
    assert abs(sum(w.values()) - 1) < 1e-4  # snapshot is rounded to 6 dp


def test_rebalancer_turnover_limit():
    rb = IndexRebalancer(max_turnover=0.05)
    rb.update({"AAPL": 1.0, "MSFT": -1.0}, {"AAPL": 10, "MSFT": 10})
    assert rb.rebalance() <= 0.05 + 1e-9


def test_stress_trigger_rules():
    assert is_trigger("Geopolitical", 7.5, "market", -0.4)
    assert not is_trigger("Geopolitical", 7.0, "market", -0.4)      # brief: impact > 7
    assert not is_trigger("Geopolitical", 9.0, "market", 0.6)       # good news is not a stress event
    assert not is_trigger("Product Launch", 10, "market", -1)
    assert corroborated("Geopolitical", 7.5, 4) and not corroborated("Geopolitical", 7.5, 1)
    assert corroborated("Geopolitical", 9.5, 1) and corroborated("Credit Event", 6.5, 1)


def test_stress_losses_and_scaling():
    pf = load_portfolio()
    mild = run_stress("Credit Event", 7.0, portfolio=pf)
    severe = run_stress("Credit Event", 10.0, portfolio=pf)
    assert severe.pnl < mild.pnl < 0
    assert severity_factor(7) == 0.5 and severity_factor(10) == 1.0
    ex = run_stress("x", 10, scenario=SCENARIOS["Assignment example"], portfolio=pf)
    assert ex.by_risk_factor["rates"] < 0 and ex.by_risk_factor["equity"] < 0


def test_rates_up_hurts_bonds():
    pf = pd.DataFrame([{"asset_class": "Bond", "instrument": "Corporate Bond", "sector": "Energy", "region": "Europe",
                        "rating": "A", "currency": "USD", "notional": 100.0, "market_value": 100.0, "mod_duration": 5.0,
                        "spread_duration": 5.0, "convexity": 0.25, "equity_beta": 0, "pd_1y": 0.0006, "lgd": 0.6,
                        "dv01": 0, "cs01": 0, "fx_direction": 0, "trade_id": "X", "counterparty": "c"}])
    r = run_stress("x", 10, scenario=SCENARIOS["Assignment example"], portfolio=pf)
    assert r.pnl == pytest.approx(-100 * 5 * 0.02 + 0.5 * 0.25 * 100 * 0.02 ** 2 * 100, abs=1)


def test_live_sources_parse_offline(monkeypatch):
    """Live connectors parse real-format payloads (network mocked)."""
    import feedparser
    import requests

    from src.engine.ingestion.sources import GoogleNewsRSS, RedditLive
    rss = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
      <item><title>Apple shares jump after record iPhone sales - Reuters</title><link>https://example.com/a</link>
      <pubDate>Mon, 05 Oct 2026 10:00:00 GMT</pubDate></item></channel></rss>"""
    real_parse = feedparser.parse
    monkeypatch.setattr(feedparser, "parse", lambda url: real_parse(rss))
    docs = list(GoogleNewsRSS(tickers=["AAPL"], per_query=5).documents())
    assert docs and docs[0].tickers == ["AAPL"] and "Apple" in docs[0].text

    class R:
        def raise_for_status(self): pass
        def json(self):
            return {"data": {"children": [{"data": {"id": "x1", "title": "Fed hikes rates, stocks tumble",
                                                     "selftext": "", "created_utc": 1790000000, "score": 120,
                                                     "permalink": "/r/stocks/x1"}}]}}
    monkeypatch.setattr(requests, "get", lambda *a, **k: R())
    rd = list(RedditLive(limit=1).documents())
    assert rd and rd[0].source == "social"
    sig = RiskEngine().analyze_batch(docs + rd)
    assert sig
