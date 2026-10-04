"""Evaluate the engine against ground truth and realised market data.

1. Event classification  - rules vs weak-supervised model vs hybrid on 200 hand-labelled texts
   (+ sentiment on the 92 market-relevant ones: tests the domain-aware routing)
2. Impact score          - do higher-impact days coincide with larger absolute price moves?
3. Sentiment             - does the daily sentiment score line up with same-day returns?

Writes results into models/metrics.json (shown on the dashboard's "Model quality" tab).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import EVAL, MARKET, MODELS, RAW, TICKERS  # noqa: E402
from src.engine.nlp.events import EventModel, HybridEventClassifier, RuleTagger  # noqa: E402
from src.engine.store import read_daily  # noqa: E402

BUCKETS = [0, 3, 5, 7, 10.01]
BUCKET_LABELS = ["1-3 (low)", "3-5", "5-7", "7-10 (high)"]


def eval_events() -> dict:
    gold = pd.read_csv(EVAL / "event_gold.csv")
    texts, y = gold.text.tolist(), gold.event_type.values
    rules = [r[0] for r in RuleTagger().predict(texts)]
    model = EventModel()
    proba = model.predict_proba(texts)
    mdl = [model.classes[i] for i in proba.argmax(1)]
    hyb = [r[0] for r in HybridEventClassifier().predict(texts)]
    res = {}
    for name, pred in [("rules_only", rules), ("weak_supervised_model_only", mdl), ("hybrid", hyb)]:
        res[name] = {"accuracy": round(accuracy_score(y, pred), 4),
                     "macro_f1": round(f1_score(y, pred, average="macro", zero_division=0), 4)}
    res["hybrid_report"] = classification_report(y, hyb, output_dict=True, zero_division=0)
    res["n"] = len(gold)
    return res


def eval_sentiment_gold() -> dict:
    """Sentiment on the market-relevant part of the hand-labelled set (news headlines and
    tweets about geopolitics, macro, credit, earnings ...), i.e. the engine's real domain."""
    from src.engine.nlp.sentiment import get_sentiment, label
    gold = pd.read_csv(EVAL / "event_gold.csv").dropna(subset=["sentiment"])
    gold = gold[gold.sentiment != ""]
    texts, y = gold.text.tolist(), gold.sentiment.values
    srcs = gold.src.tolist()
    ens = get_sentiment()
    events = [r[0] for r in HybridEventClassifier().predict(texts)]
    no_route, parts = ens.score(texts, srcs, None)
    routed, _ = ens.score(texts, srcs, events)
    res = {"n": int(len(gold))}
    for name, pred in [("model_only", [label(x, 0.35) for x in parts["model"]]),
                       ("finance_lexicon_only", [label(x, 0.05) for x in parts["lexicon"]]),
                       ("ensemble_without_domain_routing", [label(x) for x in no_route]),
                       ("ensemble_with_domain_routing", [label(x) for x in routed])]:
        res[name] = {"accuracy": round(accuracy_score(y, pred), 4),
                     "macro_f1": round(f1_score(y, pred, average="macro", zero_division=0), 4)}
    return res


def _returns() -> pd.DataFrame:
    px = pd.read_csv(RAW / "prices.csv", parse_dates=["Date"]).set_index("Date")
    return px.pct_change()


def eval_impact_and_sentiment() -> dict:
    daily = read_daily()
    rets = _returns()
    rows = []
    for ent, col in [(t, t) for t in TICKERS] + [(MARKET, "DJIA")]:
        d = daily[daily.entity == ent].set_index("date")
        r = rets[col]
        j = d.join(r.rename("ret"), how="inner").join(r.shift(-1).rename("ret_next"), how="left")
        j["entity"] = ent
        rows.append(j.reset_index())
    j = pd.concat(rows).dropna(subset=["ret"])
    j["abs_ret"] = j.ret.abs()
    # normalise by each stock's own volatility so that volatile names do not dominate
    j["abs_ret_z"] = j.groupby("entity").abs_ret.transform(lambda s: s / s.mean())
    j["bucket"] = pd.cut(j.max_impact, BUCKETS, labels=BUCKET_LABELS, right=False)

    out = {}
    for scope, sub in [("companies", j[j.entity != MARKET]), ("market_djia", j[j.entity == MARKET])]:
        tbl = sub.groupby("bucket", observed=False).agg(days=("abs_ret", "size"),
                                                        mean_abs_return_pct=("abs_ret", lambda s: round(100 * s.mean(), 3)),
                                                        rel_move_vs_typical=("abs_ret_z", lambda s: round(s.mean(), 3)))
        out[scope] = {
            "by_impact_bucket": tbl.reset_index().astype({"bucket": str}).to_dict(orient="records"),
            "spearman_impact_vs_abs_return": round(sub.max_impact.corr(sub.abs_ret_z, method="spearman"), 4),
        }
    # market-wide: many headlines per day, so use the count of high-impact (>=7) market signals
    mk = j[j.entity == MARKET].copy()
    mk["high_bucket"] = pd.cut(mk.n_high, [-1, 0, 2, 5, 1000], labels=["0", "1-2", "3-5", "6+"])
    tbl = mk.groupby("high_bucket", observed=False).agg(days=("abs_ret", "size"),
                                                        mean_abs_return_pct=("abs_ret", lambda s: round(100 * s.mean(), 3)))
    out["market_djia"]["by_high_impact_count"] = tbl.reset_index().astype({"high_bucket": str}).to_dict(orient="records")
    out["market_djia"]["spearman_n_high_vs_abs_return"] = round(mk.n_high.corr(mk.abs_ret, method="spearman"), 4)
    comp = j[(j.entity != MARKET) & (j.n_docs >= 3)]
    out["sentiment"] = {
        "n_entity_days": int(len(comp)),
        "spearman_score_vs_same_day_return": float(round(comp.score.corr(comp.ret, method="spearman"), 4)),
        "spearman_score_vs_next_day_return": float(round(comp.score.corr(comp.ret_next, method="spearman"), 4)),
        "mean_return_pct_when_score_pos": float(round(100 * comp[comp.score > 0.05].ret.mean(), 3)),
        "mean_return_pct_when_score_neg": float(round(100 * comp[comp.score < -0.05].ret.mean(), 3)),
    }
    return out


def main() -> None:
    path = MODELS / "metrics.json"
    metrics = json.loads(path.read_text()) if path.exists() else {}
    ev = eval_events()
    print("Event classification on hand-labelled set (n=%d):" % ev["n"])
    for k in ("rules_only", "weak_supervised_model_only", "hybrid"):
        print(f"  {k:28s} {ev[k]}")
    sg = eval_sentiment_gold()
    print("\nSentiment on market-relevant hand-labelled texts (n=%d):" % sg["n"])
    for k, v in sg.items():
        if k != "n":
            print(f"  {k:34s} {v}")
    iv = eval_impact_and_sentiment()
    print("\nImpact validation (companies):")
    for r in iv["companies"]["by_impact_bucket"]:
        print("  ", r)
    print("  spearman:", iv["companies"]["spearman_impact_vs_abs_return"])
    print("Impact validation (market / DJIA):")
    for r in iv["market_djia"]["by_impact_bucket"]:
        print("  ", r)
    print("  spearman:", iv["market_djia"]["spearman_impact_vs_abs_return"])
    for r in iv["market_djia"]["by_high_impact_count"]:
        print("  ", r)
    print("  spearman n_high:", iv["market_djia"]["spearman_n_high_vs_abs_return"])
    print("\nSentiment vs returns:", iv["sentiment"])
    metrics.update({"events_eval": ev, "sentiment_gold_eval": sg, "impact_validation": iv})
    path.write_text(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
