"""Module A - Tactical, sentiment-driven index rebalancer.

Subscribes to the engine's per-stock Sentiment Score and tilts a 20-stock mock index:

    1. smooth      S_i(t) = lam * S_i(t-1) + (1 - lam) * s_i(t)       (EWMA, half-life 5 days)
    2. standardise z_i(t) = (S_i - mean_j S_j) / max(std_j S_j, floor)  (relative sentiment)
    3. tilt        w*_i  ~ w_bench_i * exp(kappa * z_i(t))            (positive -> up, negative -> down)
    4. constrain   clip to [1 %, 12 %] and renormalise to 100 %
    5. trade       move at most `max_turnover` towards the target (limits churn & cost)
    6. drift       between rebalances weights drift with prices

Standardising makes the tilt depend on how a stock's news compares with its peers today:
a market-wide mood swing moves every stock equally and so (correctly) changes no weights,
and the `floor` stops tiny differences on quiet days from being amplified.

The same class runs live (``update`` per signal batch) and in the back-test loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import RAW, REBALANCE, TICKERS, UNIVERSE
from src.engine.store import read_daily


def _cap_normalise(w: np.ndarray, lo: float, hi: float, iters: int = 50) -> np.ndarray:
    """Project weights onto {sum = 1, lo <= w <= hi} by iterative clipping."""
    w = np.clip(w, 0, None)
    w = w / w.sum()
    for _ in range(iters):
        clipped = np.clip(w, lo, hi)
        free = (clipped > lo) & (clipped < hi)
        excess = 1.0 - clipped.sum()
        if abs(excess) < 1e-10 or not free.any():
            w = clipped
            break
        clipped[free] += excess * clipped[free] / clipped[free].sum()
        w = clipped
    return w / w.sum()


@dataclass
class IndexRebalancer:
    tickers: list[str] = field(default_factory=lambda: list(TICKERS))
    kappa: float = REBALANCE["kappa"]
    half_life_days: float = REBALANCE["half_life_days"]
    min_weight: float = REBALANCE["min_weight"]
    max_weight: float = REBALANCE["max_weight"]
    max_turnover: float = REBALANCE["max_turnover"]
    min_mentions: int = REBALANCE["min_mentions"]
    z_floor: float = REBALANCE["z_floor"]

    def __post_init__(self) -> None:
        n = len(self.tickers)
        self.bench = np.full(n, 1.0 / n)
        self.weights = self.bench.copy()
        self.ewma = np.zeros(n)
        self.lam = 0.5 ** (1.0 / self.half_life_days)
        self.idx = {t: i for i, t in enumerate(self.tickers)}

    # -- signal subscription -------------------------------------------------
    def update(self, scores: dict[str, float], n_docs: dict[str, int] | None = None, days: int = 1) -> None:
        """Feed one day of sentiment scores (missing tickers count as 0 = no news)."""
        s = np.zeros(len(self.tickers))
        for t, v in scores.items():
            if t in self.idx and (n_docs is None or n_docs.get(t, 0) >= self.min_mentions):
                s[self.idx[t]] = v
        lam = self.lam ** days
        self.ewma = lam * self.ewma + (1 - lam) * s

    def zscores(self) -> np.ndarray:
        return (self.ewma - self.ewma.mean()) / max(self.ewma.std(), self.z_floor)

    def target(self) -> np.ndarray:
        raw = self.bench * np.exp(self.kappa * np.clip(self.zscores(), -3, 3))
        return _cap_normalise(raw, self.min_weight, self.max_weight)

    def rebalance(self) -> float:
        """Move towards the target subject to the turnover limit; returns one-way turnover."""
        tgt = self.target()
        diff = tgt - self.weights
        turnover = 0.5 * np.abs(diff).sum()
        alpha = 1.0 if turnover <= self.max_turnover else self.max_turnover / turnover
        self.weights = self.weights + alpha * diff
        return float(alpha * turnover)

    def drift(self, returns: np.ndarray) -> None:
        w = self.weights * (1 + np.nan_to_num(returns))
        self.weights = w / w.sum()

    def snapshot(self) -> dict[str, float]:
        return {t: round(float(self.weights[i]), 6) for t, i in self.idx.items()}


# ---------------------------------------------------------------------------
# Back-test
# ---------------------------------------------------------------------------
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    px = pd.read_csv(RAW / "prices.csv", parse_dates=["Date"]).set_index("Date")[TICKERS]
    daily = read_daily()
    daily = daily[daily.entity.isin(TICKERS)]
    scores = daily.pivot(index="date", columns="entity", values="score").reindex(columns=TICKERS).fillna(0.0)
    counts = daily.pivot(index="date", columns="entity", values="n_docs").reindex(columns=TICKERS).fillna(0).astype(int)
    return px, scores, counts


def _stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    years = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    return {"total_return_pct": round(100 * (eq.iloc[-1] - 1), 2), "cagr_pct": round(100 * cagr, 2),
            "volatility_pct": round(100 * vol, 2), "sharpe": round(cagr / vol, 3) if vol else 0.0,
            "max_drawdown_pct": round(100 * dd, 2)}


def backtest(start: str | None = None, end: str | None = None, cost_bps: float = REBALANCE["cost_bps"], **params) -> dict:
    """Daily close-to-close back-test. Weights decided with signals available at close t
    (see aggregate.signal_date) earn the returns of t -> t+1. No look-ahead."""
    px, scores, counts = load_inputs()
    rets = px.pct_change()
    dates = px.index[(px.index >= pd.Timestamp(start or scores.index.min())) & (px.index <= pd.Timestamp(end or scores.index.max()))]
    rb = IndexRebalancer(**params)
    cal = pd.date_range(dates[0] - pd.Timedelta(days=7), dates[-1])
    scores = scores.reindex(cal).fillna(0.0)
    counts = counts.reindex(cal).fillna(0).astype(int)

    w_hist, tgt_hist, port, bench, turns = [], [], [], [], []
    last_day = cal[0]
    for d in cal:  # feed every calendar day so weekend news is not lost
        rb.update(scores.loc[d].to_dict(), counts.loc[d].to_dict())
        if d not in dates:
            continue
        last_day = d
        turnover = rb.rebalance()
        w_hist.append(pd.Series(rb.weights.copy(), index=TICKERS, name=d))
        tgt_hist.append(pd.Series(rb.ewma.copy(), index=TICKERS, name=d))
        turns.append(turnover)
        nxt = px.index[px.index.get_loc(d) + 1] if px.index.get_loc(d) + 1 < len(px.index) else None
        if nxt is None:
            break
        r_next = rets.loc[nxt].values
        port.append(pd.Series({"date": nxt, "ret": float(np.nansum(rb.weights * r_next)) - turnover * cost_bps / 1e4}))
        bench.append(pd.Series({"date": nxt, "ret": float(np.nanmean(r_next))}))
        rb.drift(r_next)

    weights = pd.DataFrame(w_hist)
    ewma = pd.DataFrame(tgt_hist)
    perf = pd.DataFrame({"sentiment_index": pd.DataFrame(port).set_index("date").ret,
                         "equal_weight": pd.DataFrame(bench).set_index("date").ret}).astype(float)
    turnover = pd.Series(turns, index=weights.index[: len(turns)])
    active = perf.sentiment_index - perf.equal_weight
    metrics = {
        "period": f"{perf.index.min().date()} to {perf.index.max().date()}",
        "trading_days": int(len(perf)),
        "sentiment_index": _stats(perf.sentiment_index),
        "equal_weight": _stats(perf.equal_weight),
        "avg_daily_turnover_pct": round(100 * turnover.mean(), 2),
        "tracking_error_pct": round(100 * active.std() * np.sqrt(252), 2),
        "information_ratio": round(active.mean() / active.std() * np.sqrt(252), 3) if active.std() else 0.0,
        "hit_rate_pct": round(100 * (active > 0).mean(), 1),
        "avg_max_weight_pct": round(100 * weights.max(axis=1).mean(), 2),
        "avg_min_weight_pct": round(100 * weights.min(axis=1).mean(), 2),
        "params": {"kappa": rb.kappa, "z_floor": rb.z_floor, "half_life_days": rb.half_life_days, "min_weight": rb.min_weight,
                   "max_weight": rb.max_weight, "max_turnover": rb.max_turnover, "cost_bps": cost_bps},
    }
    return {"weights": weights, "ewma": ewma, "perf": perf, "turnover": turnover, "metrics": metrics,
            "last_day": last_day}


def sector_weights(weights: pd.DataFrame) -> pd.DataFrame:
    sec = {t: UNIVERSE[t]["sector"] for t in weights.columns}
    return weights.T.groupby(sec).sum().T


def main() -> None:
    import json

    from src.config import MODELS, PROCESSED
    res = backtest()
    res["weights"].round(6).to_csv(PROCESSED / "index_weights.csv", index_label="date")
    res["perf"].round(8).to_csv(PROCESSED / "index_performance.csv", index_label="date")
    print(json.dumps(res["metrics"], indent=2))
    path = MODELS / "metrics.json"
    m = json.loads(path.read_text()) if path.exists() else {}
    m["module_a_backtest"] = res["metrics"]
    path.write_text(json.dumps(m, indent=2, default=float))


if __name__ == "__main__":
    main()
