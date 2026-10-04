"""Module B - Event-driven stress testing of a synthetic wholesale-banking book.

Subscribes to Event Classification + Impact Score. When a market-level signal crosses the
trigger threshold for its class (config.STRESS_TRIGGERS, e.g. Geopolitical > 7), the
matching shock scenario is scaled by the impact score and applied to every trade:

    Bonds        dV = MV * ( -ModDur*dr + 0.5*Cvx*dr^2 - SpreadDur*ds )
    Loans        dV = -Drawn * LGD * (PD_stressed - PD)          (expected-loss increase)
    Equity       dV = MV * beta * equity_shock * sector_mult
    IR swaps     dV = DV01 * dr_bps
    CDS          dV = CS01 * ds_bps   (protection buyer gains when spreads widen)
    FX forwards  dV = Notional * direction * fx_move
    FX on non-USD sovereign bonds is applied on top of the bond's rate/spread P&L.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from collections import defaultdict

from src.config import (MARKET, PORTFOLIO, STRESS_CORROBORATION, STRESS_SINGLE_REPORT_OVERRIDE,
                        STRESS_TRIGGERS)
from src.modules.stress.scenarios import Scenario, scenario_for, severity_factor

ASSUMED_CAPITAL_RATIO = 0.10  # capital held against the book, for a simple "capital at risk" metric


def load_portfolio() -> pd.DataFrame:
    return pd.read_csv(PORTFOLIO / "transactions.csv")


def apply_scenario(pf: pd.DataFrame, sc: Scenario) -> pd.DataFrame:
    df = pf.copy()
    dr = sc.rates_bps / 1e4
    sec_eq = df.sector.map(sc.sector_equity_mult).fillna(1.0)
    sec_cr = df.sector.map(sc.sector_credit_mult).fillna(1.0)
    ds_bps = df.rating.map(sc.spread_bps).fillna(0.0) * sec_cr
    em_sov = (df.instrument == "Sovereign Bond") & (df.region == "Emerging Markets")
    ds_bps = ds_bps + np.where(em_sov, sc.em_sovereign_bps, 0.0)
    ds_bps = np.where((df.instrument == "Sovereign Bond") & (df.rating == "AAA"), 0.0, ds_bps)  # safe havens
    fx = df.currency.map(sc.fx_pct).fillna(0.0) / 100

    pnl = np.zeros(len(df))
    comp = {k: np.zeros(len(df)) for k in ("rates", "credit_spread", "default_loss", "equity", "fx")}

    bond = df.asset_class == "Bond"
    comp["rates"][bond] = df.market_value[bond] * (-df.mod_duration[bond] * dr + 0.5 * df.convexity[bond] * 100 * dr ** 2)
    comp["credit_spread"][bond] = -df.market_value[bond] * df.spread_duration[bond] * ds_bps[bond] / 1e4
    comp["fx"][bond] = df.market_value[bond] * df.fx_direction[bond] * fx[bond]

    loan = df.asset_class == "Loan"
    pd_mult = 1 + (sc.pd_multiplier - 1) * sec_cr
    stressed_pd = np.minimum(df.pd_1y * pd_mult, 1.0)
    comp["default_loss"][loan] = -df.market_value[loan] * df.lgd[loan] * (stressed_pd[loan] - df.pd_1y[loan])
    # CDS: protection sold also takes a jump-to-default style hit through the PD channel
    eq = df.asset_class == "Equity"
    comp["equity"][eq] = df.market_value[eq] * df.equity_beta[eq] * (sc.equity_pct / 100) * sec_eq[eq]

    irs = df.instrument.str.startswith("Interest Rate Swap")
    comp["rates"][irs] = df.dv01[irs] * sc.rates_bps
    cds = df.instrument.str.startswith("Credit Default Swap")
    comp["credit_spread"][cds] = df.cs01[cds] * ds_bps[cds]
    fxf = df.instrument.str.startswith("FX Forward")
    comp["fx"][fxf] = df.notional[fxf] * df.fx_direction[fxf] * fx[fxf]

    for k, v in comp.items():
        df[f"pnl_{k}"] = np.round(v, 0)
        pnl += v
    df["pnl"] = np.round(pnl, 0)
    df["stressed_value"] = df.market_value + df.pnl
    df["spread_shock_bps"] = np.round(ds_bps, 1)
    return df


@dataclass
class StressResult:
    event_type: str
    impact: float
    severity: float
    scenario: dict
    trigger_text: str
    trigger_time: str
    value_before: float
    value_after: float
    pnl: float
    pnl_pct: float
    capital_at_risk_pct: float
    by_asset_class: list[dict]
    by_sector: list[dict]
    by_risk_factor: dict
    worst_positions: list[dict]
    positions: pd.DataFrame

    def summary(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "positions"}
        return d


def run_stress(event_type: str, impact: float, trigger_text: str = "", trigger_time: str | None = None,
               scenario: Scenario | None = None, portfolio: pd.DataFrame | None = None) -> StressResult:
    pf = portfolio if portfolio is not None else load_portfolio()
    sc = scenario or scenario_for(event_type, impact)
    res = apply_scenario(pf, sc)
    before, pnl = float(res.market_value.sum()), float(res.pnl.sum())

    def grp(col):
        g = res.groupby(col).agg(value_before=("market_value", "sum"), pnl=("pnl", "sum")).reset_index()
        g["value_after"] = g.value_before + g.pnl
        g["pnl_pct"] = np.where(g.value_before.abs() > 1, 100 * g.pnl / g.value_before.abs(), 0.0)
        return g.round(2).sort_values("pnl").to_dict(orient="records")

    worst = res.nsmallest(10, "pnl")[["trade_id", "instrument", "counterparty", "sector", "region", "rating",
                                       "market_value", "pnl"]].to_dict(orient="records")
    return StressResult(
        event_type=event_type, impact=impact, severity=severity_factor(impact) if scenario is None else 1.0,
        scenario=sc.to_dict(), trigger_text=trigger_text, trigger_time=trigger_time or datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        value_before=round(before, 0), value_after=round(before + pnl, 0), pnl=round(pnl, 0),
        pnl_pct=round(100 * pnl / before, 3), capital_at_risk_pct=round(100 * -pnl / (ASSUMED_CAPITAL_RATIO * before), 2),
        by_asset_class=grp("asset_class"), by_sector=grp("sector"),
        by_risk_factor={k.replace("pnl_", ""): round(float(res[k].sum()), 0)
                        for k in res.columns if k.startswith("pnl_")},
        worst_positions=worst, positions=res,
    )


# ---------------------------------------------------------------------------
# Signal subscription
# ---------------------------------------------------------------------------
def is_trigger(event_type: str, impact: float, entity_type: str = "market", sentiment: float = -1.0) -> bool:
    """High-impact, market-level, non-positive news of a stress-relevant class.
    (A ceasefire is high-impact geopolitical news too, but it is not a risk event.)"""
    thr = STRESS_TRIGGERS.get(event_type)
    return (thr is not None and impact > thr and entity_type in ("market", "sector") and sentiment <= 0.0)


def corroborated(event_type: str, impact: float, n_reports_today: int) -> bool:
    need = STRESS_CORROBORATION.get(event_type, 1)
    return n_reports_today >= need or impact >= STRESS_SINGLE_REPORT_OVERRIDE


class StressMonitor:
    """Streams signals; fires at most one stress test per event class per day, once the
    trigger is met and (for noisy classes) corroborated by several reports."""

    def __init__(self) -> None:
        self.portfolio = load_portfolio()
        self.fired: set[tuple[str, str]] = set()
        self.counts: dict[tuple[str, str], int] = defaultdict(int)

    def on_signal(self, sig: dict) -> StressResult | None:
        if not is_trigger(sig["event_type"], sig["impact"], sig.get("entity_type", "market"), sig.get("sentiment", -1.0)):
            return None
        key = (sig["event_type"], str(sig["timestamp"])[:10])
        self.counts[key] += 1
        if key in self.fired or not corroborated(sig["event_type"], sig["impact"], self.counts[key]):
            return None
        self.fired.add(key)
        return run_stress(sig["event_type"], sig["impact"], sig.get("text", ""), str(sig["timestamp"]),
                          portfolio=self.portfolio)


def triggered_events(signals: pd.DataFrame) -> pd.DataFrame:
    """Historical stress triggers: one per class per day (highest-impact report), after
    the threshold, negative-sentiment and corroboration rules."""
    s = signals[signals.entity == MARKET].copy()
    s = s[[is_trigger(e, i, "market", x) for e, i, x in zip(s.event_type, s.impact, s.sentiment)]]
    s["day"] = s.timestamp.dt.normalize()
    s["reports"] = s.groupby(["event_type", "day"]).doc_id.transform("count")
    s = s.sort_values("impact", ascending=False).drop_duplicates(["event_type", "day"])
    s = s[[corroborated(e, i, n) for e, i, n in zip(s.event_type, s.impact, s.reports)]]
    return s.sort_values("timestamp")[["timestamp", "event_type", "impact", "sentiment", "reports", "text",
                                       "doc_id"]].reset_index(drop=True)
