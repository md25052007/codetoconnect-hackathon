"""Shock library: one severe-case scenario per event class, scaled by the Impact Score.

Shock definitions are simplified versions of the ones regulators use (e.g. EBA / Fed
CCAR adverse scenarios): an equity move, a parallel rate shift, credit-spread widening by
rating bucket, FX moves against USD, and a multiplier on probabilities of default.
Sector multipliers express who is hit harder (e.g. energy benefits from a geopolitical
oil spike; banks suffer most in a credit event).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field


@dataclass
class Scenario:
    name: str
    description: str
    equity_pct: float                      # e.g. -12 = equities fall 12 %
    rates_bps: float                       # parallel shift of the yield curve
    spread_bps: dict[str, float]           # widening by rating bucket
    em_sovereign_bps: float                # extra widening for EM sovereigns
    fx_pct: dict[str, float]               # currency move vs USD (+ = currency strengthens)
    pd_multiplier: float                   # stressed PD = PD x multiplier (sector-adjusted)
    sector_equity_mult: dict[str, float] = field(default_factory=dict)
    sector_credit_mult: dict[str, float] = field(default_factory=dict)

    def scaled(self, factor: float) -> "Scenario":
        s = deepcopy(self)
        s.equity_pct *= factor
        s.rates_bps *= factor
        s.spread_bps = {k: v * factor for k, v in s.spread_bps.items()}
        s.em_sovereign_bps *= factor
        s.fx_pct = {k: v * factor for k, v in s.fx_pct.items()}
        s.pd_multiplier = 1 + (s.pd_multiplier - 1) * factor
        return s

    def to_dict(self) -> dict:
        return asdict(self)


SPREADS = lambda ig, bbb, hy: {"AAA": ig * 0.5, "AA": ig * 0.7, "A": ig, "BBB": bbb, "BB": hy, "B": hy * 1.4}  # noqa: E731

SCENARIOS: dict[str, Scenario] = {
    "Geopolitical": Scenario(
        name="Geopolitical shock",
        description="Armed conflict / sanctions: risk-off, flight to quality, oil spike, EM capital flight.",
        equity_pct=-12, rates_bps=-30, spread_bps=SPREADS(50, 90, 250), em_sovereign_bps=250,
        fx_pct={"EUR": -3, "GBP": -3, "JPY": 4, "AUD": -6, "INR": -7, "BRL": -10, "MXN": -8, "IDR": -8},
        pd_multiplier=1.6,
        sector_equity_mult={"Energy": -0.4, "Industrials": 1.2, "Financials": 1.3, "Consumer": 1.1,
                            "Healthcare": 0.6, "Utilities": 0.5, "Technology": 1.1, "Real Estate": 1.0},
        sector_credit_mult={"Energy": 0.7, "Industrials": 1.3, "Consumer": 1.2},
    ),
    "Macroeconomic": Scenario(
        name="Monetary tightening / growth shock",
        description="Inflation surprise: central banks hike +200 bp, equities -10 %, spreads widen, USD rallies.",
        equity_pct=-10, rates_bps=200, spread_bps=SPREADS(40, 80, 180), em_sovereign_bps=150,
        fx_pct={"EUR": -3, "GBP": -3, "JPY": -2, "AUD": -5, "INR": -5, "BRL": -8, "MXN": -6, "IDR": -6},
        pd_multiplier=1.4,
        sector_equity_mult={"Technology": 1.4, "Real Estate": 1.5, "Utilities": 1.2, "Financials": 0.7,
                            "Energy": 0.9, "Healthcare": 0.7, "Consumer": 1.1, "Industrials": 1.0},
        sector_credit_mult={"Real Estate": 1.6, "Consumer": 1.2},
    ),
    "Credit Event": Scenario(
        name="Credit event / default contagion",
        description="Large default or sovereign debt crisis: spreads gap wider, defaults rise, banks hit hardest.",
        equity_pct=-15, rates_bps=-50, spread_bps=SPREADS(100, 180, 450), em_sovereign_bps=350,
        fx_pct={"EUR": -5, "GBP": -4, "JPY": 5, "AUD": -6, "INR": -6, "BRL": -12, "MXN": -8, "IDR": -9},
        pd_multiplier=2.5,
        sector_equity_mult={"Financials": 1.6, "Real Estate": 1.3, "Healthcare": 0.6, "Utilities": 0.6},
        sector_credit_mult={"Financials": 1.5, "Real Estate": 1.4},
    ),
    "Regulatory/Legal": Scenario(
        name="Regulatory crackdown",
        description="Major fines / new regulation on a sector: targeted equity and spread repricing.",
        equity_pct=-5, rates_bps=0, spread_bps=SPREADS(15, 30, 80), em_sovereign_bps=20,
        fx_pct={}, pd_multiplier=1.15,
        sector_equity_mult={"Financials": 1.8, "Technology": 1.6, "Healthcare": 1.4, "Utilities": 0.5},
        sector_credit_mult={"Financials": 1.5, "Technology": 1.3},
    ),
    "Merger/Acquisition": Scenario(
        name="Leveraged M&A wave",
        description="Large debt-funded deal: acquirer leverage rises, HY spreads widen modestly.",
        equity_pct=-2, rates_bps=0, spread_bps=SPREADS(10, 25, 60), em_sovereign_bps=0,
        fx_pct={}, pd_multiplier=1.1,
    ),
    "Assignment example": Scenario(
        name="Equities -10 %, rates +2 %",
        description="The example shock set given in the case study brief.",
        equity_pct=-10, rates_bps=200, spread_bps=SPREADS(0, 0, 0), em_sovereign_bps=0, fx_pct={}, pd_multiplier=1.0,
    ),
}


def severity_factor(impact: float) -> float:
    """Impact 7 -> 50 % of the severe scenario, impact 10 -> 100 %. Linear in between."""
    return round(max(0.25, min(1.0, 0.5 + (impact - 7.0) / 6.0)), 3)


def scenario_for(event_type: str, impact: float) -> Scenario:
    base = SCENARIOS.get(event_type, SCENARIOS["Macroeconomic"])
    return base.scaled(severity_factor(impact))
