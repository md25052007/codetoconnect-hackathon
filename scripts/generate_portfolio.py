"""Generate the synthetic wholesale-banking book used by Module B.

Writes data/portfolio/transactions.csv: one row per trade, with the risk attributes a
simplified sensitivity-based stress test needs (duration, spread duration, beta, PD, LGD,
DV01 / CS01, FX direction). All counterparties are fictional. Seeded -> reproducible.

    python scripts/generate_portfolio.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PORTFOLIO  # noqa: E402

RNG = np.random.default_rng(42)
SECTORS = ["Energy", "Financials", "Technology", "Industrials", "Consumer", "Healthcare", "Utilities", "Real Estate"]
REGIONS = ["North America", "Europe", "Asia Pacific", "Emerging Markets"]
REGION_P = [0.40, 0.30, 0.15, 0.15]
RATINGS = ["AA", "A", "BBB", "BB", "B"]
RATING_P = [0.08, 0.25, 0.37, 0.20, 0.10]
PD_1Y = {"AAA": 0.0001, "AA": 0.0002, "A": 0.0006, "BBB": 0.0020, "BB": 0.0090, "B": 0.0350}  # long-run averages
SPREAD = {"AAA": 30, "AA": 50, "A": 80, "BBB": 140, "BB": 300, "B": 480}
SOVEREIGNS = [("US Treasury", "North America", "AAA", "USD"), ("German Bund", "Europe", "AAA", "EUR"),
              ("UK Gilt", "Europe", "AA", "GBP"), ("Japan JGB", "Asia Pacific", "A", "JPY"),
              ("Italy BTP", "Europe", "BBB", "EUR"), ("India G-Sec", "Emerging Markets", "BBB", "INR"),
              ("Brazil NTN", "Emerging Markets", "BB", "BRL"), ("Indonesia INDOGB", "Emerging Markets", "BBB", "IDR")]
CCY_BY_REGION = {"North America": ["USD"], "Europe": ["EUR", "GBP"], "Asia Pacific": ["JPY", "AUD"],
                 "Emerging Markets": ["INR", "BRL", "MXN", "IDR"]}
NAME_A = ["Northwind", "Bluepeak", "Silverline", "Harbor", "Crescent", "Summit", "Ironbridge", "Redwood",
          "Lakeside", "Granite", "Evergreen", "Atlas", "Pioneer", "Meridian", "Oakridge", "Falcon"]
NAME_B = {"Energy": "Energy", "Financials": "Capital", "Technology": "Systems", "Industrials": "Industries",
          "Consumer": "Brands", "Healthcare": "Health", "Utilities": "Power", "Real Estate": "Properties"}


def _cpty(sector: str) -> str:
    return f"{RNG.choice(NAME_A)} {NAME_B[sector]} (fictional)"


def generate(n_loans=70, n_corp_bonds=45, n_equity=25, n_irs=20, n_cds=15, n_fx=20) -> pd.DataFrame:
    rows = []
    tid = 1000

    def add(**kw):
        nonlocal tid
        tid += 1
        base = {"trade_id": f"T{tid}", "trade_date": str(pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(RNG.integers(0, 365))))[:10],
                "currency": "USD", "notional": 0.0, "market_value": 0.0, "mod_duration": 0.0, "spread_duration": 0.0,
                "convexity": 0.0, "equity_beta": 0.0, "pd_1y": 0.0, "lgd": 0.0, "dv01": 0.0, "cs01": 0.0, "fx_direction": 0}
        base.update(kw)
        rows.append(base)

    for _ in range(n_loans):
        sector = RNG.choice(SECTORS)
        region = RNG.choice(REGIONS, p=REGION_P)
        rating = RNG.choice(RATINGS, p=RATING_P)
        kind = RNG.choice(["Term Loan", "Revolving Credit Facility", "Trade Finance"], p=[0.55, 0.30, 0.15])
        notional = float(RNG.lognormal(np.log(60e6), 0.6))
        drawn = notional * (RNG.uniform(0.3, 0.9) if kind == "Revolving Credit Facility" else 1.0)
        add(asset_class="Loan", instrument=kind, counterparty=_cpty(sector), sector=sector, region=region,
            rating=rating, notional=round(notional, -3), market_value=round(drawn, -3),
            maturity_years=round(float(RNG.uniform(0.5, 1) if kind == "Trade Finance" else RNG.uniform(2, 7)), 1),
            pd_1y=PD_1Y[rating], lgd=0.35 if kind == "Term Loan" else 0.45,
            spread_bps=SPREAD[rating] + int(RNG.integers(-20, 40)))

    for _ in range(n_corp_bonds):
        sector = RNG.choice(SECTORS)
        region = RNG.choice(REGIONS, p=REGION_P)
        rating = RNG.choice(RATINGS, p=RATING_P)
        mv = float(RNG.lognormal(np.log(40e6), 0.5))
        mat = float(RNG.uniform(2, 12))
        dur = mat * 0.85
        add(asset_class="Bond", instrument="Corporate Bond", counterparty=_cpty(sector), sector=sector, region=region,
            rating=rating, notional=round(mv, -3), market_value=round(mv, -3), maturity_years=round(mat, 1),
            mod_duration=round(dur, 2), spread_duration=round(dur, 2), convexity=round(dur ** 2 / 100, 3),
            pd_1y=PD_1Y[rating], lgd=0.6, spread_bps=SPREAD[rating] + int(RNG.integers(-20, 40)))

    for name, region, rating, ccy in SOVEREIGNS:
        mv = float(RNG.lognormal(np.log(180e6), 0.3))
        mat = float(RNG.uniform(3, 15))
        dur = mat * 0.9
        add(asset_class="Bond", instrument="Sovereign Bond", counterparty=name, sector="Sovereign", region=region,
            rating=rating, currency=ccy, notional=round(mv, -3), market_value=round(mv, -3), maturity_years=round(mat, 1),
            mod_duration=round(dur, 2), spread_duration=round(dur, 2) if region == "Emerging Markets" or rating != "AAA" else 0.0,
            convexity=round(dur ** 2 / 100, 3), pd_1y=PD_1Y[rating], lgd=0.6,
            fx_direction=1 if ccy != "USD" else 0)

    for _ in range(n_equity):
        sector = RNG.choice(SECTORS)
        region = RNG.choice(REGIONS, p=REGION_P)
        mv = float(RNG.lognormal(np.log(15e6), 0.5))
        add(asset_class="Equity", instrument="Listed Equity", counterparty=_cpty(sector), sector=sector, region=region,
            rating="NR", notional=round(mv, -3), market_value=round(mv, -3),
            equity_beta=round(float(RNG.uniform(0.7, 1.4)), 2))

    for _ in range(n_irs):
        notional = float(RNG.lognormal(np.log(150e6), 0.5))
        mat = float(RNG.choice([2, 5, 7, 10, 30]))
        direction = RNG.choice(["Pay Fixed", "Receive Fixed"], p=[0.45, 0.55])
        dv01 = notional * mat * 0.88 * 1e-4 * (1 if direction == "Pay Fixed" else -1)  # value change per +1bp
        add(asset_class="Derivative", instrument=f"Interest Rate Swap ({direction})", counterparty=_cpty("Financials"),
            sector="Rates", region=RNG.choice(REGIONS[:2]), rating="A", notional=round(notional, -3),
            market_value=round(float(RNG.normal(0, notional * 0.004)), -3), maturity_years=mat, dv01=round(dv01, 0))

    for _ in range(n_cds):
        sector = RNG.choice(SECTORS)
        notional = float(RNG.lognormal(np.log(50e6), 0.4))
        side = RNG.choice(["Protection Bought", "Protection Sold"], p=[0.6, 0.4])
        mat = 5.0
        cs01 = notional * 4.5 * 1e-4 * (1 if side == "Protection Bought" else -1)  # value change per +1bp spread
        rating = RNG.choice(RATINGS, p=RATING_P)
        add(asset_class="Derivative", instrument=f"Credit Default Swap ({side})", counterparty=_cpty(sector),
            sector=sector, region=RNG.choice(REGIONS, p=REGION_P), rating=rating, notional=round(notional, -3),
            market_value=round(float(RNG.normal(0, notional * 0.003)), -3), maturity_years=mat, cs01=round(cs01, 0))

    for _ in range(n_fx):
        region = RNG.choice(["Europe", "Asia Pacific", "Emerging Markets"])
        ccy = RNG.choice(CCY_BY_REGION[region])
        notional = float(RNG.lognormal(np.log(80e6), 0.5))
        direction = int(RNG.choice([1, -1], p=[0.55, 0.45]))  # +1 = long foreign currency vs USD
        add(asset_class="Derivative", instrument=f"FX Forward ({'Long' if direction > 0 else 'Short'} {ccy}/USD)",
            counterparty=_cpty("Financials"), sector="FX", region=region, rating="A", currency=ccy,
            notional=round(notional, -3), market_value=round(float(RNG.normal(0, notional * 0.005)), -3),
            maturity_years=round(float(RNG.uniform(0.25, 1)), 2), fx_direction=direction)

    df = pd.DataFrame(rows)
    df["maturity_years"] = df["maturity_years"].fillna(0)
    df["spread_bps"] = df["spread_bps"].fillna(0).astype(int)
    return df


if __name__ == "__main__":
    PORTFOLIO.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(PORTFOLIO / "transactions.csv", index=False)
    print(df.groupby(["asset_class", "instrument"]).agg(n=("trade_id", "size"), mv_musd=("market_value", lambda s: round(s.sum() / 1e6, 1))))
    print(f"total market value: {df.market_value.sum() / 1e9:.2f} bn USD, trades: {len(df)}")
