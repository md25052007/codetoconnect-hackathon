"""Central configuration: paths, the mock index universe, event taxonomy and thresholds."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
TRAIN = DATA / "train"
EVAL = DATA / "eval"
PROCESSED = DATA / "processed"
PORTFOLIO = DATA / "portfolio"
MODELS = ROOT / "models"

# Signal outputs consumed by downstream modules (the engine's "contract")
SIGNALS_JSONL = PROCESSED / "signals.jsonl.gz"      # one row per document
DAILY_SIGNALS_CSV = PROCESSED / "daily_signals.csv"  # aggregated per entity per day
SIGNALS_DB = PROCESSED / "signals.db"                # SQLite copy used by the API

# Demo window (bounded by the open StockNet tweet dataset)
START_DATE = "2014-01-01"
END_DATE = "2016-03-31"

# ---------------------------------------------------------------------------
# Mock index: 20 S&P 100 constituents
# ---------------------------------------------------------------------------
UNIVERSE: dict[str, dict] = {
    "AAPL": {"name": "Apple", "sector": "Technology", "aliases": ["apple", "iphone", "ipad", "tim cook", "macbook", "itunes"]},
    "MSFT": {"name": "Microsoft", "sector": "Technology", "aliases": ["microsoft", "windows", "xbox", "satya nadella", "azure"]},
    "GOOG": {"name": "Alphabet (Google)", "sector": "Technology", "aliases": ["google", "alphabet", "android", "youtube"]},
    "FB":   {"name": "Meta (Facebook)", "sector": "Technology", "aliases": ["facebook", "zuckerberg", "instagram", "whatsapp"]},
    "INTC": {"name": "Intel", "sector": "Technology", "aliases": ["intel"]},
    "CSCO": {"name": "Cisco", "sector": "Technology", "aliases": ["cisco"]},
    "AMZN": {"name": "Amazon", "sector": "Consumer", "aliases": ["amazon", "bezos", "aws", "kindle"]},
    "WMT":  {"name": "Walmart", "sector": "Consumer", "aliases": ["walmart", "wal-mart"]},
    "KO":   {"name": "Coca-Cola", "sector": "Consumer", "aliases": ["coca-cola", "coca cola", "coke"]},
    "DIS":  {"name": "Walt Disney", "sector": "Consumer", "aliases": ["disney", "espn", "pixar", "marvel", "star wars"]},
    "JPM":  {"name": "JPMorgan Chase", "sector": "Financials", "aliases": ["jpmorgan", "jp morgan", "jamie dimon", "chase bank"]},
    "BAC":  {"name": "Bank of America", "sector": "Financials", "aliases": ["bank of america", "bofa", "merrill lynch"]},
    "WFC":  {"name": "Wells Fargo", "sector": "Financials", "aliases": ["wells fargo"]},
    "C":    {"name": "Citigroup", "sector": "Financials", "aliases": ["citigroup", "citibank", "citi "]},
    "XOM":  {"name": "Exxon Mobil", "sector": "Energy", "aliases": ["exxon", "exxonmobil"]},
    "CVX":  {"name": "Chevron", "sector": "Energy", "aliases": ["chevron"]},
    "JNJ":  {"name": "Johnson & Johnson", "sector": "Healthcare", "aliases": ["johnson & johnson", "johnson and johnson", "j&j"]},
    "PFE":  {"name": "Pfizer", "sector": "Healthcare", "aliases": ["pfizer"]},
    "BA":   {"name": "Boeing", "sector": "Industrials", "aliases": ["boeing", "737", "dreamliner"]},
    "GE":   {"name": "General Electric", "sector": "Industrials", "aliases": ["general electric", "jeff immelt"]},
}
TICKERS = list(UNIVERSE)

# Sector-level keywords: a headline about "oil prices" is relevant to every Energy name.
SECTOR_KEYWORDS: dict[str, list[str]] = {
    "Energy": ["oil", "crude", "opec", "brent", "natural gas", "petroleum", "refinery", "pipeline"],
    "Financials": ["bank", "banks", "banking", "lender", "wall street", "bailout", "fed ", "federal reserve", "interest rate"],
    "Technology": ["tech giant", "silicon valley", "semiconductor", "smartphone", "cyber", "hack"],
    "Healthcare": ["drug", "pharma", "vaccine", "fda", "ebola", "pandemic", "epidemic"],
    "Industrials": ["airline", "aircraft", "plane crash", "manufacturing"],
    "Consumer": ["retail", "consumer spending", "shoppers"],
}

MARKET = "MARKET"  # entity label for market-wide / macro signals

# ---------------------------------------------------------------------------
# Event taxonomy and impact model parameters
# ---------------------------------------------------------------------------
EVENT_TYPES = [
    "Geopolitical",
    "Macroeconomic",
    "Credit Event",
    "Merger/Acquisition",
    "Product Launch",
    "Earnings",
    "Regulatory/Legal",
    "Other",
]

# Base severity of each event class (0-1). Credit and geopolitical shocks are the
# most likely to propagate across markets; product news is usually idiosyncratic.
EVENT_SEVERITY = {
    "Credit Event": 0.85,
    "Geopolitical": 0.80,
    "Macroeconomic": 0.75,
    "Merger/Acquisition": 0.60,
    "Regulatory/Legal": 0.55,
    "Earnings": 0.55,
    "Product Launch": 0.40,
    "Other": 0.20,
}

SOURCE_CREDIBILITY = {"news": 1.0, "social": 0.75}

# Module B: stress test is triggered when a signal of this class reaches this impact
STRESS_TRIGGERS = {
    "Geopolitical": 7,
    "Macroeconomic": 7,
    "Credit Event": 6,
    "Regulatory/Legal": 8,
    "Merger/Acquisition": 9,
}

# Module A: rebalancer parameters
REBALANCE = {
    "kappa": 1.5,          # tilt strength: weight multiplier = exp(kappa * score)
    "half_life_days": 3,   # EWMA memory of the sentiment signal
    "min_weight": 0.01,
    "max_weight": 0.12,
    "max_turnover": 0.20,  # max one-way turnover per rebalance
    "cost_bps": 5,         # transaction cost per unit turnover
    "min_mentions": 2,     # ignore days with fewer than this many documents for a stock
}
