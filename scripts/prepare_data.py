"""Build the compact, committed datasets in data/ from the original public sources.

You only need this if you want to rebuild data/ from scratch; the outputs are
already committed. Sources (all public):

  * StockNet (Xu & Cohen, ACL 2018, MIT licence) - real tweets + daily prices
        git clone --depth 1 https://github.com/yumoxu/stocknet-dataset
  * "Daily News for Stock Market Prediction" (Kaggle, Aaron7sun) - Reddit r/worldnews
    top-25 headlines per day; mirrored at https://github.com/victorwlu/news-stock-returns
  * Financial PhraseBank v1.0 (Malo et al. 2014, CC BY-NC-SA 3.0) - expert-labelled
    financial sentences; mirrored at https://github.com/Hugoverissimo21/finphrasebank-sentiment

Usage:
    python scripts/prepare_data.py --stocknet PATH --reddit PATH/RedditNews.csv \
        --djia PATH/DJIA_table.csv --phrasebank PATH/FinancialPhraseBank-v1.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import END_DATE, RAW, START_DATE, TICKERS, TRAIN  # noqa: E402


def build_tweets(stocknet: Path) -> pd.DataFrame:
    rows = []
    for t in TICKERS:
        folder = stocknet / "tweet" / "raw" / t
        for f in sorted(folder.iterdir()):
            for line in f.read_text(encoding="utf-8").splitlines():
                try:
                    tw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if tw.get("lang") not in (None, "en"):
                    continue
                u = tw.get("user", {})
                rows.append({
                    "id": tw["id_str"],
                    "timestamp": pd.to_datetime(tw["created_at"], format="%a %b %d %H:%M:%S %z %Y"),
                    "ticker": t,
                    "text": " ".join(tw["text"].split()),
                    "followers": u.get("followers_count", 0),
                    "verified": bool(u.get("verified", False)),
                    "retweets": tw.get("retweet_count", 0),
                })
    df = pd.DataFrame(rows)
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df[(df.timestamp >= START_DATE) & (df.timestamp < pd.Timestamp(END_DATE) + pd.Timedelta(days=1))]
    # drop exact duplicate texts per ticker (spam bots / copy-paste)
    df = df.drop_duplicates(["ticker", "text"]).sort_values("timestamp")
    return df.reset_index(drop=True)


def build_news(reddit_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(reddit_csv)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df.Date >= START_DATE) & (df.Date <= END_DATE)].copy()
    df["News"] = df["News"].astype(str).str.replace(r'^b["\']|["\']$', "", regex=True).str.strip()
    df = df.drop_duplicates("News")
    # headlines carry only a date; spread them through the trading day so the replay
    # stream interleaves news and tweets realistically (rank 0 = top-voted headline)
    df["rank"] = df.groupby("Date").cumcount()
    df["timestamp"] = df["Date"] + pd.to_timedelta(13 + df["rank"] * 0.33, unit="h")
    out = df.rename(columns={"News": "text"})[["timestamp", "text", "rank"]]
    out.insert(0, "id", ["news-%d" % i for i in range(len(out))])
    return out.sort_values("timestamp").reset_index(drop=True)


def build_prices(stocknet: Path, djia_csv: Path) -> pd.DataFrame:
    frames = []
    for t in TICKERS:
        p = pd.read_csv(stocknet / "price" / "raw" / f"{t}.csv", parse_dates=["Date"])
        frames.append(p[["Date", "Adj Close"]].rename(columns={"Adj Close": t}).set_index("Date"))
    d = pd.read_csv(djia_csv, parse_dates=["Date"])[["Date", "Adj Close"]].rename(columns={"Adj Close": "DJIA"})
    frames.append(d.set_index("Date"))
    px = pd.concat(frames, axis=1, sort=True)
    px = px.loc[pd.Timestamp(START_DATE) - pd.Timedelta(days=10): pd.Timestamp(END_DATE) + pd.Timedelta(days=10)]
    return px.dropna(how="all").round(4)


def build_phrasebank(folder: Path) -> pd.DataFrame:
    """Keep each sentence once, with the strictest annotator-agreement level it appears in."""
    levels = [("AllAgree", 1.0), ("75Agree", 0.75), ("66Agree", 0.66), ("50Agree", 0.5)]
    seen: dict[str, tuple[str, float]] = {}
    for name, agree in levels:
        for line in (folder / f"Sentences_{name}.txt").read_text(encoding="latin-1").splitlines():
            if "@" not in line:
                continue
            text, label = line.rsplit("@", 1)
            text = text.strip()
            if text not in seen:
                seen[text] = (label.strip(), agree)
    return pd.DataFrame([(t, l, a) for t, (l, a) in seen.items()], columns=["text", "label", "agreement"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocknet", type=Path, required=True)
    ap.add_argument("--reddit", type=Path, required=True)
    ap.add_argument("--djia", type=Path, required=True)
    ap.add_argument("--phrasebank", type=Path, required=True)
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    TRAIN.mkdir(parents=True, exist_ok=True)

    tw = build_tweets(a.stocknet)
    tw.to_csv(RAW / "tweets_stocknet.csv.gz", index=False)
    print(f"tweets: {len(tw):,} rows")

    nw = build_news(a.reddit)
    nw.to_csv(RAW / "news_reddit_worldnews.csv.gz", index=False)
    print(f"news:   {len(nw):,} rows")

    px = build_prices(a.stocknet, a.djia)
    px.to_csv(RAW / "prices.csv")
    print(f"prices: {px.shape}")

    pb = build_phrasebank(a.phrasebank)
    pb.to_csv(TRAIN / "financial_phrasebank.csv", index=False)
    print(f"phrasebank: {len(pb):,} sentences\n{pb.label.value_counts()}")


if __name__ == "__main__":
    main()
