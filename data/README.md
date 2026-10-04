# Data

All data is public or synthetic. No proprietary or client data is used.

| File | Rows | Source | Licence | Used for |
|---|---|---|---|---|
| `raw/tweets_stocknet.csv.gz` | 59,777 | Real tweets mentioning the 20 index stocks, Jan 2014 – Mar 2016. From the StockNet dataset (Xu & Cohen, ACL 2018), github.com/yumoxu/stocknet-dataset | MIT | Source #1: social media |
| `raw/news_reddit_worldnews.csv.gz` | 20,485 | Top-25 daily Reddit r/worldnews headlines, same period. From Kaggle's "Daily News for Stock Market Prediction" (Aaron7sun) | CC BY-NC-SA 4.0 | Source #2: news feed |
| `raw/prices.csv` | 577 days | Daily adjusted closes for the 20 stocks (StockNet / Yahoo Finance) and the DJIA | MIT / public | Back-testing Module A and validating the impact score |
| `train/financial_phrasebank.csv` | 4,838 | Financial PhraseBank v1.0 (Malo et al., 2014): financial news sentences labelled by 16 domain experts | CC BY-NC-SA 3.0 | Training and testing the sentiment model |
| `eval/event_gold.csv` | 200 | Headlines and tweets from the two sources, annotated with an event type (and, for the 92 market-relevant ones, sentiment) with AI assistance | this repo | Evaluating the event classifier |
| `portfolio/transactions.csv` | synthetic | Wholesale banking trades (loans, bonds, IRS, CDS, FX forwards, equities) made by `scripts/generate_portfolio.py`, seed 42 | this repo | Module B portfolio |
| `processed/*` | generated | Engine output (`python main.py pipeline`) | this repo | Signals consumed by the modules, the API and the dashboard |

Rebuild `raw/` and `train/` from the original sources with `scripts/prepare_data.py` (see its docstring).

**Assumptions**
- Reddit headlines only have a date. They are spread through the day (13:00 UTC onwards, by vote rank) so the replay stream can mix news and tweets.
- Tweets are English only, with exact duplicates per ticker removed.
- Prices are adjusted closes, so splits and dividends are already accounted for.
