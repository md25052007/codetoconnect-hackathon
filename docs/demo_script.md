# Demo video script (target 5 minutes)

Record the screen at 1080p with a voice-over. Upload to YouTube as **Unlisted**, check it plays in an incognito window, and paste the link into the README.

| Time | Show | Say |
|---|---|---|
| 0:00–0:30 | Slide 1 → slide 2 | "Markets move on text, but risk systems need numbers. I built an AI/NLP risk engine that turns news headlines and tweets into structured signals (sentiment, event type and impact), then used them in both downstream modules." |
| 0:30–1:00 | Terminal: `pip install -r requirements.txt`, `python main.py dashboard` (and `python main.py api` in a second tab) | "It runs locally from the README commands. The models and processed signals are committed, so it starts in seconds." |
| 1:00–2:00 | **Risk engine** tab: pick "Russia invades Crimea…", then type your own headline, open "Why this score?". Click **Start replay** for 3 Mar 2014. | "Each text gets a sentiment between −1 and +1, an event class and a 1–10 impact score, and every factor is explained. Here the merged news and Twitter stream replays in real time through the engine." |
| 2:00–2:20 | Browser: `localhost:8000/docs`, run `POST /analyze` | "The same signals are available to any downstream system through a REST API, including a live server-sent-events stream." |
| 2:20–3:20 | **Module A** tab: sector chart, heatmap, then move the date slider and show the headlines table. Change kappa. | "Module A tilts a 20-stock index towards positive news and away from negative news, with caps and a turnover limit. Blue means overweight after good news. It tracks equal weight with 1.5% tracking error; before costs it adds about one point." |
| 3:20–4:20 | **Module B** tab: the trigger timeline, select "Brazil downgraded to junk", show before/after and the waterfall. Then "Type a headline". | "Module B stress-tests an $8.1bn synthetic banking book. When the engine detects a negative, market-wide event above the threshold, for example a credit event with impact above 6, it scales the matching shock scenario by the impact and revalues every trade. This downgrade costs about $460m, mostly from credit spreads and FX." |
| 4:20–4:50 | **Model quality** tab | "The sentiment model is 79–80% accurate versus 49–54% for plain VADER, and higher impact scores line up with bigger real price moves." |
| 4:50–5:00 | Slide 7 | "Limitations: out-of-domain sentiment and a noisy news feed. The next step is FinBERT fine-tuned on world news. Thank you." |

**Before the jury pitch:** have both `python main.py dashboard` and `python main.py api` running, and the repo open.
