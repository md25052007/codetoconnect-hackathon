"""Sentiment scoring in [-1, 1].

Three back-ends, combined in an ensemble:

1. ``FinSentModel`` - TF-IDF (word + character n-grams) + logistic regression trained on
   14k labelled examples: Financial PhraseBank (expert-labelled news sentences) and
   Twitter Financial News Sentiment (labelled finance tweets). Understands phrasing such
   as "operating profit fell to EUR 3m" that general-purpose lexicons miss.
2. ``FinLexicon`` - VADER (built for social media: slang, emphasis, emoji, negation)
   extended with ~80 finance/trader terms (bullish, downgrade, default, sell-off ...).
3. ``FinBERTSentiment`` (optional) - ProsusAI/finbert via HuggingFace transformers. Used
   automatically when ``RISK_ENGINE_FINBERT=1`` and transformers is installed.

score = P(positive) - P(negative). The ensemble weight of the lexicon and the neutral band
are tuned per source (news vs social) on out-of-fold predictions (models/sentiment_ensemble.json).
Geopolitical / macro / credit texts are outside the model's training domain, so for those
the lexicon is up-weighted (domain-aware routing, evaluated in scripts/evaluate.py).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import joblib
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import MODELS
from src.engine.nlp.text import clean

SENTIMENT_MODEL_PATH = MODELS / "finsent_tfidf_lr.joblib"

# Finance / trading vocabulary added to VADER's lexicon (valence on VADER's -4..4 scale)
FINANCE_LEXICON = {
    "bullish": 2.5, "bearish": -2.5, "bull": 1.5, "bear": -1.5, "upgrade": 2.0, "upgraded": 2.0,
    "upgrades": 2.0, "downgrade": -2.0, "downgraded": -2.2, "downgrades": -2.0, "outperform": 2.0,
    "underperform": -2.0, "overweight": 1.5, "underweight": -1.5, "beat": 1.2, "beats": 1.5,
    "miss": -1.5, "misses": -1.8, "missed": -1.8, "soar": 2.5, "soars": 2.5, "soared": 2.5,
    "surge": 2.2, "surges": 2.2, "surged": 2.2, "rally": 2.0, "rallies": 2.0, "jump": 1.5,
    "jumps": 1.5, "gain": 1.5, "gains": 1.5, "rebound": 1.5, "breakout": 1.5, "record": 1.0,
    "plunge": -3.0, "plunges": -3.0, "plunged": -3.0, "plummet": -3.0, "plummets": -3.0,
    "slump": -2.5, "slumps": -2.5, "tumble": -2.5, "tumbles": -2.5, "sink": -2.0, "sinks": -2.0,
    "selloff": -2.5, "sell-off": -2.5, "crash": -3.2, "crashes": -3.2, "collapse": -3.0,
    "default": -3.0, "defaults": -3.0, "bankrupt": -3.5, "bankruptcy": -3.5, "insolvent": -3.2,
    "lawsuit": -2.0, "sued": -2.0, "fraud": -3.2, "probe": -1.5, "fine": -0.5, "fined": -2.2,
    "sanctions": -2.0, "recession": -3.0, "layoffs": -2.5, "layoff": -2.5, "cuts": -1.2,
    "inflation": -1.0, "unemployment": -1.5, "war": -3.0, "invasion": -3.0, "missile": -2.0,
    "terror": -3.0, "stimulus": 1.0, "buyback": 1.5, "dividend": 1.0, "profit": 1.5,
    "profitable": 2.0, "loss": -1.8, "losses": -2.0, "writedown": -2.2, "write-down": -2.2,
    "moon": 2.0, "mooning": 2.5, "dump": -2.0, "dumping": -2.0, "puts": -1.0, "short": -0.8,
    "shorts": -0.8, "overvalued": -1.5, "undervalued": 1.5, "expands": 1.0, "growth": 1.2,
    "weak": -1.5, "weaker": -1.5, "strong": 1.5, "stronger": 1.5, "warning": -1.5, "warns": -1.8,
    "volatile": -1.0, "turmoil": -2.5, "crisis": -3.0, "bailout": -1.5, "halted": -1.5,
    "invade": -2.5, "invades": -2.5, "invaded": -2.5, "airstrike": -2.5, "airstrikes": -2.5,
    "shelling": -2.5, "escalates": -1.5, "escalation": -2.0, "junk": -2.0, "jobless": -1.5,
    "ceasefire": 1.5, "truce": 1.5, "hyperinflation": -3.0, "devaluation": -2.0, "shutdown": -2.0,
}

# Event classes outside the supervised model's training domain (company news & market
# tweets). For these the general-language lexicon ("killed", "invades", "collapse") is
# more reliable, so it receives a higher weight.
OUT_OF_DOMAIN_EVENTS = {"Geopolitical", "Macroeconomic", "Credit Event"}
OOD_LEXICON_WEIGHT = 0.8
OOD_NEUTRAL_BAND = 0.2


class FinLexicon:
    def __init__(self) -> None:
        self.vader = SentimentIntensityAnalyzer()
        self.vader.lexicon.update(FINANCE_LEXICON)

    def score(self, texts: list[str]) -> np.ndarray:
        return np.array([self.vader.polarity_scores(clean(t))["compound"] for t in texts])


class FinSentModel:
    """Wrapper around the trained scikit-learn pipeline (see src/engine/train.py)."""

    def __init__(self, path=SENTIMENT_MODEL_PATH) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run `python main.py train` first")
        self.pipe = joblib.load(path)
        self.classes = list(self.pipe.classes_)

    def proba(self, texts: list[str]) -> np.ndarray:
        return self.pipe.predict_proba([clean(t) for t in texts])

    def score(self, texts: list[str]) -> np.ndarray:
        p = self.proba(texts)
        return p[:, self.classes.index("positive")] - p[:, self.classes.index("negative")]


class FinBERTSentiment:  # pragma: no cover - optional heavy dependency
    def __init__(self) -> None:
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)

    def score(self, texts: list[str]) -> np.ndarray:
        out = []
        for res in self.pipe([clean(t) for t in texts], batch_size=32):
            d = {r["label"].lower(): r["score"] for r in res}
            out.append(d.get("positive", 0) - d.get("negative", 0))
        return np.array(out)


ENSEMBLE_PATH = MODELS / "sentiment_ensemble.json"


class SentimentEnsemble:
    # (model weight, lexicon weight) per source; overwritten by weights tuned in train.py
    WEIGHTS = {"news": (0.8, 0.2), "social": (0.7, 0.3)}

    def __init__(self) -> None:
        if ENSEMBLE_PATH.exists():
            tuned = json.loads(ENSEMBLE_PATH.read_text())
            self.WEIGHTS = {k: (1 - v["lexicon_weight"], v["lexicon_weight"]) for k, v in tuned.items()}
            self.bands = {k: v["neutral_band"] for k, v in tuned.items()}
        else:
            self.bands = {"news": 0.15, "social": 0.15}
        self.lexicon = FinLexicon()
        self.model = None
        if os.getenv("RISK_ENGINE_FINBERT") == "1":
            try:
                self.model = FinBERTSentiment()
                self.backend = "finbert+lexicon"
            except Exception as exc:  # noqa: BLE001
                print(f"[sentiment] FinBERT unavailable ({exc}); falling back to TF-IDF model")
        if self.model is None:
            try:
                self.model = FinSentModel()
                self.backend = "tfidf-lr+lexicon"
            except FileNotFoundError:
                self.backend = "lexicon-only"

    def score(self, texts: list[str], sources: list[str], event_types: list[str] | None = None) -> tuple[np.ndarray, dict]:
        """Return calibrated scores in [-1, 1] where 0 means neutral.

        The raw ensemble output is passed through a dead-zone transform using the tuned
        neutral band b:  s = sign(raw) * max(0, |raw| - b) / (1 - b).  Scores inside the
        band map to exactly 0, so downstream modules never react to neutral text.
        """
        lex = self.lexicon.score(texts)
        mod = self.model.score(texts) if self.model is not None else np.zeros_like(lex)
        if self.model is None:
            w = np.tile([0.0, 1.0], (len(texts), 1))
        else:
            w = np.array([self.WEIGHTS.get(s, (0.8, 0.2)) for s in sources])
        band = np.array([self.bands.get(s, 0.15) for s in sources])
        if event_types is not None and self.model is not None:
            ood = np.array([e in OUT_OF_DOMAIN_EVENTS for e in event_types])
            w[ood] = [1 - OOD_LEXICON_WEIGHT, OOD_LEXICON_WEIGHT]
            band[ood] = OOD_NEUTRAL_BAND
        raw = np.clip(w[:, 0] * mod + w[:, 1] * lex, -1, 1)
        cal = np.sign(raw) * np.maximum(0.0, np.abs(raw) - band) / (1 - band)
        return np.round(cal, 4), {"lexicon": lex, "model": mod, "raw": raw}


def label(score: float, band: float = 0.0) -> str:
    return "positive" if score > band else "negative" if score < -band else "neutral"


@lru_cache(maxsize=1)
def get_sentiment() -> SentimentEnsemble:
    return SentimentEnsemble()
