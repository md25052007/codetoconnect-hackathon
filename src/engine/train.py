"""Train the engine's two ML models and write metrics to models/metrics.json.

1. Sentiment: TF-IDF + logistic regression on Financial PhraseBank (80/20 stratified split)
2. Events:    TF-IDF + logistic regression trained by weak supervision from the rule tagger
"""
from __future__ import annotations

import json
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import MODELS, RAW, TRAIN
from src.engine.nlp.events import _COMPILED, EVENT_MODEL_PATH, RuleTagger
from src.engine.nlp.sentiment import SENTIMENT_MODEL_PATH, FinLexicon, label
from src.engine.nlp.text import clean

SEED = 42


def _tfidf_lr(c: float = 4.0, max_word: int = 40000, max_char: int = 60000) -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_word, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=max_char, sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", feats), ("lr", LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))])


def train_sentiment() -> dict:
    """Train on two labelled corpora so the model handles formal news AND social posts:
    Financial PhraseBank (news sentences) + Twitter Financial News Sentiment (tweets)."""
    pb = pd.read_csv(TRAIN / "financial_phrasebank.csv").assign(domain="news")
    tw = pd.read_csv(TRAIN / "twitter_financial_news_sentiment.csv").assign(domain="social")
    df = pd.concat([pb[["text", "label", "domain"]], tw[["text", "label", "domain"]]], ignore_index=True)
    df["clean"] = df.text.map(clean)
    df["strat"] = df.domain + "_" + df.label
    tr, te = train_test_split(df, test_size=0.2, stratify=df.strat, random_state=SEED)

    pipe = _tfidf_lr()
    # out-of-fold probabilities on the training split -> tune ensemble weights without touching test
    oof = cross_val_predict(pipe, tr.clean, tr.label, cv=5, method="predict_proba", n_jobs=-1)
    cls = sorted(tr.label.unique())
    lex_model = FinLexicon()
    tr_lex = lex_model.score(tr.text.tolist())
    oof_score = oof[:, cls.index("positive")] - oof[:, cls.index("negative")]

    weights = {}
    for dom in ("news", "social"):
        msk = (tr.domain == dom).values
        best = (-1, 0.0, 0.15)
        for w in np.arange(0.0, 0.55, 0.05):
            for band in np.arange(0.05, 0.40, 0.025):
                s = (1 - w) * oof_score[msk] + w * tr_lex[msk]
                f1 = f1_score(tr.label.values[msk], [label(x, band) for x in s], average="macro")
                if f1 > best[0]:
                    best = (f1, round(float(w), 2), round(float(band), 3))
        weights[dom] = {"lexicon_weight": best[1], "neutral_band": best[2], "cv_macro_f1": round(best[0], 4)}

    pipe.fit(tr.clean, tr.label)
    p = pipe.predict_proba(te.clean)
    mscore = p[:, cls.index("positive")] - p[:, cls.index("negative")]
    te_lex = lex_model.score(te.text.tolist())
    vader = SentimentIntensityAnalyzer()
    te_vader = np.array([vader.polarity_scores(t)["compound"] for t in te.clean])

    def m(y, yhat):
        return {"accuracy": round(accuracy_score(y, yhat), 4), "macro_f1": round(f1_score(y, yhat, average="macro"), 4)}

    results = {}
    for dom in ("news", "social"):
        msk = (te.domain == dom).values
        y = te.label.values[msk]
        w, band = weights[dom]["lexicon_weight"], weights[dom]["neutral_band"]
        ens = (1 - w) * mscore[msk] + w * te_lex[msk]
        results[dom] = {
            "n_test": int(msk.sum()),
            "baseline_vader": m(y, [label(x, 0.05) for x in te_vader[msk]]),
            "baseline_vader_finance_lexicon": m(y, [label(x, 0.05) for x in te_lex[msk]]),
            "tfidf_lr_model": m(y, [cls[i] for i in p[msk].argmax(1)]),
            "ensemble": m(y, [label(x, band) for x in ens]),
        }
        if dom == "news":
            allagree = te[msk].merge(pb[["text", "agreement"]], on="text", how="left").agreement.values == 1.0
            results[dom]["ensemble_allagree_accuracy"] = round(
                accuracy_score(y[allagree], np.array([label(x, band) for x in ens])[allagree]), 4)

    metrics = {
        "datasets": "Financial PhraseBank v1.0 (4,838) + Twitter Financial News Sentiment (9,543); 20% stratified hold-out",
        "n_train": len(tr), "n_test": len(te), "ensemble_weights": weights, "test": results,
    }
    pipe.fit(df.clean, df.label)  # refit on all labelled data for production
    joblib.dump(pipe, SENTIMENT_MODEL_PATH, compress=3)
    (MODELS / "sentiment_ensemble.json").write_text(json.dumps(weights, indent=2))
    return metrics


def _mask_rule_spans(text: str, rng: random.Random) -> str:
    """Remove the words that triggered the rules so the model must learn from context."""
    for pats in _COMPILED.values():
        for rx, _ in pats:
            if rng.random() < 0.7:
                text = rx.sub(" ", text)
    return " ".join(text.split())


def train_events() -> dict:
    rng = random.Random(SEED)
    news = pd.read_csv(RAW / "news_reddit_worldnews.csv.gz").text
    tweets = pd.read_csv(RAW / "tweets_stocknet.csv.gz").text
    pb = pd.read_csv(TRAIN / "financial_phrasebank.csv").text
    corpus = pd.concat([news, tweets, pb]).map(lambda t: clean(t, keep_cashtags=False)).drop_duplicates()
    corpus = corpus[corpus.str.len() > 15].tolist()

    tagger = RuleTagger()
    rows = []
    for t in corpus:
        lab, _, score = tagger.predict_one(t)
        if lab != "Other" and score >= 1.5:
            rows.append((t, lab))
        elif lab == "Other" and score == 0:
            rows.append((t, "Other"))
    wdf = pd.DataFrame(rows, columns=["text", "label"])
    # cap the dominant "Other" class and very large classes to keep training balanced and fast
    wdf = pd.concat([g.sample(min(len(g), 6000), random_state=SEED) for _, g in wdf.groupby("label")])
    # augmentation: a masked copy of each weakly-labelled positive example
    aug = wdf[wdf.label != "Other"].copy()
    aug["text"] = aug.text.map(lambda t: _mask_rule_spans(t, rng))
    aug = aug[aug.text.str.split().str.len() >= 4]
    train_df = pd.concat([wdf, aug]).sample(frac=1, random_state=SEED)

    tr, te = train_test_split(train_df, test_size=0.15, stratify=train_df.label, random_state=SEED)
    pipe = _tfidf_lr(c=3.0, max_word=40000, max_char=40000)
    pipe.fit(tr.text, tr.label)
    pred = pipe.predict(te.text)
    metrics = {
        "weak_labels": wdf.label.value_counts().to_dict(),
        "n_train_with_augmentation": len(train_df),
        "agreement_with_weak_labels_heldout": {
            "accuracy": round(accuracy_score(te.label, pred), 4),
            "macro_f1": round(f1_score(te.label, pred, average="macro"), 4),
        },
    }
    pipe.fit(train_df.text, train_df.label)
    joblib.dump(pipe, EVENT_MODEL_PATH, compress=3)
    return metrics


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    print("Training sentiment model (Financial PhraseBank + financial tweets) ...")
    s = train_sentiment()
    for dom, res in s["test"].items():
        print(f"  [{dom}] n_test={res['n_test']}")
        for k in ("baseline_vader", "baseline_vader_finance_lexicon", "tfidf_lr_model", "ensemble"):
            print(f"    {k:32s} {res[k]}")
    print(f"  ensemble weights: {s['ensemble_weights']}")
    print("Training event classifier by weak supervision ...")
    e = train_events()
    print(f"  weak labels: {e['weak_labels']}")
    print(f"  held-out agreement: {e['agreement_with_weak_labels_heldout']}")
    out = MODELS / "metrics.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update({"sentiment": s, "events_training": e})
    out.write_text(json.dumps(existing, indent=2))
    print(f"Saved models + {out}")


if __name__ == "__main__":
    main()
