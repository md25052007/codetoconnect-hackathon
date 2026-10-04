"""Event classification.

Hybrid design (no hand-labelled training set is needed):

1. ``RuleTagger`` - weighted regular-expression patterns per event class, written from
   domain knowledge. High precision, but blind to phrasing it has never seen.
2. ``EventModel`` - TF-IDF + logistic regression trained by *weak supervision*: the
   rule tagger labels ~80k unlabelled headlines/tweets, the confident labels become the
   training set, and the model learns the surrounding vocabulary (e.g. "Kiev",
   "separatists", "Pyongyang" co-occur with geopolitical rules and generalise).
3. ``HybridEventClassifier`` - trusts strong rule hits, otherwise defers to the model.

Optional: a zero-shot transformer (facebook/bart-large-mnli) when RISK_ENGINE_ZEROSHOT=1.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

import joblib
import numpy as np

from src.config import EVENT_TYPES, MODELS
from src.engine.nlp.text import clean

EVENT_MODEL_PATH = MODELS / "event_tfidf_lr.joblib"

# (pattern, weight). Weights: 2 = decisive, 1 = typical, 0.5 = weak hint.
RULES: dict[str, list[tuple[str, float]]] = {
    "Geopolitical": [
        (r"(?<!price )(?<!car )(?<!trade )(?<!currency )(?<!bidding )(?<!console )\bwar\b(?! (?:on|with) (?:itself|uber|prices))|\bwarfare\b|civil war|\binvasion\b|\binvade[sd]?\b|annex(?:ation|es|ed)?\b", 2),
        (r"launch(?:es|ed)? (?:an? |the )?(?:military )?(?:operation|offensive|attack|air ?strikes?|raids?)|\brockets?\b|\bairstrikes?\b|\bjihad", 2),
        (r"\bmilitary\b|\btroops?\b|\bmissiles?\b|air ?strikes?|\bbomb(?:s|ing|ed)?\b|drone strikes?|\bshelling\b", 1.5),
        (r"\bterror(?:ist|ism|ists)?\b|\bisis\b|\bisil\b|islamic state|al[- ]qaeda|taliban|boko haram|hezbollah|hamas", 2),
        (r"\bsanctions?\b|\bembargo\b|\bnato\b|ceasefire|\bcoup\b|\brebels?\b|separatists?|insurgen(?:ts?|cy)", 1.5),
        (r"nuclear (?:weapons?|test|deal|program|talks)|north korea|pyongyang|\bcrimea\b|\bkiev\b|\bdonetsk\b", 1.5),
        (r"\bukraine\b|\bsyria[n]?\b|\biraq\b|\biran\b|\bisrael\b|\bgaza\b|\bputin\b|\bkremlin\b|south china sea", 0.75),
        (r"\bprotests?(?:ers)?\b|\briots?\b|\bhostages?\b|\brefugees?\b|\bgenocide\b|\bmassacre\b|\bdiplomat(?:s|ic)?\b|\bembassy\b|border clash", 0.75),
        (r"\bgeopolitic", 2),
    ],
    "Macroeconomic": [
        (r"\bgdp\b|\binflation\b|\bdeflation\b|\brecession\b|\bunemployment\b|jobs report|non-?farm|\bpayrolls?\b", 2),
        (r"interest rates?|rate (?:hike|cut|rise|increase)s?|central bank|federal reserve|\bthe fed\b|\becb\b|bank of (?:england|japan)|\bpboc\b", 2),
        (r"quantitative easing|\bqe\b|\bstimulus\b|monetary policy|fiscal (?:policy|cliff)|\bausterity\b|budget deficit|trade deficit|\btariffs?\b", 1.5),
        (r"\bdevaluation\b|\bdevalue[sd]?\b|\b(?:yuan|ruble|rouble|euro|yen|peso|rupee|dollar|currency)\b.{0,30}\b(?:fall|falls|plunge|crash|slump|record|low|high|weak)", 1.5),
        (r"oil prices?|crude (?:oil|prices?)|\bopec\b|commodity prices|bond yields?|treasury yields?|consumer prices|\bcpi\b|\bpmi\b|housing market", 1.5),
        (r"\beconom(?:y|ic|ies)\b|\bimf\b|world bank|stock markets?|\bmarkets? (?:fall|rally|crash|tumble)|global growth|growth forecast", 0.75),
    ],
    "Credit Event": [
        (r"\bdefault(?:s|ed|ing)?\b(?! (?:settings?|browser|apps?|image|viewer|search|option|password|mode))|(?<!morally )\bbankrupt(?:cy|cies)?\b|chapter 11|\binsolven(?:t|cy)\b|receivership", 2),
        (r"debt (?:crisis|restructuring|default|relief)|sovereign debt|missed (?:a |its )?payment|bond default|\bhaircut\b|bank run", 2),
        (r"(?:credit )?rating (?:cut|downgrade)|downgrade[sd]? (?:to|by) (?:moody|s&p|fitch)|\bmoody'?s\b|\bfitch\b|junk (?:status|bond|rating)", 1.5),
        (r"\bbail ?out\b|\bcreditors?\b|bad loans|non-performing|write-?downs?|liquidity crisis|credit crunch|\bdebt\b", 0.75),
    ],
    "Merger/Acquisition": [
        (r"\bacquir(?:e|es|ed|ing)\b|\bacquisitions?\b|\bmergers?\b|\bmerg(?:e|es|ed|ing)\b|\btakeover\b|\bbuyout\b", 2),
        (r"(?:agree[sd]?|deal|offer|plans?|move|bid) to (?:buy|purchase|acquire)|\bbid for\b|tender offer|\bstake in\b|takes? (?:a )?stake", 1.5),
        (r"\bdivestiture\b|spin[- ]?off|joint venture|\bm&a\b|\bbuys?\b.{0,40}\bfor \$?\d", 1),
    ],
    "Product Launch": [
        (r"\blaunch(?:es|ed|ing)?\b(?! (?:an? |the |its |new )?(?:operation|offensive|attack|air ?strikes?|strikes?|missiles?|rockets?|raids?|inquiry|investigation|probe|campaign|protest|crackdown|invasion|lawsuit))|\bunveil(?:s|ed|ing)?\b|\bdebut(?:s|ed)?\b|\brolls? out\b|\brollout\b", 2),
        (r"(?:new|next) (?:iphone|ipad|phone|smartphone|device|product|model|console|watch|tablet|service|app|feature|chip|processor|aircraft|drug)", 1.5),
        (r"apple watch|iphone \d|galaxy s\d|\bwwdc\b|\bces\b|\bkeynote\b|pre-?orders?|\bprototype\b|\bfda approv", 1.5),
        (r"introduc(?:e|es|ed|ing)\b|announc(?:e|es|ed) (?:a |the )?new|release[sd]? (?:a |its |the )?new|available (?:now|today)", 1),
    ],
    "Earnings": [
        (r"\bearnings\b|\beps\b|quarterly (?:results|profit|loss|revenue|sales)|\bq[1-4]\b|net income|operating (?:profit|income|loss)", 2),
        (r"\brevenues?\b|\bprofits?\b|net sales|\bguidance\b|(?:beat|miss|top)(?:s|ed)? (?:estimates|expectations|forecasts)|per share", 1.5),
        (r"\bsales (?:rose|fell|increased|decreased|grew|dropped|declined)|\bdividend\b|\bforecasts? (?:cut|raised)|profit warning", 1),
    ],
    "Regulatory/Legal": [
        (r"\blawsuits?\b|\bsue[sd]?\b|\bsuing\b|antitrust|class action|\bindict(?:ed|ment)\b|\bconvicted\b|\bverdict\b", 2),
        (r"\bfined?\b.{0,20}(?:\$|€|£|million|billion)|\bpenalt(?:y|ies)\b|\bsettle(?:s|d|ment)?\b|\bregulators?\b|\bsec\b|\bftc\b|justice department|\bdoj\b|european commission", 1.5),
        (r"\bcourt\b|\bjudge\b|\bruling\b|\binvestigation\b|\bprobe\b|patent|\bbann?(?:ed|s)?\b|\blegislation\b|\bfraud\b|\bcompliance\b|\bregulation\b", 1),
    ],
}

_COMPILED = {k: [(re.compile(p, re.IGNORECASE), w) for p, w in v] for k, v in RULES.items()}
CLASSES = [c for c in EVENT_TYPES if c != "Other"]


class RuleTagger:
    threshold = 1.25  # minimum evidence for a non-"Other" label (one weak cue is not enough)
    strong = 2.0      # evidence above which rules override the model

    def scores(self, text: str) -> dict[str, float]:
        return {c: sum(w for rx, w in _COMPILED[c] if rx.search(text)) for c in CLASSES}

    def predict_one(self, text: str) -> tuple[str, float, float]:
        sc = self.scores(clean(text, keep_cashtags=True))
        best = max(sc, key=sc.get)
        total = sc[best]
        if total < self.threshold:
            return "Other", 0.0, 0.0
        conf = total / (total + 1.0)
        return best, round(conf, 3), total

    def predict(self, texts: list[str]) -> list[tuple[str, float, float]]:
        return [self.predict_one(t) for t in texts]


class EventModel:
    def __init__(self, path=EVENT_MODEL_PATH) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run `python main.py train` first")
        self.pipe = joblib.load(path)
        self.classes = list(self.pipe.classes_)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipe.predict_proba([clean(t) for t in texts])


class ZeroShotEvents:  # pragma: no cover - optional heavy dependency
    def __init__(self) -> None:
        from transformers import pipeline
        self.pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def predict(self, texts: list[str]) -> list[tuple[str, float]]:
        out = self.pipe([clean(t) for t in texts], candidate_labels=EVENT_TYPES, batch_size=16)
        out = out if isinstance(out, list) else [out]
        return [(o["labels"][0], float(o["scores"][0])) for o in out]


class HybridEventClassifier:
    def __init__(self) -> None:
        self.rules = RuleTagger()
        self.model = None
        self.zeroshot = None
        self.backend = "rules"
        try:
            self.model = EventModel()
            self.backend = "rules+weak-supervised-lr"
        except FileNotFoundError:
            pass
        if os.getenv("RISK_ENGINE_ZEROSHOT") == "1":
            try:
                self.zeroshot = ZeroShotEvents()
                self.backend += "+zero-shot"
            except Exception as exc:  # noqa: BLE001
                print(f"[events] zero-shot unavailable ({exc})")

    def predict(self, texts: list[str], model_min_proba: float = 0.55) -> list[tuple[str, float, str]]:
        """Return (event_type, confidence, method) for each text."""
        rule_out = self.rules.predict(texts)
        proba = self.model.predict_proba(texts) if self.model is not None else None
        zs = self.zeroshot.predict(texts) if self.zeroshot is not None else None
        results = []
        for i, (r_label, r_conf, r_score) in enumerate(rule_out):
            if r_score >= self.rules.strong:
                results.append((r_label, r_conf, "rules"))
                continue
            if zs is not None and zs[i][1] >= 0.5:
                results.append((zs[i][0], round(zs[i][1], 3), "zero-shot"))
                continue
            if proba is not None:
                j = int(np.argmax(proba[i]))
                m_label, m_conf = self.model.classes[j], float(proba[i, j])
                if m_label != "Other" and m_conf >= model_min_proba:
                    results.append((m_label, round(m_conf, 3), "model"))
                    continue
            results.append((r_label, r_conf, "rules" if r_label != "Other" else "none"))
        return results


@lru_cache(maxsize=1)
def get_event_classifier() -> HybridEventClassifier:
    return HybridEventClassifier()
