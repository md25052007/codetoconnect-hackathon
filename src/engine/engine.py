"""The unified AI/NLP Risk Engine: Document(s) in -> RiskSignal(s) out."""
from __future__ import annotations

from typing import Iterable, Iterator

from src.engine.nlp.entities import link_entities
from src.engine.nlp.events import get_event_classifier
from src.engine.nlp.impact import impact_score
from src.engine.nlp.sentiment import get_sentiment, label
from src.engine.schema import Document, RiskSignal


class RiskEngine:
    def __init__(self) -> None:
        self.sentiment = get_sentiment()
        self.events = get_event_classifier()

    @property
    def backends(self) -> dict[str, str]:
        return {"sentiment": self.sentiment.backend, "events": self.events.backend}

    def analyze_batch(self, docs: list[Document], keep_unlinked: bool = False) -> list[RiskSignal]:
        """Vectorised analysis of a batch of documents."""
        if not docs:
            return []
        texts = [d.text for d in docs]
        sent, parts = self.sentiment.score(texts, [d.source for d in docs])
        evs = self.events.predict(texts)
        out: list[RiskSignal] = []
        for i, d in enumerate(docs):
            ev, ev_conf, ev_method = evs[i]
            s = float(sent[i])
            ents = link_entities(d.text, d.tickers, ev)
            if not ents and keep_unlinked:
                from src.engine.nlp.entities import EntityMatch
                ents = [EntityMatch("UNLINKED", "none", 0.0, "none")]
            for e in ents:
                imp, why = impact_score(d.text, ev, s, d.source, e.entity_type, d.meta)
                out.append(RiskSignal(
                    doc_id=d.id, timestamp=d.timestamp.isoformat(), source=d.source, channel=d.channel,
                    entity=e.entity, entity_type=e.entity_type, relevance=e.relevance,
                    sentiment=round(s, 4), sentiment_label=label(s), event_type=ev,
                    event_confidence=ev_conf, impact=imp, text=d.text,
                    explain={"event_method": ev_method, "linked_via": e.via,
                             "sent_model": round(float(parts["model"][i]), 3),
                             "sent_lexicon": round(float(parts["lexicon"][i]), 3),
                             "sent_raw": round(float(parts["raw"][i]), 3), **why},
                ))
        return out

    def analyze(self, doc: Document) -> list[RiskSignal]:
        return self.analyze_batch([doc])

    def analyze_text(self, text: str, source: str = "news", tickers: list[str] | None = None) -> list[RiskSignal]:
        """Convenience entry point for ad-hoc text (API /analyze, dashboard)."""
        from datetime import datetime
        doc = Document(id="adhoc", source=source, channel="manual", timestamp=datetime.utcnow(),
                       text=text, tickers=tickers or [])
        return self.analyze_batch([doc], keep_unlinked=True)

    def process_stream(self, docs: Iterable[Document], batch_size: int = 512) -> Iterator[RiskSignal]:
        batch: list[Document] = []
        for d in docs:
            batch.append(d)
            if len(batch) >= batch_size:
                yield from self.analyze_batch(batch)
                batch = []
        yield from self.analyze_batch(batch)
