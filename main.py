"""Single entry point.

    python main.py dashboard     # open the Streamlit dashboard (uses committed signals)
    python main.py api           # REST API on http://localhost:8000/docs
    python main.py pipeline      # re-run the NLP engine over all historical documents
    python main.py train         # re-train the sentiment + event models
    python main.py evaluate      # model quality + impact validation -> models/metrics.json
    python main.py backtest      # Module A back-test
    python main.py stress --event Geopolitical --impact 8.5   # Module B one-off stress test
    python main.py live          # pull live Google News + Reddit posts and score them
    python main.py all           # train -> pipeline -> evaluate -> backtest
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def cmd_live(limit: int) -> None:
    import pandas as pd

    from src.config import PROCESSED
    from src.engine.engine import RiskEngine
    from src.engine.ingestion.sources import GoogleNewsRSS, RedditLive
    from src.engine.store import write_jsonl
    from src.modules.stress.engine import StressMonitor

    engine, monitor = RiskEngine(), StressMonitor()
    docs = []
    for src in (GoogleNewsRSS(per_query=limit), RedditLive(limit=limit)):
        try:
            got = list(src.documents())
            print(f"[live] {src.name}: {len(got)} documents")
            docs += got
        except Exception as exc:  # noqa: BLE001
            print(f"[live] {src.name} unavailable: {exc}")
    if not docs:
        print("[live] no documents fetched (offline?). The dashboard still works on the bundled data.")
        return
    sigs = [s.to_dict() for s in engine.analyze_batch(sorted(docs, key=lambda d: d.timestamp))]
    write_jsonl(sigs, PROCESSED / "live_signals.jsonl.gz")
    df = pd.DataFrame(sigs)
    pd.set_option("display.width", 200)
    print(df.sort_values("impact", ascending=False)[["timestamp", "channel", "entity", "sentiment", "event_type",
                                                      "impact", "text"]].head(25).to_string(max_colwidth=70))
    for s in sigs:
        r = monitor.on_signal(s)
        if r:
            print(f"\n[STRESS] {r.event_type} impact {r.impact}: '{r.trigger_text[:80]}' -> "
                  f"P&L {r.pnl / 1e6:,.1f}m ({r.pnl_pct:.2f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["dashboard", "api", "pipeline", "train", "evaluate", "backtest", "stress",
                                        "portfolio", "live", "all"])
    ap.add_argument("--event", default="Geopolitical")
    ap.add_argument("--impact", type=float, default=8.5)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--port", type=int, default=None)
    a = ap.parse_args()

    if a.command == "dashboard":
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py",
                        "--server.port", str(a.port or 8501)], check=False)
    elif a.command == "api":
        subprocess.run([sys.executable, "-m", "uvicorn", "src.api.main:app", "--port", str(a.port or 8000)], check=False)
    elif a.command == "train":
        from src.engine.train import main as train
        train()
    elif a.command == "pipeline":
        from src.engine.pipeline import run
        run()
    elif a.command == "evaluate":
        subprocess.run([sys.executable, "scripts/evaluate.py"], check=True)
    elif a.command == "backtest":
        from src.modules.rebalancer import main as bt
        bt()
    elif a.command == "portfolio":
        subprocess.run([sys.executable, "scripts/generate_portfolio.py"], check=True)
    elif a.command == "stress":
        from src.modules.stress.engine import run_stress
        r = run_stress(a.event, a.impact, "command line")
        print(f"{r.scenario['name']} (severity {r.severity:.0%})")
        print(f"value before {r.value_before / 1e9:.3f}bn -> after {r.value_after / 1e9:.3f}bn | "
              f"P&L {r.pnl / 1e6:,.1f}m ({r.pnl_pct:.2f}%) | capital at risk {r.capital_at_risk_pct:.1f}%")
        for row in r.by_asset_class:
            print(f"  {row['asset_class']:<11s} {row['pnl'] / 1e6:>9,.1f}m")
    elif a.command == "live":
        cmd_live(a.limit)
    elif a.command == "all":
        from src.engine.pipeline import run
        from src.engine.train import main as train
        from src.modules.rebalancer import main as bt
        train()
        run()
        subprocess.run([sys.executable, "scripts/evaluate.py"], check=True)
        bt()


if __name__ == "__main__":
    main()
