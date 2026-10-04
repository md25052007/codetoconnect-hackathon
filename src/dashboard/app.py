"""Streamlit dashboard: risk engine, Module A (index rebalancer), Module B (stress test).

Run:  streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (MARKET, MODELS, ROOT, STRESS_CORROBORATION, STRESS_SINGLE_REPORT_OVERRIDE,  # noqa: E402
                        STRESS_TRIGGERS, TICKERS, UNIVERSE)
from src.engine.engine import RiskEngine  # noqa: E402
from src.engine.store import read_daily, read_signals  # noqa: E402
from src.modules.rebalancer import backtest, sector_weights  # noqa: E402
from src.modules.stress.engine import is_trigger, load_portfolio, run_stress, triggered_events  # noqa: E402
from src.modules.stress.scenarios import SCENARIOS, Scenario, scenario_for  # noqa: E402

st.set_page_config(page_title="AI/NLP Risk Engine", page_icon="📡", layout="wide")

# Validated categorical palette (fixed order, never cycled) + diverging pair + neutrals
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
POS, NEG, NEUTRAL = "#2a78d6", "#e34948", "#9a9993"
SECTOR_COLORS = dict(zip(["Technology", "Consumer", "Financials", "Energy", "Healthcare", "Industrials"], SERIES))
EVENT_COLORS = {"Geopolitical": SERIES[0], "Macroeconomic": SERIES[1], "Credit Event": SERIES[2]}
FONT = dict(family="Inter, system-ui, sans-serif", size=13)


def style(fig: go.Figure, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=48, b=10), font=FONT, hovermode="x unified",
                      title_font=dict(size=15), title_x=0, title_y=0.98, title_yanchor="top",
                      legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, title=None) if legend else None,
                      showlegend=legend)
    fig.update_xaxes(showgrid=False, linecolor="rgba(128,128,128,.4)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,.15)", zeroline=False)
    return fig


def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    return f"{sign}${x / 1e9:.2f}bn" if x >= 1e9 else f"{sign}${x / 1e6:.1f}m"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading NLP models ...")
def get_engine() -> RiskEngine:
    return RiskEngine()


@st.cache_data(show_spinner="Loading signals ...")
def get_signals() -> pd.DataFrame:
    s = read_signals()
    s["date"] = s.timestamp.dt.normalize()
    return s


@st.cache_data
def get_daily() -> pd.DataFrame:
    return read_daily()


@st.cache_data(show_spinner="Running index back-test ...")
def get_backtest(kappa: float, half_life: float, max_w: float, max_turnover: float) -> dict:
    r = backtest(kappa=kappa, half_life_days=half_life, max_weight=max_w, max_turnover=max_turnover)
    return {k: v for k, v in r.items()}


@st.cache_data
def get_triggers() -> pd.DataFrame:
    return triggered_events(get_signals())


@st.cache_data
def get_metrics() -> dict:
    p = MODELS / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def get_portfolio() -> pd.DataFrame:
    return load_portfolio()


engine = get_engine()
sig = get_signals()
daily = get_daily()
metrics = get_metrics()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("AI/NLP Financial Risk Engine")
st.caption("Turns news headlines and tweets into structured risk signals (sentiment, event type, impact), then uses them "
           "to rebalance a 20-stock index and to stress-test a wholesale banking book. "
           f"Back-ends: sentiment = `{engine.backends['sentiment']}`, events = `{engine.backends['events']}`.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Documents processed", f"{sig.doc_id.nunique():,}")
k2.metric("Structured signals", f"{len(sig):,}")
k3.metric("Sources", "News + Twitter")
k4.metric("Stress triggers found", f"{len(get_triggers()):,}")
sm = metrics.get("sentiment", {}).get("test", {}).get("news", {}).get("tfidf_lr_model", {})
k5.metric("Sentiment accuracy (PhraseBank)", f"{100 * sm.get('accuracy', 0):.0f}%" if sm else "n/a",
          help="Held-out test set, TF-IDF + logistic regression vs 54% for plain VADER")

tab_engine, tab_a, tab_b, tab_q, tab_about = st.tabs(
    ["Risk engine", "Module A: index rebalancer", "Module B: stress test", "Model quality", "Data & architecture"])

# ===========================================================================
# TAB 1 - ENGINE
# ===========================================================================
with tab_engine:
    st.subheader("Analyse any headline or post")
    c1, c2 = st.columns([3, 1])
    examples = ["Russia invades Crimea as Ukraine crisis escalates, markets plunge",
                "Greece defaults on IMF loan as bailout talks collapse",
                "Fed unexpectedly raises interest rates by 50bps amid inflation fears",
                "$AAPL crushing it, new iPhone launch looks huge. bullish!",
                "JPMorgan fined $13 billion by Justice Department over mortgage securities",
                "Exxon quarterly profit falls 50% as oil prices slump"]
    pick = c2.selectbox("Example", examples, index=0)
    source = c2.radio("Source type", ["news", "social"], horizontal=True)
    text = c1.text_area("Text", value=pick, height=100)
    if text.strip():
        out = engine.analyze_text(text, source)
        rows = [{"entity": s.entity, "entity type": s.entity_type, "sentiment": s.sentiment, "label": s.sentiment_label,
                 "event": s.event_type, "event conf.": s.event_confidence, "impact (1-10)": s.impact,
                 "stress trigger?": "yes" if is_trigger(s.event_type, s.impact, s.entity_type, s.sentiment) else "no"}
                for s in out]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        top = max(out, key=lambda s: s.impact)
        with st.expander("Why this score? (explainability)", expanded=False):
            e = top.explain
            st.markdown(
                f"**Sentiment** = ensemble of supervised model `{e['sent_model']:+.2f}` and finance lexicon "
                f"`{e['sent_lexicon']:+.2f}` -> raw `{e['sent_raw']:+.2f}` -> calibrated `{top.sentiment:+.2f}` "
                f"(scores inside the neutral band map to 0).  \n"
                f"**Event** = `{top.event_type}` via `{e['event_method']}`.  \n"
                f"**Impact** = 1 + 9 x [severity {e['severity']} x intensity {e['intensity']} x credibility "
                f"{e['credibility']} x reach {e['reach']} + urgency {e['urgency']} + magnitude {e['magnitude']} "
                f"+ popularity {e['popularity']}] = **{top.impact}**")
            st.json(top.to_dict())

    st.divider()
    st.subheader("Real-time replay of the historical stream")
    st.caption("Replays the merged news + Twitter stream in time order through the engine, as it would run live.")
    r1, r2, r3 = st.columns([1, 1, 2])
    day = r1.date_input("Day", value=pd.Timestamp("2014-03-03"), min_value=sig.date.min(), max_value=sig.date.max())
    speed = r2.select_slider("Speed (signals/sec)", options=[5, 10, 20, 50, 100], value=20)
    if r3.button("Start replay", type="primary"):
        feed = sig[sig.date == pd.Timestamp(day)].sort_values("timestamp")
        feed = feed[feed.entity != "UNLINKED"]
        holder, stats = st.empty(), st.empty()
        shown = []
        for _, s in feed.head(300).iterrows():
            shown.insert(0, {"time": s.timestamp.strftime("%H:%M:%S"), "source": s.channel, "entity": s.entity,
                             "sentiment": round(s.sentiment, 2), "event": s.event_type, "impact": s.impact,
                             "text": s.text[:120]})
            holder.dataframe(pd.DataFrame(shown[:15]), hide_index=True, width="stretch")
            hi = sum(1 for x in shown if x["impact"] >= 7)
            stats.caption(f"{len(shown)} signals | {hi} with impact >= 7")
            time.sleep(1.0 / speed)

    st.divider()
    st.subheader("Signal explorer")
    f1, f2, f3, f4 = st.columns(4)
    ent = f1.selectbox("Entity", ["AAPL"] + [t for t in TICKERS if t != "AAPL"] + [MARKET])
    evs = f2.multiselect("Event types", sorted(sig.event_type.unique()), default=[])
    min_imp = f3.slider("Min impact", 1.0, 10.0, 1.0, 0.5)
    rng = f4.date_input("Date range", value=(pd.Timestamp("2015-01-01"), pd.Timestamp("2015-12-31")),
                        min_value=sig.date.min(), max_value=sig.date.max())
    d0, d1 = (pd.Timestamp(rng[0]), pd.Timestamp(rng[1])) if isinstance(rng, (tuple, list)) and len(rng) == 2 else (sig.date.min(), sig.date.max())

    dd = daily[(daily.entity == ent) & (daily.date >= d0) & (daily.date <= d1)]
    fig = go.Figure(go.Bar(x=dd.date, y=dd.score, marker_color=np.where(dd.score >= 0, POS, NEG),
                           marker_line_width=0, name="daily sentiment score",
                           hovertemplate="%{x|%d %b %Y}<br>score %{y:+.3f}<extra></extra>"))
    fig.update_layout(title=f"{ent}: daily sentiment score (impact-weighted, shrunk by volume)")
    st.plotly_chart(style(fig, 300, legend=False), width="stretch")

    view = sig[(sig.entity == ent) & (sig.impact >= min_imp) & (sig.date >= d0) & (sig.date <= d1)]
    if evs:
        view = view[view.event_type.isin(evs)]
    cA, cB = st.columns([2, 1])
    cA.dataframe(view.sort_values("impact", ascending=False)[["timestamp", "channel", "sentiment", "event_type", "impact", "text"]]
                 .head(300), hide_index=True, width="stretch", height=320)
    cnt = view[view.event_type != "Other"].event_type.value_counts().sort_values()
    fig = go.Figure(go.Bar(x=cnt.values, y=cnt.index, orientation="h", marker_color=SERIES[0], marker_line_width=0,
                           hovertemplate="%{y}: %{x:,}<extra></extra>"))
    fig.update_layout(title="Event mix (excl. Other)")
    cB.plotly_chart(style(fig, 320, legend=False), width="stretch")

# ===========================================================================
# TAB 2 - MODULE A
# ===========================================================================
with tab_a:
    st.subheader("Tactical index rebalancing driven by sentiment")
    st.caption("20 S&P 100 stocks start equal-weighted (5% each). Each day the engine's sentiment scores are smoothed "
               "(EWMA), compared across stocks, and turned into tilts: positive news -> higher weight, negative -> lower, "
               "with 1-12% caps and a 20% turnover limit. Weights then drift with prices. No look-ahead.")
    p1, p2, p3, p4 = st.columns(4)
    kappa = p1.slider("Tilt strength (kappa)", 0.0, 1.0, 0.35, 0.05)
    hl = p2.slider("Signal half-life (days)", 1, 15, 5)
    maxw = p3.slider("Max weight", 0.06, 0.20, 0.12, 0.01)
    mto = p4.slider("Max daily turnover", 0.05, 0.5, 0.20, 0.05)
    bt = get_backtest(kappa, hl, maxw, mto)
    W, perf, m = bt["weights"], bt["perf"], bt["metrics"]

    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Sentiment index return", f"{m['sentiment_index']['total_return_pct']:.1f}%",
              f"{m['sentiment_index']['total_return_pct'] - m['equal_weight']['total_return_pct']:+.1f} pts vs equal weight")
    a2.metric("Equal-weight return", f"{m['equal_weight']['total_return_pct']:.1f}%")
    a3.metric("Sharpe (index / EW)", f"{m['sentiment_index']['sharpe']:.2f} / {m['equal_weight']['sharpe']:.2f}")
    a4.metric("Tracking error", f"{m['tracking_error_pct']:.2f}%")
    a5.metric("Avg daily turnover", f"{m['avg_daily_turnover_pct']:.1f}%")

    sw = sector_weights(W)
    fig = go.Figure()
    for sec in [s for s in SECTOR_COLORS if s in sw.columns]:
        fig.add_trace(go.Scatter(x=sw.index, y=100 * sw[sec], name=sec, stackgroup="one", mode="lines",
                                 line=dict(width=0.5, color=SECTOR_COLORS[sec]), fillcolor=SECTOR_COLORS[sec],
                                 hovertemplate=f"{sec}: %{{y:.1f}}%<extra></extra>"))
    fig.update_layout(title="Index weights by sector over time (%)", yaxis_range=[0, 100])
    st.plotly_chart(style(fig, 360), width="stretch")

    wk = (W - 1 / len(TICKERS)).resample("W-FRI").last() * 100
    order = sorted(TICKERS, key=lambda t: (UNIVERSE[t]["sector"], t))
    lim = max(1.0, float(np.nanpercentile(np.abs(wk.values), 98)))
    fig = go.Figure(go.Heatmap(z=wk[order].T.values, x=wk.index, y=[f"{t} ({UNIVERSE[t]['sector'][:4]})" for t in order],
                               zmin=-lim, zmax=lim, zmid=0,
                               colorscale=[[0, NEG], [0.5, "#f0efec"], [1, POS]], xgap=1, ygap=1,
                               colorbar=dict(title="pts vs 5%"),
                               hovertemplate="%{y}<br>week of %{x|%d %b %Y}<br>active weight %{z:+.2f} pts<extra></extra>"))
    fig.update_layout(title="Active weight vs equal weight, weekly (blue = overweight after positive news, red = underweight)")
    st.plotly_chart(style(fig, 520, legend=False), width="stretch")

    c1, c2 = st.columns(2)
    picks = c1.multiselect("Compare stocks (max 4)", TICKERS, default=["AAPL", "XOM", "JPM", "FB"], max_selections=4)
    fig = go.Figure()
    for i, t in enumerate(picks):
        fig.add_trace(go.Scatter(x=W.index, y=100 * W[t], name=t, mode="lines", line=dict(width=2, color=SERIES[i]),
                                 hovertemplate=f"{t}: %{{y:.2f}}%<extra></extra>"))
    fig.add_hline(y=5, line_dash="dot", line_color=NEUTRAL, annotation_text="benchmark 5%")
    fig.update_layout(title="Weight over time (%)")
    c1.plotly_chart(style(fig, 360), width="stretch")

    eq = (1 + perf).cumprod() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.sentiment_index, name="Sentiment index", line=dict(width=2, color=SERIES[0])))
    fig.add_trace(go.Scatter(x=eq.index, y=eq.equal_weight, name="Equal weight", line=dict(width=2, color=SERIES[1])))
    fig.update_layout(title="Growth of 100 (net of 5 bp costs)")
    c2.plotly_chart(style(fig, 360), width="stretch")

    st.markdown("**Inspect a rebalancing day**")
    day_a = st.select_slider("Date", options=list(W.index.date), value=W.index.date[len(W) // 2])
    ts = pd.Timestamp(day_a)
    i = W.index.get_loc(ts)
    prev = W.iloc[max(i - 5, 0)]
    cur = W.loc[ts]
    chg = ((cur - prev) * 100).sort_values()
    c3, c4 = st.columns([1, 1])
    fig = go.Figure(go.Bar(x=chg.values, y=chg.index, orientation="h", marker_line_width=0,
                           marker_color=np.where(chg.values >= 0, POS, NEG),
                           hovertemplate="%{y}: %{x:+.2f} pts<extra></extra>"))
    fig.update_layout(title=f"Weight change over the 5 trading days to {day_a} (pts)")
    c3.plotly_chart(style(fig, 520, legend=False), width="stretch")
    movers = list(chg.index[:2]) + list(chg.index[-2:])
    news = sig[(sig.entity.isin(movers)) & (sig.date <= ts) & (sig.date > ts - pd.Timedelta(days=7)) & (sig.sentiment != 0)]
    news = news.reindex(news.sentiment.abs().sort_values(ascending=False).index).groupby("entity").head(3)
    c4.markdown("Headlines driving the biggest moves")
    c4.dataframe(news[["entity", "timestamp", "sentiment", "impact", "text"]], hide_index=True, width="stretch", height=480)

# ===========================================================================
# TAB 3 - MODULE B
# ===========================================================================
with tab_b:
    st.subheader("Event-driven stress testing of a wholesale banking book")
    pf = get_portfolio()
    corr = ", ".join(f"{k} needs {v}+ reports/day" for k, v in STRESS_CORROBORATION.items())
    st.caption(f"Synthetic book: {len(pf)} trades, {money(pf.market_value.sum())} (loans, corporate & sovereign bonds, "
               f"listed equity, IR swaps, CDS, FX forwards). A stress test fires when a market-level signal is negative and "
               f"its impact exceeds the trigger ({', '.join(f'{k} > {v}' for k, v in STRESS_TRIGGERS.items())}); "
               f"{corr} unless one report scores >= {STRESS_SINGLE_REPORT_OVERRIDE}.")

    trig = get_triggers()
    fig = go.Figure()
    for ev, col in EVENT_COLORS.items():
        t = trig[trig.event_type == ev]
        fig.add_trace(go.Scatter(x=t.timestamp, y=t.impact, mode="markers", name=ev,
                                 marker=dict(size=9, color=col, line=dict(width=2, color="rgba(255,255,255,.9)")),
                                 text=t.text.str.slice(0, 110), customdata=t.reports,
                                 hovertemplate="%{x|%d %b %Y}<br>impact %{y}<br>%{customdata} reports<br>%{text}<extra></extra>"))
    fig.update_layout(title="Detected stress-trigger events (hover for the headline)", hovermode="closest", yaxis_title="impact")
    st.plotly_chart(style(fig, 330), width="stretch")

    mode = st.radio("Choose the trigger", ["Detected event", "Type a headline", "Custom shocks"], horizontal=True)
    scenario = None
    if mode == "Detected event":
        opts = trig.sort_values("impact", ascending=False)
        label_of = {i: f"{r.timestamp:%Y-%m-%d} | {r.event_type} | impact {r.impact} | {r.text[:90]}" for i, r in opts.iterrows()}
        keys = list(label_of)
        credit = [k for k in keys if opts.loc[k, "event_type"] == "Credit Event"]
        sel = st.selectbox("Event", keys, index=keys.index(credit[0]) if credit else 0, format_func=label_of.get)
        row = trig.loc[sel]
        ev_type, impact, trig_text = row.event_type, float(row.impact), row.text
    elif mode == "Type a headline":
        trig_text = st.text_input("Headline", "Russia invades Ukraine, NATO on high alert as global markets plunge")
        top = max(engine.analyze_text(trig_text, "news"), key=lambda s: s.impact)
        ev_type, impact = top.event_type, top.impact
        fired = is_trigger(ev_type, impact, "market", top.sentiment)
        st.info(f"Engine output: **{ev_type}**, impact **{impact}**, sentiment **{top.sentiment:+.2f}** -> "
                + ("trigger fires." if fired else "below trigger; running anyway for illustration."))
        if ev_type not in SCENARIOS:
            ev_type = "Macroeconomic"
    else:
        base = st.selectbox("Start from", list(SCENARIOS), index=list(SCENARIOS).index("Assignment example"))
        b = SCENARIOS[base]
        s1, s2, s3, s4 = st.columns(4)
        eqs = s1.slider("Equity move %", -40.0, 10.0, float(b.equity_pct), 1.0)
        rts = s2.slider("Rates shift (bp)", -200.0, 400.0, float(b.rates_bps), 10.0)
        hy = s3.slider("HY spread widening (bp)", 0.0, 1000.0, float(b.spread_bps["BB"]), 10.0)
        pdm = s4.slider("PD multiplier", 1.0, 5.0, float(b.pd_multiplier), 0.1)
        ratio = (hy / b.spread_bps["BB"]) if b.spread_bps["BB"] else 0
        spreads = {k: (v * ratio if b.spread_bps["BB"] else hy * {"AAA": .1, "AA": .15, "A": .25, "BBB": .45, "BB": 1, "B": 1.4}[k])
                   for k, v in b.spread_bps.items()}
        scenario = Scenario(name="Custom", description="User-defined shocks", equity_pct=eqs, rates_bps=rts,
                            spread_bps=spreads, em_sovereign_bps=hy * 0.6, fx_pct=b.fx_pct, pd_multiplier=pdm,
                            sector_equity_mult=b.sector_equity_mult, sector_credit_mult=b.sector_credit_mult)
        ev_type, impact, trig_text = base, 10.0, "custom scenario"

    res = run_stress(ev_type, impact, trig_text, scenario=scenario, portfolio=pf)
    sc = res.scenario
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Value before", money(res.value_before))
    b2.metric("Value after", money(res.value_after), f"{res.pnl_pct:.2f}%")
    b3.metric("Stress loss", money(res.pnl))
    b4.metric("Capital at risk", f"{res.capital_at_risk_pct:.0f}%", help="Loss as % of an assumed 10% capital buffer")
    st.caption(f"Scenario **{sc['name']}** at severity {res.severity:.0%}: equities {sc['equity_pct']:+.1f}%, "
               f"rates {sc['rates_bps']:+.0f} bp, BBB spreads +{sc['spread_bps'].get('BBB', 0):.0f} bp, "
               f"HY +{sc['spread_bps'].get('BB', 0):.0f} bp, PD x{sc['pd_multiplier']:.2f}. {sc['description']}")

    c1, c2 = st.columns(2)
    ac = pd.DataFrame(res.by_asset_class)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ac.asset_class, y=ac.value_before / 1e6, name="Before", marker_color=SERIES[0], marker_line_width=0,
                         hovertemplate="%{x} before: $%{y:,.0f}m<extra></extra>"))
    fig.add_trace(go.Bar(x=ac.asset_class, y=ac.value_after / 1e6, name="After stress", marker_color=SERIES[1], marker_line_width=0,
                         hovertemplate="%{x} after: $%{y:,.0f}m<extra></extra>"))
    fig.update_layout(title="Portfolio value before vs after, by asset class ($m)", barmode="group", bargap=0.3, bargroupgap=0.05)
    c1.plotly_chart(style(fig, 360), width="stretch")

    rf = pd.Series(res.by_risk_factor).drop("pnl", errors="ignore")
    rf = rf[rf.index != "pnl"]
    fig = go.Figure(go.Waterfall(x=[k.replace("_", " ") for k in rf.index] + ["total"],
                                 y=list(rf.values / 1e6) + [0], measure=["relative"] * len(rf) + ["total"],
                                 decreasing=dict(marker=dict(color=NEG)), increasing=dict(marker=dict(color=POS)),
                                 totals=dict(marker=dict(color=NEUTRAL)),
                                 hovertemplate="%{x}: $%{y:,.1f}m<extra></extra>"))
    fig.update_layout(title="Where the loss comes from: P&L by risk factor ($m)")
    c2.plotly_chart(style(fig, 360, legend=False), width="stretch")

    c3, c4 = st.columns(2)
    bs = pd.DataFrame(res.by_sector).sort_values("pnl")
    fig = go.Figure(go.Bar(x=bs.pnl / 1e6, y=bs.sector, orientation="h", marker_line_width=0,
                           marker_color=np.where(bs.pnl >= 0, POS, NEG),
                           hovertemplate="%{y}: $%{x:,.1f}m<extra></extra>"))
    fig.update_layout(title="P&L by sector ($m)")
    c3.plotly_chart(style(fig, 380, legend=False), width="stretch")
    c4.markdown("**Ten worst-hit positions**")
    wp = pd.DataFrame(res.worst_positions)
    wp["market_value"] = wp.market_value.map(money)
    wp["pnl"] = wp.pnl.map(money)
    c4.dataframe(wp, hide_index=True, width="stretch", height=360)

    with st.expander("All positions after stress"):
        st.dataframe(res.positions[["trade_id", "asset_class", "instrument", "counterparty", "sector", "region", "rating",
                                    "market_value", "pnl_rates", "pnl_credit_spread", "pnl_default_loss", "pnl_equity",
                                    "pnl_fx", "pnl", "stressed_value"]], hide_index=True, width="stretch")

# ===========================================================================
# TAB 4 - MODEL QUALITY
# ===========================================================================
with tab_q:
    st.subheader("How good are the signals?")
    s = metrics.get("sentiment", {}).get("test", {})
    if s:
        rows = []
        for dom, res_ in s.items():
            for k in ("baseline_vader", "baseline_vader_finance_lexicon", "tfidf_lr_model", "ensemble"):
                rows.append({"test set": "Financial PhraseBank (news)" if dom == "news" else "Financial tweets",
                             "method": k.replace("_", " "), "accuracy": res_[k]["accuracy"], "macro F1": res_[k]["macro_f1"]})
        df = pd.DataFrame(rows)
        c1, c2 = st.columns([3, 2])
        fig = go.Figure()
        for i, mth in enumerate(df.method.unique()):
            d = df[df.method == mth]
            fig.add_trace(go.Bar(x=d["test set"], y=100 * d.accuracy, name=mth, marker_color=SERIES[i], marker_line_width=0,
                                 hovertemplate=f"{mth}: %{{y:.1f}}%<extra></extra>"))
        fig.update_layout(title="Sentiment accuracy on held-out labelled data (%)", barmode="group", yaxis_range=[0, 100])
        c1.plotly_chart(style(fig, 360), width="stretch")
        c2.dataframe(df, hide_index=True, width="stretch")

    sg = metrics.get("sentiment_gold_eval")
    ev = metrics.get("events_eval")
    c1, c2 = st.columns(2)
    if ev:
        c1.markdown(f"**Event classification** on {ev['n']} hand-labelled headlines and tweets")
        c1.dataframe(pd.DataFrame([{"method": k.replace("_", " "), **ev[k]} for k in
                                   ("rules_only", "weak_supervised_model_only", "hybrid")]), hide_index=True, width="stretch")
    if sg:
        c2.markdown(f"**Sentiment on market-relevant world news** ({sg['n']} hand-labelled texts)")
        c2.dataframe(pd.DataFrame([{"method": k.replace("_", " "), **v} for k, v in sg.items() if k != "n"]),
                     hide_index=True, width="stretch")
        c2.caption("The supervised model is weak on geopolitical/macro text it was never trained on; routing those "
                   "texts to the lexicon is why the engine up-weights it there.")

    iv = metrics.get("impact_validation", {})
    if iv:
        b = pd.DataFrame(iv["companies"]["by_impact_bucket"])
        fig = go.Figure(go.Bar(x=b.bucket, y=b.rel_move_vs_typical, marker_color=SERIES[0], marker_line_width=0,
                               text=[f"{v:.2f}x" for v in b.rel_move_vs_typical], textposition="outside",
                               customdata=b.days, hovertemplate="impact %{x}<br>%{y:.2f}x typical move<br>%{customdata} stock-days<extra></extra>"))
        fig.add_hline(y=1, line_dash="dot", line_color=NEUTRAL)
        fig.update_layout(title="Impact score vs realised price move: same-day |return| relative to each stock's typical day",
                          yaxis_title="x typical absolute move")
        st.plotly_chart(style(fig, 340, legend=False), width="stretch")
        se = iv.get("sentiment", {})
        st.markdown(f"Days with a **positive** sentiment score averaged **{se.get('mean_return_pct_when_score_pos', 0):+.2f}%** "
                    f"same-day return vs **{se.get('mean_return_pct_when_score_neg', 0):+.2f}%** on negative days "
                    f"(Spearman {se.get('spearman_score_vs_same_day_return', 0):.3f}; next-day "
                    f"{se.get('spearman_score_vs_next_day_return', 0):.3f}: the signal describes the news, it does not "
                    "forecast tomorrow's return).")

# ===========================================================================
# TAB 5 - ABOUT
# ===========================================================================
with tab_about:
    arch = ROOT / "docs" / "architecture.png"
    if arch.exists():
        st.image(str(arch), width="stretch")
    st.markdown((ROOT / "data" / "README.md").read_text())
