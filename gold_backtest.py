"""
Gold Sentiment Backtester
--------------------------
Tests whether the bot's bullish/bearish/neutral calls on HISTORICAL headlines
would have matched gold's actual price direction afterward.

Pipeline:
  1. FETCH HEADLINES  - pull historical gold/macro headlines from Alpha Vantage
                        News Sentiment API (with real publish timestamps)
  2. FETCH PRICES     - pull historical gold futures (GC=F) prices via yfinance
  3. JUDGE            - run the same bullish/bearish/neutral prompt as the live
                        bot, batched, against every historical headline
  4. SCORE            - for each headline, measure gold's % price move over the
                        next N hours and check it against the verdict
  5. REPORT           - print hit rates / average returns per verdict bucket,
                        and save a CSV with every scored headline

Run it with: python gold_backtest.py

Needs:
  ANTHROPIC_API_KEY     - same key your live bot uses
  ALPHAVANTAGE_API_KEY  - free key: https://www.alphavantage.co/support/#api-key

Install deps:
  pip install anthropic requests yfinance pandas

IMPORTANT CAVEATS (read before trusting the numbers):
  - This checks DIRECTIONAL accuracy of the sentiment call, not real trading
    P&L. It ignores spread, slippage, and execution delay.
  - Alpha Vantage free tier = 25 requests/day. This script makes one request
    PER TOPIC (see NEWS_TOPICS below), so a single run uses a few of your 25.
  - Alpha Vantage caps each request at 1000 articles and returns the
    EARLIEST 1000 within your time_from/time_to window when sort=EARLIEST.
    If your window has more than 1000 matching articles for a topic, the
    extras are silently dropped and your headlines will cluster at the
    START of the window instead of spreading across it. Keep LOOKBACK_DAYS
    modest (default 30) unless you've checked your topic/window doesn't
    exceed the cap.
  - yfinance hourly ("1h") data is only available for roughly the last 730
    days. For older history you'd need daily ("1d") resolution instead.
  - A backtest showing an edge does NOT guarantee that edge holds going
    forward (markets adapt, and this is a small, simple signal).
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import yfinance as yf
from anthropic import Anthropic, APIStatusError

# Set to True to also send the summary to Telegram (needs TELEGRAM_BOT_TOKEN
# and TELEGRAM_CHAT_ID env vars — same ones your live bot uses).
SEND_TELEGRAM = True

# ---------- CONFIG ----------

ALPHA_VANTAGE_KEY = os.environ["ALPHAVANTAGE_API_KEY"]
client = Anthropic()  # reads ANTHROPIC_API_KEY from environment

GOLD_TICKER = "GC=F"          # COMEX gold futures on yfinance
LOOKBACK_DAYS = 30            # how far back to pull headlines (see cap note below)
FORWARD_WINDOWS_HOURS = (1, 4, 24)  # check price move 1h, 4h, and 24h after each headline
JUDGE_BATCH_SIZE = 20         # headlines per Claude call (keeps prompts manageable)

# Same relevance filter as the live bot, so the backtest tests the same
# universe of headlines the bot would actually see.
GOLD_KEYWORDS = ["gold", "bullion", "xau", "precious metal"]
MACRO_KEYWORDS = [
    "federal reserve", "fed chair", "fomc", "interest rate", "rate hike",
    "rate cut", "rate decision", "jackson hole", "central bank",
    "ecb", "bank of japan", "boj",
    "inflation", "cpi", "pce", "consumer price index", "core inflation",
    "dollar index", "u.s. dollar", "us dollar", "greenback", "treasury yield",
    "bond yield", "10-year yield", "real yields",
    "nonfarm payrolls", "jobs report", "unemployment rate", "recession",
    "gdp growth",
    "geopolitical", "war", "sanctions", "conflict", "banking crisis",
    "debt ceiling", "government shutdown", "sovereign debt",
    "budget deficit", "national debt", "treasury buyback",
    "stock market selloff", "market crash", "safe haven",
]


def is_relevant(title):
    t = title.lower()
    return any(k in t for k in GOLD_KEYWORDS) or any(k in t for k in MACRO_KEYWORDS)


# ---------- 1. FETCH HISTORICAL HEADLINES ----------

# Query topics one at a time (rather than comma-joined in one call) — more
# robust, and only uses a few of your 25 free daily requests per run.
NEWS_TOPICS = ["financial_markets", "economy_macro", "economy_monetary"]


def fetch_headlines_for_topic(topic, time_from, time_to, limit=1000):
    """Pull historical headlines for ONE topic from Alpha Vantage."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "topics": topic,
        "time_from": time_from.strftime("%Y%m%dT%H%M"),
        "time_to": time_to.strftime("%Y%m%dT%H%M"),
        "limit": limit,
        "sort": "EARLIEST",
        "apikey": ALPHA_VANTAGE_KEY,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "feed" not in data:
        print(f"  Alpha Vantage returned no 'feed' for topic '{topic}': {data}")
        return []

    headlines = []
    for item in data["feed"]:
        title = item.get("title", "").strip()
        ts_raw = item.get("time_published")  # format: YYYYMMDDTHHMMSS
        if not title or not ts_raw:
            continue
        try:
            ts = datetime.strptime(ts_raw, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        headlines.append({
            "title": title,
            "time_published": ts,
            "source": item.get("source", ""),
            "url": item.get("url", ""),
        })
    return headlines


def fetch_historical_headlines(time_from, time_to, limit=1000):
    """Pull historical headlines across all NEWS_TOPICS, deduped by URL.
    Uses one Alpha Vantage request per topic (len(NEWS_TOPICS) total)."""
    seen_urls = set()
    all_headlines = []
    for topic in NEWS_TOPICS:
        print(f"  Fetching topic '{topic}'...")
        for h in fetch_headlines_for_topic(topic, time_from, time_to, limit):
            if h["url"] and h["url"] in seen_urls:
                continue
            seen_urls.add(h["url"])
            all_headlines.append(h)
    return all_headlines


# ---------- 2. FETCH HISTORICAL GOLD PRICES ----------

def fetch_gold_prices(start, end, interval="1h"):
    """Returns a DataFrame indexed by UTC datetime with a 'Close' column."""
    df = yf.download(GOLD_TICKER, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError(
            "No price data returned. Check the date range and interval "
            "(yfinance hourly data usually only goes back ~730 days)."
        )
    # Recent yfinance versions return MultiIndex columns like ('Close', 'GC=F')
    # even for a single ticker. Flatten to plain column names ('Close', etc.)
    # so downstream code can index with df["Close"] as expected.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def price_at_or_after(df, ts):
    future = df[df.index >= ts]
    if future.empty:
        return None
    return future.iloc[0]


def _as_float(value):
    """Safely coerce a price value to a plain float, even if pandas hands
    back a length-1 Series (can happen with duplicate index timestamps)."""
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def forward_return(df, ts, hours_ahead):
    """% price change from the first bar at/after ts to the first bar at/after
    ts + hours_ahead. Returns None if either side falls outside the data."""
    entry = price_at_or_after(df, ts)
    if entry is None:
        return None
    target_ts = entry.name + timedelta(hours=hours_ahead)
    exit_bar = price_at_or_after(df, target_ts)
    if exit_bar is None:
        return None
    entry_close = _as_float(entry["Close"])
    exit_close = _as_float(exit_bar["Close"])
    return (exit_close - entry_close) / entry_close


# ---------- 3. JUDGE (same prompt/logic as the live bot, batched) ----------

JUDGE_PROMPT = """You are a financial news classifier. You will be given a numbered list of \
news headlines about gold or markets. For EACH headline, decide whether it is short-term \
BULLISH, BEARISH, or NEUTRAL for the gold price.

Rules of thumb:
- Hawkish Fed / rate hikes / strong dollar / rising real yields -> usually BEARISH for gold
- Dovish Fed / rate cuts / weak dollar / falling real yields -> usually BULLISH for gold
- Safe-haven demand (war, crisis, debt fears, bank stress) -> usually BULLISH for gold
- Risk-on rallies in stocks with calm markets -> usually mildly BEARISH for gold
- If a headline is ambiguous, mixed, or not really about a market-moving driver -> NEUTRAL

Respond ONLY with a compact JSON array, no other text, with exactly one object per headline, \
in the same order as given, in this exact shape:
[{"index": 1, "verdict": "bullish" | "bearish" | "neutral", "reason": "one short plain-English sentence"}, ...]

Headlines:
{headline_list}
"""


def judge_batch(batch):
    numbered = "\n".join(f"{i+1}. {h['title']}" for i, h in enumerate(batch))
    prompt = JUDGE_PROMPT.replace("{headline_list}", numbered)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200 + 100 * len(batch),
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        results = json.loads(raw)
        by_index = {r.get("index"): r for r in results if isinstance(r, dict)}

        out = []
        for i in range(len(batch)):
            r = by_index.get(i + 1)
            if not r:
                out.append({"verdict": "neutral", "reason": "(could not score)"})
                continue
            verdict = str(r.get("verdict", "neutral")).lower()
            if verdict not in ("bullish", "bearish", "neutral"):
                verdict = "neutral"
            out.append({"verdict": verdict, "reason": r.get("reason", "")})
        return out

    except APIStatusError as e:
        if e.status_code == 400 and "credit balance" in str(e).lower():
            print(f"BILLING ISSUE — stopping run: {e}")
            sys.exit(1)
        print(f"Judge error on batch: {e}")
        return [{"verdict": "neutral", "reason": "(error)"} for _ in batch]

    except Exception as e:
        print(f"Judge error on batch: {e}")
        return [{"verdict": "neutral", "reason": "(error)"} for _ in batch]


def judge_all_headlines(headlines, batch_size=JUDGE_BATCH_SIZE):
    results = []
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i + batch_size]
        print(f"  Judging headlines {i+1}-{i+len(batch)} of {len(headlines)}...")
        results.extend(judge_batch(batch))
        time.sleep(0.3)
    return results


# ---------- 4. SCORE THE BACKTEST ----------

def compute_baseline(price_df, windows=FORWARD_WINDOWS_HOURS):
    """For each window, what % of ALL hours in the price history saw gold go
    up over that window — regardless of any headline? This is the number to
    beat: if a headline-based hit rate isn't meaningfully above this, the
    'signal' is probably just riding the market's overall trend, not adding
    real information."""
    baseline = {}
    for w in windows:
        up_count = 0
        total = 0
        for ts in price_df.index:
            r = forward_return(price_df, ts, w)
            if r is None:
                continue
            total += 1
            if r > 0:
                up_count += 1
        baseline[w] = (up_count / total * 100) if total > 0 else None
    return baseline


def run_backtest(headlines, price_df, windows=FORWARD_WINDOWS_HOURS):
    verdicts = judge_all_headlines(headlines)

    rows = []
    for h, v in zip(headlines, verdicts):
        row = {
            "title": h["title"],
            "time_published": h["time_published"],
            "verdict": v["verdict"],
            "reason": v["reason"],
        }
        for w in windows:
            row[f"return_{w}h"] = forward_return(price_df, h["time_published"], w)
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_backtest(df, baseline=None, windows=FORWARD_WINDOWS_HOURS):
    """Prints the backtest summary AND returns it as a plain-text string
    (used for the Telegram message). `baseline` is the dict from
    compute_baseline() — the % of ALL hours (not just headline hours) where
    gold went up over each window, used as the "beat this" reference."""
    lines = []
    lines.append("=" * 60)
    lines.append("BACKTEST RESULTS")
    lines.append("=" * 60)

    for w in windows:
        col = f"return_{w}h"
        sub = df.dropna(subset=[col])
        if sub.empty:
            lines.append(f"\n--- {w}h forward window: no data (price history didn't cover this range) ---")
            continue

        base_up = baseline.get(w) if baseline else None
        base_str = f" | baseline: gold rose {base_up:.1f}% of ALL hours" if base_up is not None else ""
        lines.append(f"\n--- {w}h forward window ({len(sub)} headlines with price data){base_str} ---")
        for verdict in ("bullish", "bearish", "neutral"):
            v = sub[sub["verdict"] == verdict][col]
            if len(v) == 0:
                lines.append(f"  {verdict:8s} n=0")
                continue
            avg_return = v.mean() * 100
            if verdict == "bullish":
                hit_rate = (v > 0).mean() * 100
            elif verdict == "bearish":
                hit_rate = (v < 0).mean() * 100
            else:
                hit_rate = None
            hit_str = f"  hit_rate={hit_rate:.1f}%" if hit_rate is not None else ""
            lines.append(f"  {verdict:8s} n={len(v):4d}  avg_return={avg_return:+.3f}%{hit_str}")

        # Naive strategy: go long on "bullish" calls, short on "bearish" calls,
        # sit flat on "neutral". No compounding, no costs — direction-only check.
        def signal_return(r):
            if r["verdict"] == "bullish":
                return r[col]
            elif r["verdict"] == "bearish":
                return -r[col]
            return 0.0

        strat_returns = sub.apply(signal_return, axis=1)
        lines.append(f"  Naive directional strategy: avg return/signal = {strat_returns.mean()*100:+.3f}%, "
                      f"sum (no compounding) = {strat_returns.sum()*100:+.3f}%")

    lines.append("\nReminder: this is directional accuracy only — no spread, slippage, or")
    lines.append("execution delay included. A positive result here is a reason to keep")
    lines.append("investigating, not a green light to trade real money.")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    return summary_text


def format_telegram_summary(df, baseline=None, windows=FORWARD_WINDOWS_HOURS):
    """A plain-English, HTML-formatted version of the summary for Telegram
    (Telegram messages have a 4096-char limit, so keep this compact).
    `baseline` is the dict from compute_baseline() — used to show whether
    the bot is beating "gold was just trending" or not."""
    lines = ["<b>Gold Backtest — Weekly Report</b>", ""]

    for w in windows:
        col = f"return_{w}h"
        sub = df.dropna(subset=[col])
        if sub.empty:
            lines.append(f"<b>{w} hours after the headline:</b> no price data")
            continue

        base_up = baseline.get(w) if baseline else None
        lines.append(f"<b>Looking {w} hours after each headline</b> ({len(sub)} headlines checked)")
        if base_up is not None:
            lines.append(f"<i>(For comparison: gold rose after {base_up:.0f}% of ALL {w}h periods in this stretch, headline or not.)</i>")

        for verdict in ("bullish", "bearish", "neutral"):
            v = sub[sub["verdict"] == verdict][col]
            if len(v) == 0:
                continue
            avg_return = v.mean() * 100
            emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}[verdict]
            direction = "up" if avg_return >= 0 else "down"

            if verdict == "bullish":
                hit_rate = (v > 0).mean() * 100
                beat_str = ""
                if base_up is not None:
                    diff = hit_rate - base_up
                    beat_str = f" (baseline: {base_up:.0f}%, {'beats it' if diff > 3 else 'about the same as' if abs(diff) <= 3 else 'below it'})"
                lines.append(
                    f"{emoji} {len(v)} headlines called BULLISH — gold moved {direction} "
                    f"{abs(avg_return):.2f}% on average afterward. It was actually right "
                    f"(price went up) {hit_rate:.0f}% of the time{beat_str}."
                )
            elif verdict == "bearish":
                hit_rate = (v < 0).mean() * 100
                base_down = (100 - base_up) if base_up is not None else None
                beat_str = ""
                if base_down is not None:
                    diff = hit_rate - base_down
                    beat_str = f" (baseline: {base_down:.0f}%, {'beats it' if diff > 3 else 'about the same as' if abs(diff) <= 3 else 'below it'})"
                lines.append(
                    f"{emoji} {len(v)} headlines called BEARISH — gold moved {direction} "
                    f"{abs(avg_return):.2f}% on average afterward. It was actually right "
                    f"(price went down) {hit_rate:.0f}% of the time{beat_str}."
                )
            else:
                lines.append(
                    f"{emoji} {len(v)} headlines called NEUTRAL — gold moved {direction} "
                    f"{abs(avg_return):.2f}% on average afterward (no strong call made)."
                )
        lines.append("")

    lines.append("\"Beats it\" means the bot's calls were right meaningfully more often")
    lines.append("than gold's normal up/down tendency during this period — i.e. the")
    lines.append("signal may add real information, not just ride a general trend.")
    lines.append("This checks direction only, not real trading profit — no fees, ")
    lines.append("spreads, or slippage included. Full details in this week's CSV.")
    return "\n".join(lines)


def send_telegram_message(text):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, data={
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    })
    if resp.status_code != 200:
        print(f"Telegram send failed: {resp.status_code} {resp.text}")
    else:
        print("Telegram message sent.")


# ---------- MAIN ----------

def main():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)

    print(f"Fetching headlines from {start.date()} to {end.date()}...")
    raw_headlines = fetch_historical_headlines(start, end)
    headlines = [h for h in raw_headlines if is_relevant(h["title"])]
    print(f"{len(raw_headlines)} total headlines fetched, {len(headlines)} relevant after filtering.")

    if not headlines:
        print("No relevant headlines found in this window. Try widening LOOKBACK_DAYS or the topics filter.")
        return

    print("Fetching gold price history...")
    price_df = fetch_gold_prices(start - timedelta(days=1), end + timedelta(days=2), interval="1h")
    print(f"Got {len(price_df)} hourly price bars.")

    print("Computing baseline (gold's normal up/down tendency, independent of any headline)...")
    baseline = compute_baseline(price_df)

    print(f"Judging {len(headlines)} headlines in batches of {JUDGE_BATCH_SIZE}...")
    df = run_backtest(headlines, price_df)

    out_path = "backtest_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")

    summarize_backtest(df, baseline=baseline)

    if SEND_TELEGRAM:
        telegram_text = format_telegram_summary(df, baseline=baseline)
        send_telegram_message(telegram_text)


if __name__ == "__main__":
    main()
