"""
Gold Sentiment Bot
-------------------
1. COLLECT   - pull recent gold-related headlines from a few free RSS feeds
2. JUDGE     - ask Claude to score ALL headlines in ONE call: bullish / bearish / neutral
3. ADD IT UP - tally the votes into one verdict
4. TELL YOU  - send the verdict + evidence to Telegram

Run it with: python gold_bot.py
Needs three environment variables (set as GitHub Actions secrets in production):
  ANTHROPIC_API_KEY
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
"""

import os
import json
import sys
import feedparser
import requests
from anthropic import Anthropic, APIStatusError

# ---------- 1. COLLECT ----------

# Free RSS feeds covering gold directly, plus general markets/economy feeds
# (gold-moving news often shows up there before it's framed as "gold news").
RSS_FEEDS = [
    "https://www.investing.com/rss/commodities_Gold.rss",
    "https://www.fxstreet.com/rss/news",
    "https://www.kitco.com/rss/KitcoNews.xml",
    "https://www.investing.com/rss/news_25.rss",          # economic indicators
    "https://www.investing.com/rss/news_285.rss",         # Fed / central banks
    "https://www.marketwatch.com/rss/topstories",
    "https://www.cnbc.com/id/20910258/device/rss/rss.html",  # CNBC top news
    "https://www.federalreserve.gov/feeds/press_all.xml",  # Fed press releases, straight from the source
]

# Tier 1: headline mentions gold/precious metals directly -> always keep.
GOLD_KEYWORDS = ["gold", "bullion", "xau", "precious metal"]

# Tier 2: no "gold" word, but these are the big levers that move gold's price
# even when gold isn't named in the headline (rates, inflation, dollar,
# safe-haven triggers, and gold's biggest buyers/sellers).
MACRO_KEYWORDS = [
    # central bank / interest rates
    "federal reserve", "fed chair", "fomc", "interest rate", "rate hike",
    "rate cut", "rate decision", "jackson hole", "central bank",
    "ecb", "bank of japan", "boj",
    # inflation
    "inflation", "cpi", "pce", "consumer price index", "core inflation",
    # currency & bonds
    "dollar index", "u.s. dollar", "us dollar", "greenback", "treasury yield",
    "bond yield", "10-year yield", "real yields",
    # jobs / growth
    "nonfarm payrolls", "jobs report", "unemployment rate", "recession",
    "gdp growth",
    # safe-haven triggers
    "geopolitical", "war", "sanctions", "conflict", "banking crisis",
    "debt ceiling", "government shutdown", "sovereign debt",
    # fiscal / debt
    "budget deficit", "national debt", "treasury buyback",
    # market stress (risk-off often = gold up)
    "stock market selloff", "market crash", "safe haven",
]

MAX_HEADLINES = 15  # keep the AI bill small and the message short


def collect_headlines():
    """Pull headlines from RSS feeds, keeping ones about gold directly
    or about the big macro levers that move gold's price."""
    headlines = []

    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                title = entry.get("title", "").strip()
                if not title:
                    continue
                title_lower = title.lower()
                is_gold = any(k in title_lower for k in GOLD_KEYWORDS)
                is_macro = any(k in title_lower for k in MACRO_KEYWORDS)
                if is_gold or is_macro:
                    headlines.append({
                        "title": title,
                        "source": feed.feed.get("title", url),
                        "link": entry.get("link", ""),
                    })
        except Exception as e:
            print(f"Could not read feed {url}: {e}")

    # de-duplicate by title, cap the total
    seen = set()
    unique = []
    for h in headlines:
        if h["title"] not in seen:
            seen.add(h["title"])
            unique.append(h)

    return unique[:MAX_HEADLINES]


# ---------- 2. JUDGE (all headlines in a single API call) ----------

client = Anthropic()  # reads ANTHROPIC_API_KEY from environment automatically

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


def judge_all_headlines(headlines):
    """Score every headline in a single API call. Returns a list of
    (verdict, reason) tuples in the same order as the input headlines."""
    numbered = "\n".join(f"{i+1}. {h['title']}" for i, h in enumerate(headlines))
    prompt = JUDGE_PROMPT.replace("{headline_list}", numbered)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200 + 100 * len(headlines),  # scale room with headline count
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        results = json.loads(raw)

        # map by index so we're robust to the model re-ordering slightly
        by_index = {r.get("index"): r for r in results if isinstance(r, dict)}

        scored = []
        for i, h in enumerate(headlines):
            r = by_index.get(i + 1)
            if not r:
                scored.append(("neutral", "(could not score this one)"))
                continue
            verdict = str(r.get("verdict", "neutral")).lower()
            if verdict not in ("bullish", "bearish", "neutral"):
                verdict = "neutral"
            reason = r.get("reason", "")
            scored.append((verdict, reason))
        return scored

    except APIStatusError as e:
        if e.status_code == 400 and "credit balance" in str(e).lower():
            print(f"BILLING ISSUE — stopping run: {e}")
            sys.exit(1)  # don't fall back to 15 fake "neutral" scores, just quit loudly
        print(f"Judge error (API): {e}")
        return [("neutral", "(could not score this one)") for _ in headlines]

    except Exception as e:
        print(f"Judge error: {e}")
        return [("neutral", "(could not score this one)") for _ in headlines]


# ---------- 3. ADD IT UP ----------

def build_verdict(scored):
    bulls = sum(1 for s in scored if s["verdict"] == "bullish")
    bears = sum(1 for s in scored if s["verdict"] == "bearish")
    neutrals = sum(1 for s in scored if s["verdict"] == "neutral")

    score = bulls - bears
    if score >= 2:
        overall = "🟢 LEANING BULLISH"
    elif score <= -2:
        overall = "🔴 LEANING BEARISH"
    else:
        overall = "⚪ MIXED / UNCLEAR"

    return overall, bulls, bears, neutrals


# ---------- 4. TELL YOU (Telegram) ----------

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


def format_message(overall, bulls, bears, neutrals, scored):
    lines = [
        f"<b>Gold Sentiment Update</b>",
        f"{overall}",
        f"🟢 {bulls} bullish · 🔴 {bears} bearish · ⚪ {neutrals} neutral",
        "",
    ]
    for s in scored[:8]:  # keep the message from getting huge
        emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}[s["verdict"]]
        lines.append(f"{emoji} {s['title']}")
        lines.append(f"   <i>{s['reason']}</i>")
    lines.append("")
    lines.append("Not financial advice — a rough news-sentiment read only.")
    return "\n".join(lines)


# ---------- MAIN ----------

def main():
    print("Collecting headlines...")
    headlines = collect_headlines()

    if not headlines:
        print("No gold-related headlines found this run. Skipping.")
        return

    print(f"Found {len(headlines)} headlines. Judging all of them in one call...")
    results = judge_all_headlines(headlines)
    scored = [
        {**h, "verdict": v, "reason": r}
        for h, (v, r) in zip(headlines, results)
    ]

    overall, bulls, bears, neutrals = build_verdict(scored)
    message = format_message(overall, bulls, bears, neutrals, scored)

    print(message)
    send_telegram_message(message)


if __name__ == "__main__":
    main()
