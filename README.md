# Gold Sentiment Bot 🪙

A tiny robot that reads gold news every 3 hours and texts you a bullish/bearish
verdict on Telegram. Runs for free using GitHub Actions as the "always-on" timer
— no server to rent or maintain.

## How it works

```
Timer (every 3h) → collect_headlines() → judge_headline() for each → tally votes → Telegram message
```

- **Collect**: pulls headlines from a few free gold/markets RSS feeds
- **Judge**: sends each headline to Claude, asking "bullish, bearish, or neutral?"
- **Add it up**: counts the votes into one overall verdict
- **Tell you**: sends a formatted summary to your Telegram

## One-time setup (10 minutes)

### 1. Create a GitHub repository
- Go to github.com, click "New repository", name it e.g. `gold-sentiment-bot`
- Upload the three files/folders from this project: `gold_bot.py`, `requirements.txt`,
  and the `.github/workflows/run-bot.yml` folder (keep that folder structure exactly)

### 2. Add your secrets
GitHub Actions needs your keys, but they should **never** be typed into the code itself.
Instead:
- In your new repo, go to **Settings → Secrets and variables → Actions**
- Click **New repository secret** three times, adding:
  - `ANTHROPIC_API_KEY` — your key from console.anthropic.com
  - `TELEGRAM_BOT_TOKEN` — the token BotFather gave you
  - `TELEGRAM_CHAT_ID` — your numeric ID from @userinfobot

### 3. Test it manually
- Go to the **Actions** tab in your repo
- Click **Gold Sentiment Bot** in the left list, then **Run workflow** (top right)
- Wait ~30 seconds, refresh — you should get a Telegram message

### 4. Let it run itself
That's it. The schedule in `run-bot.yml` fires automatically every 3 hours on
weekdays (market hours-ish, UTC time). No further action needed.

## Tuning it later

- **Change how often it runs**: edit the `cron:` line in `run-bot.yml`.
  (Format is minute hour day month weekday — [crontab.guru](https://crontab.guru) helps.)
- **Add more news sources**: add RSS feed URLs to the `RSS_FEEDS` list in `gold_bot.py`.
- **Change the judging rules**: edit the `JUDGE_PROMPT` text in `gold_bot.py`.
- **Make it less chatty**: lower the `[:8]` slice in `format_message()` to show fewer headlines per message.

## Honest limitations

- News sentiment is a noisy signal by itself — it's a research tool, not a trading signal.
- Free RSS feeds occasionally go down or change format; if headlines stop showing up,
  check the feed URLs still work.
- This is not financial advice.
