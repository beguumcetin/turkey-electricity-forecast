"""Download hourly real-time consumption and EPİAŞ's day-ahead load plan.

Needs a free EPİAŞ Transparency Platform account. Put the credentials in a
.env file at the project root (see .env.example).

    python -m src.fetch_consumption
"""

import os
import time
from datetime import date, timedelta

import pandas as pd
from eptr2 import EPTR2

from src.config import RAW_DIR, ROOT, START_DATE, TIMEZONE


def month_ranges(start: date, end: date):
    """Yield (first_day, last_day) pairs month by month; the API rejects long ranges."""
    cur = start.replace(day=1)
    while cur <= end:
        nxt = (cur + timedelta(days=32)).replace(day=1)
        yield max(cur, start), min(nxt - timedelta(days=1), end)
        cur = nxt


def call_with_retry(eptr: EPTR2, call: str, s: date, e: date, tries: int = 6) -> pd.DataFrame:
    """EPİAŞ allows 80 requests per minute and answers 429 beyond that. Pause between
    calls and back off when throttled."""
    for attempt in range(tries):
        try:
            df = eptr.call(call, start_date=s.isoformat(), end_date=e.isoformat())
            time.sleep(1.0)
            return df
        except Exception as exc:
            if "429" not in str(exc) or attempt == tries - 1:
                raise
            wait = 20 * (attempt + 1)
            print(f"throttled, waiting {wait}s")
            time.sleep(wait)


def fetch(eptr: EPTR2, call: str, value_col: str, start: date, end: date) -> pd.DataFrame:
    frames = []
    for s, e in month_ranges(start, end):
        df = call_with_retry(eptr, call, s, e)
        print(f"{call}: {s} – {e}  ({len(df)} rows)")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame(
        {
            "time": pd.to_datetime(df["date"], utc=True).dt.tz_convert(TIMEZONE),
            value_col: pd.to_numeric(df[value_col if value_col in df else df.columns[-1]]),
        }
    )
    return out.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def main():
    os.chdir(ROOT)  # eptr2 reads .env and caches its login ticket in the working directory
    eptr = EPTR2(use_dotenv=True, dotenv_path=str(ROOT / ".env"))
    start = date.fromisoformat(START_DATE)
    end = date.today() - timedelta(days=1)

    cons = fetch(eptr, "rt-cons", "consumption", start, end)
    plan = fetch(eptr, "load-plan", "lep", start, end).rename(columns={"lep": "epias_plan"})

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cons.to_parquet(RAW_DIR / "consumption.parquet", index=False)
    plan.to_parquet(RAW_DIR / "load_plan.parquet", index=False)
    print(f"Saved {len(cons)} consumption hours and {len(plan)} load plan hours to {RAW_DIR}")


if __name__ == "__main__":
    main()
