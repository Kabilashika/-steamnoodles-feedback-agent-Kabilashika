import os
import pandas as pd
import matplotlib.pyplot as plt
import dateparser

from src.utils.data_loader import load_reviews
from src.utils.llm import SentimentAndReplyLLM

def stars_to_sentiment(val):
    try:
        s = float(val)
    except Exception:
        return None
    if pd.isna(s):
        return None
    if s <= 2:
        return "negative"
    if s >= 4:
        return "positive"
    return "neutral"

def _is_relative_phrase(s: str) -> bool:
    s = (s or "").strip().lower()
    return any(word in s for word in ["last", "yesterday", "today", "this ", "past ", "previous "])

class SentimentPlotAgent:
    """
    Input: natural-language date range ("last 7 days", "June 1 to June 15 2022", "all time")
    Output: PNG plot saved under outputs/
    """
    def __init__(self, csv_path: str, model_for_sentiment: SentimentAndReplyLLM | None = None):
        self.csv_path = csv_path
        self.df = load_reviews(csv_path)  # -> review_text, review_dt, stars
        self.llm = model_for_sentiment or SentimentAndReplyLLM()

    def _parse_range_today(self, user_range: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        s = (user_range or "").strip().lower()
        if "to" in s:
            left, right = s.split("to", 1)
            start = dateparser.parse(left.strip())
            end = dateparser.parse(right.strip())
        else:
            ref = pd.Timestamp.now()
            start = dateparser.parse(s, settings={"RELATIVE_BASE": ref.to_pydatetime()})
            end = ref
        if start is None or end is None:
            raise ValueError(f"Could not parse date range: {user_range}")
        start = pd.to_datetime(start).normalize()
        end = pd.to_datetime(end).normalize()
        if start > end:
            start, end = end, start
        return start, end

    def _parse_range_relative_to(self, user_range: str, ref: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        # Re-interpret relative phrases using dataset's max date as "now"
        s = (user_range or "").strip().lower()
        if "to" in s:
            # If it's "June 1 to June 15" it's absolute; just return as-is via today parser
            return self._parse_range_today(user_range)
        start = dateparser.parse(s, settings={"RELATIVE_BASE": ref.normalize().to_pydatetime()})
        end = ref.normalize()
        if start is None:
            raise ValueError(f"Could not parse date range relative to dataset: {user_range}")
        start = pd.to_datetime(start).normalize()
        if start > end:
            start, end = end, start
        return start, end

    def run(self, user_range: str, out_path: str = "outputs/sentiment_by_day.png", kind: str = "line") -> str:
        if "review_dt" not in self.df.columns or self.df["review_dt"].isna().all():
            raise RuntimeError("Dataset has no valid dates; cannot plot by day.")

        base = self.df.dropna(subset=["review_dt"]).copy()

        # Support "all time"
        s = (user_range or "").strip().lower()
        if s in {"all", "all time", "full", "entire", "everything"}:
            start, end = base["review_dt"].min().normalize(), base["review_dt"].max().normalize()
        else:
            # First try relative to TODAY
            start, end = self._parse_range_today(user_range)

        # Filter
        df = base[(base["review_dt"] >= start) & (base["review_dt"] <= end)].copy()

        # If empty & it's a relative phrase, retry relative to dataset's latest date
        if df.empty and _is_relative_phrase(user_range):
            ref = base["review_dt"].max()
            start, end = self._parse_range_relative_to(user_range, ref)
            df = base[(base["review_dt"] >= start) & (base["review_dt"] <= end)].copy()

        if df.empty:
            min_dt, max_dt = base["review_dt"].min(), base["review_dt"].max()
            raise RuntimeError(
                f"No reviews found between {start.date()} and {end.date()}.\n"
                f"Available data range: {min_dt.date()} to {max_dt.date()}.\n"
                f'Try: --range "all time" or a window inside that span.'
            )

        # Use rating → sentiment (fast); fallback to LLM only if stars missing
        if "stars" in df.columns and df["stars"].notna().any():
            df["sentiment"] = df["stars"].apply(stars_to_sentiment)
        else:
            df["sentiment"] = self.llm.classify_series(df["review_text"].fillna("").tolist(), batch_size=32)

        # Aggregate per day
        df["day"] = df["review_dt"].dt.date
        counts = (
            df.groupby(["day", "sentiment"]).size()
              .reset_index(name="count")
              .pivot(index="day", columns="sentiment", values="count")
              .fillna(0)
              .sort_index()
        )
        for c in ("negative", "neutral", "positive"):
            if c not in counts.columns:
                counts[c] = 0
        counts = counts[["negative", "neutral", "positive"]]

        # Plot
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.figure(figsize=(10, 5))
        counts.plot(kind=kind, ax=plt.gca())
        plt.title(f"SteamNoodles Sentiment by Day ({start.date()} to {end.date()})")
        plt.xlabel("Day")
        plt.ylabel("Review Count")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return out_path
