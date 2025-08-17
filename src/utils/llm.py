import os
from typing import List
from dotenv import load_dotenv

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()

class SentimentAndReplyLLM:
    def __init__(self):
        self.use_openai = bool(OPENAI_KEY)
        self._init_clients()

    def _init_clients(self):
        self.chat = None
        self.sentiment_pipe = None
        if self.use_openai:
            try:
                from langchain_openai import ChatOpenAI
                self.chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, timeout=30)
            except Exception:
                self.use_openai = False
        if not self.use_openai:
            from transformers import pipeline
            # Fast local pipeline for sentiment
            self.sentiment_pipe = pipeline("sentiment-analysis")

        # threshold for deciding neutral when using HF binary model
        self.neutral_floor = 0.45

    # ---------- SINGLE TEXT ----------
    def classify_sentiment(self, text: str) -> str:
        if self.use_openai and self.chat is not None:
            try:
                prompt = (
                    "Classify the restaurant review strictly as one of: positive, negative, or neutral.\n"
                    f"Review: {text}\n"
                    "Answer with one word only."
                )
                result = self.chat.invoke(prompt)
                label = result.content.strip().lower()
                if "pos" in label: return "positive"
                if "neg" in label: return "negative"
                return "neutral"
            except Exception:
                # Any API error (429, network, etc.) → fallback
                self.use_openai = False

        # Fallback to HF (binary + neutral heuristic)
        pred = self.sentiment_pipe(text[:512])[0]
        lab = pred["label"].lower()
        score = float(pred.get("score", 0.0))
        if lab.startswith("pos"):
            return "positive" if score >= self.neutral_floor else "neutral"
        else:
            return "negative" if score >= self.neutral_floor else "neutral"

    # ---------- BATCH CLASSIFICATION ----------
    def classify_series(self, texts: List[str], batch_size: int = 64) -> List[str]:
        """
        Efficient for many rows. If OpenAI is selected, we still fallback to local HF
        to avoid rate limits. (Avoid per-row API calls.)
        """
        # Always use local HF for batches to avoid rate limiting
        if self.sentiment_pipe is None:
            from transformers import pipeline
            self.sentiment_pipe = pipeline("sentiment-analysis")

        out = []
        for i in range(0, len(texts), batch_size):
            chunk = [t[:512] for t in texts[i:i+batch_size]]
            preds = self.sentiment_pipe(chunk)
            for p in preds:
                lab = p["label"].lower()
                score = float(p.get("score", 0.0))
                if lab.startswith("pos"):
                    out.append("positive" if score >= self.neutral_floor else "neutral")
                else:
                    out.append("negative" if score >= self.neutral_floor else "neutral")
        return out

    def craft_reply(self, text: str, sentiment: str) -> str:
        if self.use_openai and self.chat is not None:
            try:
                prompt = f"""
You are SteamNoodles' friendly support agent. The review sentiment is {sentiment}.
Write a short, polite, personalized reply (2–3 sentences). If negative, apologize and offer to make it right; if neutral, thank and invite suggestions; if positive, appreciate and invite them back. Keep brand voice warm and concise.
Review: "{text}"
"""
                result = self.chat.invoke(prompt)
                return result.content.strip()
            except Exception:
                self.use_openai = False

        # Local template replies
        snippet = (text[:150] + "…") if len(text) > 160 else text
        if sentiment == "positive":
            return ("Thanks so much for the lovely feedback! We’re thrilled you enjoyed your visit. "
                    f"Hope to serve you again soon at SteamNoodles. ({snippet})")
        if sentiment == "negative":
            return ("Sorry to hear about your experience — we take this seriously. "
                    f"Please DM us so we can make it right on your next visit. ({snippet})")
        return ("Thank you for sharing your thoughts. We’re always improving — "
                f"any suggestions are welcome, and we hope to see you again. ({snippet})")
