from typing import Dict
from src.utils.llm import SentimentAndReplyLLM

class FeedbackResponseAgent:
    """
    Agent 1:
    - Input: review text
    - Output: {sentiment, reply}
    """
    def __init__(self):
        self.llm = SentimentAndReplyLLM()

    def run(self, review_text: str) -> Dict[str, str]:
        sentiment = self.llm.classify_sentiment(review_text)
        reply = self.llm.craft_reply(review_text, sentiment)
        return {"sentiment": sentiment, "reply": reply}
