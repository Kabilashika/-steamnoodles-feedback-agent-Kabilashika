import argparse
from src.agents.response_agent import FeedbackResponseAgent
from src.agents.plot_agent import SentimentPlotAgent

def agent_reply(args):
    agent = FeedbackResponseAgent()
    result = agent.run(args.text)
    print("Sentiment:", result["sentiment"])
    print("Auto-reply:\n", result["reply"])

def agent_plot(args):
    agent = SentimentPlotAgent(csv_path=args.csv)
    out = agent.run(args.range, out_path=args.out, kind=args.kind)
    print(f"Plot saved to: {out}")

def main():
    parser = argparse.ArgumentParser(description="SteamNoodles Multi-Agent CLI")
    sub = parser.add_subparsers(required=True)

    # Agent 1: reply
    p1 = sub.add_parser("reply", help="Generate automated reply for a review")
    p1.add_argument("--text", required=True, help="The review text")
    p1.set_defaults(func=agent_reply)

    # Agent 2: plot
    p2 = sub.add_parser("plot", help="Generate daily sentiment plot over a date range")
    p2.add_argument("--csv", required=True, help="Path to CSV (in data/)")
    p2.add_argument(
        "--range",
        required=True,
        help='Date range, e.g. "last 7 days", "June 1 to June 15, 2022", or "all time"'
    )
    p2.add_argument(
        "--kind",
        choices=["line", "bar"],
        default="line",
        help="Type of plot to generate (default: line)"
    )
    p2.add_argument("--out", default="outputs/sentiment_by_day.png", help="Output image path")
    p2.set_defaults(func=agent_plot)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
