# SteamNoodles Feedback Agents - Kabilashika

This repo implements two agents for Automated Restaurant Feedback (SteamNoodles):

- **Agent 1 - Feedback Response Agent**
  Input: one review text → Output: sentiment (positive/neutral/negative) + a short, polite auto-reply.  
  Uses "OpenAI GPT if `OPENAI_API_KEY` is set; otherwise falls back to a "local Hugging Face" sentiment model.

- **Agent 2 - Sentiment Visualization Agent**
  Input: natural-language date range (e.g., “last 7 days”, “June 1 to June 15, 2022”, or “all time”).  
  Output: a "line or bar"**" plot of daily counts by sentiment, saved under `outputs/`.

---

## Project Folder Structure

```
.
├─ data/
│  └─ yelp_reviews.csv            # dataset (Date, Rating, Review Text)
├─ outputs/                       # generated images (gitignored)
├─ src/
│  ├─ agents/
│  │  ├─ response_agent.py        # Agent 1 (auto-reply)
│  │  └─ plot_agent.py            # Agent 2 (sentiment plotting)
│  └─ utils/
│     ├─ data_loader.py           # robust CSV loader & cleaning
│     └─ llm.py                   # OpenAI/HF wrapper (sentiment + replies)
├─ .env                           # OPENAI_API_KEY
├─ .gitignore
├─ requirements.txt
├─ README.md
└─ main.py                        # CLI entrypoint
```

---

## Setup

``` run this is in the teminal to setup the environment
python -m venv .venv
# Windows PowerShell:
. .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**Dataset:** placed at `data/yelp_reviews.csv`:
- `Date` 
- `Rating` (1–5)
- `Review Text`
- `Keggle link for my dataset` https://www.kaggle.com/datasets/farukalam/yelp-restaurant-reviews

**OpenAI key:** create `.env` in project root:
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
I have created the OPENAI_API_KEY from https://platform.openai.com/api-keys
```
If not provided, the project uses a local HF model automatically.

---

## How to Test Each Agent

> Run these from the project root (same folder as `main.py`) with the virtual environment activated.

### Agent 1 - Feedback Response Agent

```bash
python main.py reply --text "The noodles were fantastic and the staff was super friendly!"

python main.py reply --text "Service was quick and the ramen was delicious!"

python main.py reply --text "the service was not statisfied. the food is not delicious as expected."
```

**Expected (example):**
```
Sentiment: positive
Auto-reply:
 Thank you so much for your kind words! We're thrilled to hear you enjoyed our noodles and had a great experience with our team. We can't wait to welcome you back for another delicious meal!

Sentiment: positive
Auto-reply:
 Thank you so much for your kind words! We're thrilled to hear you enjoyed our quick service and delicious ramen. We can't wait to welcome you back for another tasty experience!

Sentiment: negative
Auto-reply:
 Hi there! I'm truly sorry to hear that your experience didn't meet your expectations. We strive for delicious food and great service, and I’d love the opportunity to make it right. Please reach out to us directly so we can address your concerns and ensure a better experience next time. Thank you for your feedback! 
```

### Agent 2 - Sentiment Visualization Agent

**Concrete window (line plot):**
```bash
python main.py plot --csv data/yelp_reviews.csv --range "June 1 to June 15, 2022" --out outputs/sentiment_june_1_15_2022.png
```

**All-time (bar chart):**
```bash
python main.py plot --csv data/yelp_reviews.csv --range "all time" --kind bar --out outputs/sentiment_all_bar.png
```

Open the generated image in `outputs/` to view daily counts for **negative / neutral / positive**.

> # Note: If we a need relative range like “last 7 days” but the dataset is historical (e.g., ends in 2022), the agent automatically raise ValueError(f"Could not parse date range: {user_range}").


---


## Sample Prompts & Expected Outputs

**Agent 1**
- Prompt:  
  `python main.py reply --text "Service was slow, but the staff apologized and fixed my order."` 
  `python main.py reply --text "My order arrived cold and took almost 40 minutes."`
  `python main.py reply --text "Loved the spicy miso—rich flavor, and the team was super friendly!"`
- Expected:  
  `Sentiment:` neutral/negative (depends on model confidence)  
  `Auto-reply:` brief apology + invite to DM/return

**Agent 2**
- Prompt:  
  `--range "Jan 1 2022 to Aug 2 2022" --kind line`  
- Expected:  
  A line chart of daily sentiment counts saved to `outputs/sentiment_2022.png`.
For GitHub viewing, a copy of this image is included at docs/sentiment_2022.png. Because `outputs/` is usually gitignored, so adding a copy to `docs/` lets viewers see it on GitHub.

---

## Notes & Troubleshooting

- **No reviews found**: You picked a window with no data. The error will show the **available dataset range** (e.g., 2005-06-24 to 2022-08-02). Try `--range "all time"` or a window inside that span.
- **Performance**: For plotting, the agent maps **star ratings → sentiment** (fast). LLM is only used if ratings are missing.
- **Matplotlib not found**: Activate the venv and `pip install -r requirements.txt`.
- **OpenAI usage**: If `.env` lacks a key, the system automatically uses a local HF model.

---

## Demo Output
The sample output is stored in the **outputs** folder. The **response** png consist the sample feedback response of the Response Agent 
and other images are the plots generated by the Plot Agent 

## Submission Info

**Name:** Kabilashika  Thiyagarajah

**University:** University of Moratuwa   

**Year:** 4th year
