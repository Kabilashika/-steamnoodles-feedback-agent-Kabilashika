import os
import pandas as pd

def _clean_text_series(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    try:
        s = s.apply(lambda x: x.encode('latin1', 'ignore').decode('utf-8', 'ignore'))
    except Exception:
        pass
    return s.str.replace(r'\s+', ' ', regex=True).str.strip()

def _read_any_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".tsv", ".txt"]:
        errors = []
        for enc in ["utf-8", "ISO-8859-1", "cp1252"]:
            try:
                # sep=None + engine='python' makes pandas sniff commas/semicolons/tabs
                return pd.read_csv(path, encoding=enc, on_bad_lines="skip", sep=None, engine="python")
            except Exception as e:
                errors.append(f"{enc}: {e}")
        raise RuntimeError("Could not read as delimited text with utf-8/ISO-8859-1/cp1252.\n" + "\n".join(errors))
    if ext in [".xlsx", ".xls"]:
        try:
            return pd.read_excel(path)
        except Exception as e:
            raise RuntimeError(f"Could not read Excel file: {e}")

    # Last resort
    return pd.read_csv(path, on_bad_lines="skip", sep=None, engine="python")

def load_reviews(csv_path: str) -> pd.DataFrame:
    df = _read_any_table(csv_path)
    lower = {c.lower(): c for c in df.columns}

    # Detect review text
    review_col = None
    for key in ["review text", "review", "text", "review_text"]:
        if key in lower:
            review_col = lower[key]; break
    if review_col is None:
        cand = [c for c in df.columns if "review" in c.lower() and "time" not in c.lower() and "date" not in c.lower()]
        if cand: review_col = cand[0]
    if review_col is None:
        raise ValueError(f"Couldn't find review text column. Found: {list(df.columns)}")

    # Detect time column
    time_col = None
    for key in ["date", "review_time", "time", "timestamp"]:
        if key in lower:
            time_col = lower[key]; break

    # Detect rating (optional)
    rating_col = None
    for key in ["rating", "stars", "star rating"]:
        if key in lower:
            rating_col = lower[key]; break

    # Clean text
    df[review_col] = _clean_text_series(df[review_col])

    # Parse datetime (remove deprecated infer_datetime_format arg)
    if time_col is not None:
        df["review_dt"] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        df["review_dt"] = pd.NaT

    # Normalize rating
    if rating_col is not None:
        stars = df[rating_col].astype(str).str.extract(r"(\d+\.?\d*)", expand=False)
        df["stars"] = pd.to_numeric(stars, errors="coerce")
    else:
        df["stars"] = pd.NA

    out = df.rename(columns={review_col: "review_text"})
    keep = ["review_text", "review_dt", "stars"]
    for extra in ["Yelp URL", "reviewer_id", "store_name", "rating_count"]:
        if extra in out.columns and extra not in keep:
            keep.append(extra)
    return out[keep]
