import re, pandas as pd, numpy as np

def normalize_imdb(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["review","sentiment"]).copy()
    df["review"] = df["review"].astype(str).str.lower()
    df["sentiment"] = df["sentiment"].astype(str).str.lower()
    return df

def compute_bayes_table(df: pd.DataFrame, keywords, direction="positive") -> pd.DataFrame:
    sentiments = set(df["sentiment"].unique())
    if {"positive","negative"} <= sentiments:
        pos, neg = "positive", "negative"
    elif {"pos","neg"} <= sentiments:
        pos, neg = "pos", "neg"
    else:
        raise ValueError(f"Unexpected labels: {sentiments}")

    if direction not in ("positive","negative"):
        raise ValueError("direction must be 'positive' or 'negative'")

    target = pos if direction=="positive" else neg
    prior = (df["sentiment"] == target).mean()

    reviews_all = df["review"]
    reviews_target = df.loc[df["sentiment"]==target, "review"]

    rows=[]
    for kw in keywords:
        pattern = rf"\b{re.escape(kw)}\b"
        like = reviews_target.str.contains(pattern, regex=True).mean()
        marg = reviews_all.str.contains(pattern, regex=True).mean()
        post = (like * prior) / marg if marg>0 else float("nan")
        rows.append({
            "Keyword": kw,
            f"Prior P({direction.title()})": prior,
            f"Likelihood P(keyword|{direction.title()})": like,
            "Marginal P(keyword)": marg,
            f"Posterior P({direction.title()}|keyword)": post,
            "Support count keyword": int(reviews_all.str.contains(pattern, regex=True).sum())
        })
    tab = pd.DataFrame(rows)
    post_col = [c for c in tab.columns if "Posterior" in c][0]
    return tab.sort_values(by=post_col, ascending=False).reset_index(drop=True)
