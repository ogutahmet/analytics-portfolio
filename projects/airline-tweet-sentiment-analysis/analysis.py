"""
Airline Tweet Sentiment & Topic Analysis
=========================================
Analyzes the Twitter US Airline Sentiment dataset (Crowdflower/Kaggle) to explore
airline popularity, geographic engagement, predictive sentiment models, lexicon-based
sentiment scoring, and topic modeling of customer complaints.

Run with: python analysis.py
Figures are written to ./figures/, and summary numbers are printed to stdout
(also captured in results.json for use in the writeup).
"""
import json
import re
import ssl
import warnings
from collections import Counter
from pathlib import Path

import certifi
import matplotlib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from gensim import corpora, models
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
warnings.filterwarnings("ignore")
matplotlib.use("Agg")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})

ROOT = Path(__file__).parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
PALETTE = ["#2E5A87", "#5B9BD5", "#9DC3E6", "#1F3B57", "#7FA8C9", "#3D6E9E"]

results = {}


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 1. Load & explore
# ---------------------------------------------------------------------------
df = pd.read_csv(ROOT / "data" / "tweets.csv")
results["n_rows"] = int(len(df))
results["n_cols"] = int(df.shape[1])
results["n_cells"] = int(df.size)

missing = df.isnull().sum().sort_values(ascending=False)
results["missing_top5"] = {k: int(v) for k, v in missing.head(5).items()}

plt.figure(figsize=(7, 4))
df["airline_sentiment"].value_counts().reindex(["negative", "neutral", "positive"]).plot(
    kind="bar", color=[PALETTE[0], PALETTE[2], PALETTE[1]]
)
plt.xlabel("Sentiment")
plt.ylabel("Number of tweets")
plt.title("Overall sentiment distribution")
plt.xticks(rotation=0)
savefig("01_sentiment_distribution.png")

# ---------------------------------------------------------------------------
# 2. Task A - airline popularity by tweet volume
# ---------------------------------------------------------------------------
popularity = df["airline"].value_counts().reset_index()
popularity.columns = ["airline", "tweets"]
results["airline_popularity"] = dict(zip(popularity["airline"], popularity["tweets"].astype(int)))

plt.figure(figsize=(7, 4))
plt.bar(popularity["airline"], popularity["tweets"], color=PALETTE[0])
for i, v in enumerate(popularity["tweets"]):
    plt.text(i, v + 30, str(v), ha="center", fontsize=9)
plt.xlabel("Airline")
plt.ylabel("Number of tweets")
plt.title("Tweet volume by airline")
plt.xticks(rotation=30, ha="right")
savefig("02_airline_popularity.png")

# ---------------------------------------------------------------------------
# 3. Task B - top states per airline (fixed: word-boundary match, not substring)
# ---------------------------------------------------------------------------
STATE_ABBR = {
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN",
    "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH",
    "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA",
    "VT", "WA", "WI", "WV", "WY",
}
STATE_NAME_TO_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH",
    "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN",
    "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}
CITY_TO_STATE = {
    "nyc": "NY", "new york city": "NY", "manhattan": "NY", "brooklyn": "NY",
    "los angeles": "CA", "san francisco": "CA", "san diego": "CA", "sf": "CA",
    "chicago": "IL", "boston": "MA", "houston": "TX", "dallas": "TX", "austin": "TX",
    "miami": "FL", "orlando": "FL", "atlanta": "GA", "philadelphia": "PA", "philly": "PA",
    "seattle": "WA", "denver": "CO", "phoenix": "AZ", "charlotte": "NC", "nashville": "TN",
    "washington dc": "DC", "dc": "DC",
}


def get_state(location: str):
    """Map a free-text tweet_location to a US state abbreviation.

    The original coursework version checked `state_abbr in tweet_location`,
    which matches substrings anywhere in the string (e.g. "IN" inside
    "Miniapolis", "OR" inside "Orlando"), silently mis-tagging locations.
    This version requires the abbreviation as a standalone token, and adds
    full state names and common city names so real locations are not lost.
    """
    if not isinstance(location, str) or not location.strip():
        return None
    text = location.strip().lower()
    tokens = re.findall(r"[a-z]+", text)
    for name, abbr in CITY_TO_STATE.items():
        if name in text:
            return abbr
    for name, abbr in STATE_NAME_TO_ABBR.items():
        if name in text:
            return abbr
    for tok in tokens:
        if tok.upper() in STATE_ABBR:
            return tok.upper()
    return None


df_loc = df.dropna(subset=["tweet_location"]).copy()
df_loc["state"] = df_loc["tweet_location"].apply(get_state)
matched_rate = df_loc["state"].notna().mean()
results["location_match_rate"] = round(float(matched_rate), 3)

state_popularity = (
    df_loc.dropna(subset=["state"])
    .groupby(["airline", "state"])
    .size()
    .reset_index(name="count")
    .sort_values(["airline", "count"], ascending=[True, False])
)
top_states = state_popularity.groupby("airline").head(5)
results["top_states_per_airline"] = {
    airline: list(zip(g["state"], g["count"].astype(int)))
    for airline, g in top_states.groupby("airline")
}

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, (airline, g) in zip(axes.flat, top_states.groupby("airline")):
    ax.bar(g["state"], g["count"], color=PALETTE[1])
    ax.set_title(airline, fontsize=10)
    ax.set_ylabel("Tweets")
fig.suptitle("Top 5 states by tweet volume, per airline")
savefig("03_top_states_per_airline.png")

# ---------------------------------------------------------------------------
# 4. Predictive sentiment models (fixed: proper 3-class labels, not
#    "positive vs. everything else")
# ---------------------------------------------------------------------------
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["airline_sentiment"].map(label_map)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
lr_model.fit(X_train_vec, y_train)
lr_pred = lr_model.predict(X_test_vec)
lr_report = classification_report(
    y_test, lr_pred, target_names=["negative", "neutral", "positive"], output_dict=True
)

nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_pred = nb_model.predict(X_test_vec)
nb_report = classification_report(
    y_test, nb_pred, target_names=["negative", "neutral", "positive"], output_dict=True
)

results["logistic_regression"] = lr_report
results["naive_bayes"] = nb_report

metrics_df = pd.DataFrame({
    "Logistic Regression": {k: lr_report[k]["f1-score"] for k in ["negative", "neutral", "positive"]},
    "Naive Bayes": {k: nb_report[k]["f1-score"] for k in ["negative", "neutral", "positive"]},
})
ax = metrics_df.plot(kind="bar", figsize=(7, 4), color=[PALETTE[0], PALETTE[2]])
ax.set_ylabel("F1 score")
ax.set_title("Per-class F1 score: logistic regression vs. naive Bayes")
plt.xticks(rotation=0)
savefig("04_model_comparison.png")

# ---------------------------------------------------------------------------
# 5. Lexicon-based sentiment (VADER) for the top 3 airlines
# ---------------------------------------------------------------------------
analyzer = SentimentIntensityAnalyzer()
df["compound"] = df["text"].apply(lambda t: analyzer.polarity_scores(t)["compound"])


def categorize(score):
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


df["vader_category"] = df["compound"].apply(categorize)
top3 = df["airline"].value_counts().head(3).index.tolist()

vader_summary = {}
for airline in top3:
    sub = df[df["airline"] == airline]
    counts = sub["vader_category"].value_counts(normalize=True)
    vader_summary[airline] = {k: round(float(counts.get(k, 0)), 3) for k in ["positive", "negative", "neutral"]}
results["vader_summary_top3"] = vader_summary

vs_df = pd.DataFrame(vader_summary).T[["positive", "negative", "neutral"]]
vs_df.plot(kind="bar", figsize=(7, 4), color=[PALETTE[1], PALETTE[0], PALETTE[2]])
plt.ylabel("Proportion of tweets")
plt.title("Lexicon-based sentiment proportions, top 3 airlines by volume")
plt.xticks(rotation=0)
savefig("05_vader_top3.png")

# ---------------------------------------------------------------------------
# 6. Topic modeling on negative tweets (nouns only, via LDA)
# ---------------------------------------------------------------------------
nltk.data.path  # ensure resources resolve
stop_words = set(stopwords.words("english"))
negative_tweets = df[df["airline_sentiment"] == "negative"]["text"]


def extract_nouns(text):
    tokens = word_tokenize(re.sub(r"http\S+|@\w+", "", text))
    tagged = pos_tag(tokens)
    return [w.lower() for w, pos in tagged if pos.startswith("NN") and w.lower() not in stop_words and w.isalpha()]


nouns_per_tweet = negative_tweets.apply(extract_nouns)
dictionary = corpora.Dictionary(nouns_per_tweet)
dictionary.filter_extremes(no_below=5, no_above=0.4)
corpus = [dictionary.doc2bow(n) for n in nouns_per_tweet]

NUM_TOPICS = 6
lda_model = models.LdaModel(
    corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15, random_state=42
)
topics = {}
for idx in range(NUM_TOPICS):
    words = [w for w, _ in lda_model.show_topic(idx, topn=8)]
    topics[f"topic_{idx + 1}"] = words
results["lda_topics"] = topics

all_nouns_text = " ".join(w for nouns in nouns_per_tweet for w in nouns)
wc = WordCloud(width=900, height=450, background_color="white", colormap="Blues").generate(all_nouns_text)
plt.figure(figsize=(9, 4.5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Most frequent nouns in negative tweets")
savefig("06_negative_wordcloud.png")

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, idx in zip(axes.flat, range(NUM_TOPICS)):
    freqs = dict(lda_model.show_topic(idx, topn=10))
    wc_topic = WordCloud(width=500, height=300, background_color="white", colormap="Blues").generate_from_frequencies(freqs)
    ax.imshow(wc_topic, interpolation="bilinear")
    ax.set_title(f"Topic {idx + 1}", fontsize=10)
    ax.axis("off")
savefig("07_topic_wordclouds.png")

negreason_top5 = df["negativereason"].value_counts().head(5)
results["negativereason_top5"] = {k: int(v) for k, v in negreason_top5.items()}

# ---------------------------------------------------------------------------
# Write results
# ---------------------------------------------------------------------------
with open(ROOT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2)[:3000])
print("\nDone. Figures in ./figures, full results in results.json")
