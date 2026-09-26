"""
Customer Churn: Logistic Regression vs. RFM
=============================================
Tesco Clubcard case study. 20,000 customers with a full year of purchase
history are used to build a churn model, evaluated against 10,000 customers
held out for testing, and compared against a simple RFM (recency, frequency,
monetary) ranking and a random baseline using a lift (gains) chart.

Run with: python analysis.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

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
PALETTE = ["#2E5A87", "#5B9BD5", "#9DC3E6", "#C0392B"]

results = {}

train = pd.read_csv(ROOT / "data" / "training.csv")
test = pd.read_csv(ROOT / "data" / "test.csv")
results["n_train"] = int(len(train))
results["n_test"] = int(len(test))
results["train_churn_rate"] = round(float(train["churn"].mean()), 4)
results["test_churn_rate"] = round(float(test["churn"].mean()), 4)

FEATURES = [c for c in train.columns if c not in ("ID", "churn")]

# Four of the 10,000 test customers are missing SocioEconomic; training has no
# missing values at all. Filling with the training median is a small, explicit
# choice rather than letting a handful of blanks silently break the model.
n_missing_test = int(test["SocioEconomic"].isnull().sum())
results["test_rows_missing_socioeconomic"] = n_missing_test
test["SocioEconomic"] = test["SocioEconomic"].fillna(train["SocioEconomic"].median())

# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------
X_train, y_train = train[FEATURES], train["churn"]
X_test, y_test = test[FEATURES], test["churn"]

# Features span very different scales (Purchase in single digits, TotalProfit
# in hundreds), which slows convergence and makes coefficients hard to compare
# directly; standardising puts every feature on the same footing for both.
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)

pred_prob = model.predict_proba(X_test_scaled)[:, 1]
pred_class = (pred_prob >= 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, pred_class).ravel()
accuracy = accuracy_score(y_test, pred_class)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

results["logistic_regression"] = {
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "accuracy": round(float(accuracy), 4),
    "sensitivity": round(float(sensitivity), 4),
    "specificity": round(float(specificity), 4),
    "coefficients": {f: round(float(c), 4) for f, c in zip(FEATURES, model.coef_[0])},
    "intercept": round(float(model.intercept_[0]), 4),
}

plt.figure(figsize=(5, 4.5))
cm = np.array([[tn, fp], [fn, tp]])
plt.imshow(cm, cmap="Blues")
plt.xticks([0, 1], ["Predicted: stay", "Predicted: churn"])
plt.yticks([0, 1], ["Actual: stay", "Actual: churn"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
plt.title(f"Confusion matrix (accuracy {accuracy:.1%})")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# RFM scoring on the test set
# ---------------------------------------------------------------------------
rfm = test.copy()
rfm["recency"] = rfm["T.active"] - rfm["T.last"]  # time since last purchase, to end of window
rfm["frequency"] = rfm["Purchase"]
rfm["monetary"] = rfm["TotalProfit"]

# Quintile 1 = best (most recent, most frequent, highest spend); 5 = worst.
rfm["r_score"] = pd.qcut(rfm["recency"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]  # lower = more valuable/engaged customer

# ---------------------------------------------------------------------------
# Lift (gains) chart: logistic regression, RFM, random
# ---------------------------------------------------------------------------
n = len(test)
n_deciles = 10
total_churners = y_test.sum()

order_lr = np.argsort(-pred_prob)  # highest predicted churn risk first
# RFM is normally used to rank customers by best engagement first; for identifying
# who is likely to churn, that ordering needs to be reversed, since the whole
# premise of RFM is that disengaged customers (low recency, frequency, and
# spend) are the ones actually at risk. Highest rfm_score = worst RFM = highest
# assumed churn risk, first.
order_rfm = np.argsort(-rfm["rfm_score"].values)

churn_arr = y_test.values


def cumulative_capture(order, churn_arr, n_deciles):
    n = len(order)
    sorted_churn = churn_arr[order]
    capture = []
    for d in range(1, n_deciles + 1):
        cutoff = int(round(n * d / n_deciles))
        capture.append(sorted_churn[:cutoff].sum() / churn_arr.sum() * 100)
    return capture


lr_capture = cumulative_capture(order_lr, churn_arr, n_deciles)
rfm_capture = cumulative_capture(order_rfm, churn_arr, n_deciles)
random_capture = [100 * d / n_deciles for d in range(1, n_deciles + 1)]

results["lift_chart"] = {
    "decile": list(range(1, n_deciles + 1)),
    "logistic_regression_pct_churners_captured": [round(v, 2) for v in lr_capture],
    "rfm_pct_churners_captured": [round(v, 2) for v in rfm_capture],
    "random_pct_churners_captured": [round(v, 2) for v in random_capture],
}

plt.figure(figsize=(8, 5.5))
deciles = list(range(0, n_deciles + 1))
plt.plot(deciles, [0] + lr_capture, marker="o", label="Logistic regression", color=PALETTE[0])
plt.plot(deciles, [0] + rfm_capture, marker="o", label="RFM", color=PALETTE[1])
plt.plot(deciles, [0] + random_capture, linestyle="--", label="Random", color="gray")
plt.xlabel("Decile of customers contacted (ranked highest risk / RFM priority first)")
plt.ylabel("% of actual churners captured")
plt.title("Lift chart: logistic regression vs. RFM vs. random")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "02_lift_chart.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# The alignment defect found in the original spreadsheet, reproduced and
# quantified. The defect is subtler than X and y being decoupled: each
# working row's predictors and churn label are both pulled from the *same*
# source customer, just mislabeled as belonging to the row below. That means
# the fitted model still learns from 19,999 genuinely matched (X, y) pairs,
# it's simply missing one customer (replaced by a row of header text) rather
# than training on scrambled data. Reproducing that exactly, one customer
# dropped, checks how much that actually mattered.
# ---------------------------------------------------------------------------
X_one_short = X_train_scaled[1:]
y_one_short = y_train.iloc[1:].reset_index(drop=True)
model_one_short = LogisticRegression(max_iter=5000)
model_one_short.fit(X_one_short, y_one_short)
pred_prob_one_short = model_one_short.predict_proba(X_test_scaled)[:, 1]
pred_class_one_short = (pred_prob_one_short >= 0.5).astype(int)
acc_one_short = accuracy_score(y_test, pred_class_one_short)
results["one_customer_dropped_replication"] = {
    "description": "Same model, trained on 19,999 of the 20,000 customers (the actual scale of the original's alignment defect), to check whether it mattered",
    "accuracy": round(float(acc_one_short), 4),
    "accuracy_difference_vs_full_data": round(float(accuracy - acc_one_short), 4),
}

# For contrast: what training on fully decoupled (scrambled) X/y pairs would
# do, to show what a *genuinely* broken alignment looks like rather than the
# narrower defect actually present in the original.
X_decoupled = X_train_scaled[:-1]
y_decoupled = y_train.iloc[1:].reset_index(drop=True)
model_decoupled = LogisticRegression(max_iter=5000)
model_decoupled.fit(X_decoupled, y_decoupled)
pred_prob_decoupled = model_decoupled.predict_proba(X_test_scaled)[:, 1]
pred_class_decoupled = (pred_prob_decoupled >= 0.5).astype(int)
acc_decoupled = accuracy_score(y_test, pred_class_decoupled)
results["fully_decoupled_hypothetical"] = {
    "description": "Hypothetical worse case: every row's X paired with a different customer's y, for contrast with what the original's actual defect looked like",
    "accuracy": round(float(acc_decoupled), 4),
}

with open(ROOT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2)[:3000])
print("\nDone. Figures in ./figures, full results in results.json")
