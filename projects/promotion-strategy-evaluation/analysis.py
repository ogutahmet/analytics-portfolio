"""
Promotion Strategy Evaluation
=============================
Carman's Kitchen, a cereal bar brand with about 20% category share, trialled
four price promotion strategies in four comparable stores over 26 weeks, and
needs a recommendation on which one to roll out.

This evaluates the net profit impact of each strategy (incremental sales
during the promotion, minus foregone contribution on baseline volume sold at
a lower price, minus the post promotion dip, minus any retailer cost), plus
each strategy's separately observed effect over the long run on brand sales,
and compares that against the original coursework version's numbers.

Run with: python analysis.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

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
STRATEGIES = [1, 2, 3, 4]
STRATEGY_LABELS = {
    1: "S1: 10% off",
    2: "S2: 20% off",
    3: "S3: 10% off + display",
    4: "S4: 20% off + catalogue",
}

# ---------------------------------------------------------------------------
# Case study inputs, taken directly from the assignment brief.
# ---------------------------------------------------------------------------
REGULAR_PRICE = 5.0
REGULAR_MARGIN = 3.0
PROMO_PRICE = {1: 4.5, 2: 4.0, 3: 4.5, 4: 4.0}
PROMO_MARGIN = {1: 2.5, 2: 2.0, 3: 2.5, 4: 2.0}  # price minus the $2 unit cost implied by regular price/margin
WEEKLY_RETAILER_COST = {1: 0, 2: 0, 3: 300, 4: 1000}
PROMO_WEEKS = 4
POST_PROMO_WEEKS = 2

# Averages over the long run, given directly in the case data (observed over
# the 10 weeks following the post-promotion dip, not derivable from the
# 16 weeks of data used above).
LONG_TERM_CATEGORY_SALES = {1: 9764, 2: 10999, 3: 12973, 4: 13037}
LONG_TERM_BRAND_SALES = {1: 994, 2: 1320, 3: 1299, 4: 1554}

df = pd.read_csv(ROOT / "data" / "weekly_data.csv")
for s in STRATEGIES:
    df[f"strategy{s}_brand_sales"] = df[f"strategy{s}_category_sales"] * df[f"strategy{s}_market_share"]

results = {}

# ---------------------------------------------------------------------------
# Weekly time series, all four strategies
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
for i, s in enumerate(STRATEGIES):
    plt.plot(df["week"], df[f"strategy{s}_brand_sales"], marker="o", markersize=3, label=STRATEGY_LABELS[s], color=PALETTE[i])
plt.axvspan(10.5, 14.5, color="gray", alpha=0.15, label="Promotion (weeks 11-14)")
plt.axvspan(14.5, 16.5, color="gray", alpha=0.3, label="Post promotion dip (weeks 15-16)")
plt.xlabel("Week")
plt.ylabel("Brand sales (units/week)")
plt.title("Weekly brand sales by strategy")
plt.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_weekly_sales.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Per-strategy summary figures: baseline, promotion, and post promotion sales
# ---------------------------------------------------------------------------
summary = {}
for s in STRATEGIES:
    brand_col = f"strategy{s}_brand_sales"
    baseline = df.loc[df["week"].between(1, 10), brand_col].mean()
    promo = df.loc[df["week"].between(11, 14), brand_col].mean()
    post = df.loc[df["week"].between(15, 16), brand_col].mean()
    incremental_sales = promo - baseline
    post_promo_dip = post - baseline

    # The original coursework version valued incremental sales and the
    # post promotion dip at the promotional/regular *price* ($4.50, $4, $5),
    # not the promotional/regular *margin* ($2.50, $2, $3). Since a $2 unit
    # cost sits behind every one of these prices, valuing volume at price
    # overstates the actual profit impact by exactly that unit cost.
    incremental_gain_original_bug = incremental_sales * PROMO_PRICE[s] * PROMO_WEEKS
    post_promo_loss_original_bug = post_promo_dip * REGULAR_PRICE * POST_PROMO_WEEKS

    incremental_gain = incremental_sales * PROMO_MARGIN[s] * PROMO_WEEKS
    post_promo_loss = post_promo_dip * REGULAR_MARGIN * POST_PROMO_WEEKS
    foregone_contribution = (PROMO_PRICE[s] - REGULAR_PRICE) * baseline * PROMO_WEEKS
    retailer_cost = -WEEKLY_RETAILER_COST[s] * PROMO_WEEKS

    net_effect = incremental_gain + foregone_contribution + post_promo_loss + retailer_cost
    net_effect_original_bug = incremental_gain_original_bug + foregone_contribution + post_promo_loss_original_bug + retailer_cost

    long_term_brand_sales_gain = (LONG_TERM_BRAND_SALES[s] - baseline) * 10
    long_term_brand_profit_gain = long_term_brand_sales_gain * REGULAR_MARGIN

    baseline_category_sales = df.loc[df["week"].between(1, 10), f"strategy{s}_category_sales"].mean()
    baseline_share = baseline / baseline_category_sales
    long_term_share = LONG_TERM_BRAND_SALES[s] / LONG_TERM_CATEGORY_SALES[s]

    # The original labeled this row "% Gain in category sales," but its formula,
    # (brand sales over the long run / category sales over the long run) minus
    # (baseline brand sales / baseline category sales), actually computes the
    # change in the *brand's share* of the category, not category sales growth.
    # Both are useful, since the brief asks about both, so both are reported
    # here under their correct names.
    pct_gain_category_sales = (LONG_TERM_CATEGORY_SALES[s] / baseline_category_sales) - 1
    brand_share_change_pp = (long_term_share - baseline_share) * 100

    total_value = net_effect + long_term_brand_profit_gain
    total_value_original_bug = net_effect_original_bug + long_term_brand_profit_gain

    summary[s] = {
        "baseline_brand_sales": round(baseline, 1),
        "promotion_brand_sales": round(promo, 1),
        "post_promotion_brand_sales": round(post, 1),
        "incremental_sales_per_week": round(incremental_sales, 1),
        "post_promotion_dip_per_week": round(post_promo_dip, 1),
        "incremental_gain_corrected": round(incremental_gain, 2),
        "incremental_gain_original_bug": round(incremental_gain_original_bug, 2),
        "post_promo_loss_corrected": round(post_promo_loss, 2),
        "post_promo_loss_original_bug": round(post_promo_loss_original_bug, 2),
        "foregone_contribution": round(foregone_contribution, 2),
        "retailer_cost": round(retailer_cost, 2),
        "net_effect_of_promotion_corrected": round(net_effect, 2),
        "net_effect_of_promotion_original_bug": round(net_effect_original_bug, 2),
        "long_term_brand_profit_gain": round(long_term_brand_profit_gain, 2),
        "pct_gain_category_sales": round(pct_gain_category_sales * 100, 2),
        "brand_market_share_change_pp": round(brand_share_change_pp, 2),
        "total_value_corrected": round(total_value, 2),
        "total_value_original_bug": round(total_value_original_bug, 2),
    }

results["summary"] = summary

with open(ROOT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------------------
# Net effect of promotion: original (buggy) vs. corrected
# ---------------------------------------------------------------------------
labels = [STRATEGY_LABELS[s] for s in STRATEGIES]
original_vals = [summary[s]["net_effect_of_promotion_original_bug"] for s in STRATEGIES]
corrected_vals = [summary[s]["net_effect_of_promotion_corrected"] for s in STRATEGIES]

x = range(len(STRATEGIES))
width = 0.35
plt.figure(figsize=(8, 5))
plt.bar([i - width / 2 for i in x], original_vals, width, label="Original (valued at price)", color=PALETTE[2])
plt.bar([i + width / 2 for i in x], corrected_vals, width, label="Corrected (valued at margin)", color=PALETTE[0])
plt.axhline(0, color="black", linewidth=0.8)
plt.xticks(list(x), labels, rotation=15)
plt.ylabel("Net effect of promotion ($/store)")
plt.title("Promotion profit in the short term: original vs. corrected")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "02_net_effect_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Total value (the promotion's own net effect plus the long-run brand profit gain), corrected
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
net_effects = [summary[s]["net_effect_of_promotion_corrected"] for s in STRATEGIES]
long_term = [summary[s]["long_term_brand_profit_gain"] for s in STRATEGIES]
plt.bar(labels, net_effects, label="Net effect of the promotion itself", color=PALETTE[0])
plt.bar(labels, long_term, bottom=net_effects, label="Brand profit gain over the long run", color=PALETTE[1])
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("Value ($/store)")
plt.title("Total value by strategy, corrected calculations")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "03_total_value.png", dpi=150, bbox_inches="tight")
plt.close()

print(json.dumps(summary, indent=2))
print("\nDone. Figures in ./figures, full results in results.json")
