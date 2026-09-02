"""
Delhi Weather Forecasting
=========================
Forecasts Delhi's daily mean temperature 14 days ahead from four years of
daily weather observations (2013-2017), comparing an LSTM against a naive
seasonal baseline, and a univariate LSTM against a multivariate one that
also sees humidity, wind speed, and pressure.

Run with: python analysis.py
Figures are written to ./figures/, and summary numbers to results.json.
"""
import json
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.random import set_seed

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
np.random.seed(311)
set_seed(311)

ROOT = Path(__file__).parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
PALETTE = ["#2E5A87", "#5B9BD5", "#9DC3E6", "#C0392B"]

results = {}


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()


N_STEPS = 50
HORIZON = 14
FEATURES = ["meantemp", "humidity", "wind_speed", "meanpressure"]

# ---------------------------------------------------------------------------
# 1. Load & explore
# ---------------------------------------------------------------------------
df = pd.read_csv(ROOT / "data" / "delhi_weather.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.set_index("Date").sort_index()
results["n_rows"] = int(len(df))
results["missing_values"] = int(df.isnull().sum().sum())
results["date_range"] = [str(df.index.min().date()), str(df.index.max().date())]

plt.figure(figsize=(11, 4))
plt.plot(df.index, df["meantemp"], color=PALETTE[0], linewidth=0.8)
plt.title("Daily mean temperature, Delhi 2013-2017")
plt.ylabel("Mean temperature (C)")
savefig("01_daily_temperature.png")

monthly_by_year = df["meantemp"].resample("M").mean()
plt.figure(figsize=(11, 4))
for year in [2013, 2014, 2015, 2016, 2017]:
    yearly = monthly_by_year[monthly_by_year.index.year == year]
    if len(yearly):
        plt.plot(yearly.index.month, yearly.values, marker="o", label=str(year))
plt.xlabel("Month")
plt.ylabel("Mean temperature (C)")
plt.title("Monthly mean temperature by year")
plt.legend()
savefig("02_monthly_by_year.png")

yearly_avg = df["meantemp"].resample("Y").mean()
results["yearly_avg_temp"] = {str(i.year): round(float(v), 2) for i, v in yearly_avg.items()}

# ---------------------------------------------------------------------------
# 2. Train / test split (temporal, matching the original: 2013-2016 train, 2017 test)
# ---------------------------------------------------------------------------
train_df = df.loc["2013":"2016"]
test_df = df.loc["2017":]
results["n_train_days"] = int(len(train_df))
results["n_test_days"] = int(len(test_df))


def make_sequences(values, n_steps, horizon):
    X, y = [], []
    for i in range(len(values) - n_steps - horizon + 1):
        X.append(values[i:i + n_steps])
        y.append(values[i + n_steps:i + n_steps + horizon, 0])
    return np.array(X), np.array(y)


def evaluate(y_true, y_pred, label):
    mae_per_day = [mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(HORIZON)]
    rmse_per_day = [np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])) for i in range(HORIZON)]
    results[label] = {
        "mae_per_day": [round(float(m), 3) for m in mae_per_day],
        "rmse_per_day": [round(float(r), 3) for r in rmse_per_day],
        "mae_avg": round(float(np.mean(mae_per_day)), 3),
        "rmse_avg": round(float(np.mean(rmse_per_day)), 3),
    }
    return mae_per_day, rmse_per_day


# ---------------------------------------------------------------------------
# 3. Naive seasonal baseline: predict "same temperature as N_STEPS ago, repeated"
#    Concretely: tomorrow's forecast = today's observed value, held flat across
#    the horizon. This is the standard baseline a forecasting model has to beat.
# ---------------------------------------------------------------------------
train_temp = train_df["meantemp"].values.reshape(-1, 1)
test_temp = test_df["meantemp"].values.reshape(-1, 1)

X_test_raw, y_test_raw = make_sequences(test_temp, N_STEPS, HORIZON)
naive_pred = np.repeat(X_test_raw[:, -1, :], HORIZON, axis=1)
evaluate(y_test_raw, naive_pred, "naive_baseline")

# ---------------------------------------------------------------------------
# 4. Univariate LSTM (matches the original coursework approach, run with an
#    early-stopping validation split added so training doesn't overfit blindly
#    for a fixed 100 epochs)
# ---------------------------------------------------------------------------
sc_uni = MinMaxScaler(feature_range=(0, 1))
train_scaled_uni = sc_uni.fit_transform(train_temp)
test_scaled_uni = sc_uni.transform(test_temp)

X_train_uni, y_train_uni = make_sequences(train_scaled_uni, N_STEPS, HORIZON)
X_test_uni, y_test_uni = make_sequences(test_scaled_uni, N_STEPS, HORIZON)

lstm_uni = Sequential([
    LSTM(units=100, activation="tanh", input_shape=(N_STEPS, 1)),
    Dense(units=HORIZON),
])
lstm_uni.compile(optimizer=RMSprop(), loss="mse")
history_uni = lstm_uni.fit(
    X_train_uni, y_train_uni,
    validation_split=0.1,
    epochs=100, batch_size=32, verbose=0,
    callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
)
results["univariate_epochs_trained"] = len(history_uni.history["loss"])

pred_scaled_uni = lstm_uni.predict(X_test_uni, verbose=0)
pred_uni = sc_uni.inverse_transform(pred_scaled_uni)
y_test_uni_inv = sc_uni.inverse_transform(y_test_uni)
evaluate(y_test_uni_inv, pred_uni, "lstm_univariate")

# ---------------------------------------------------------------------------
# 5. Multivariate LSTM: same architecture, but the model also sees humidity,
#    wind speed, and pressure at each timestep, not just past temperature.
# ---------------------------------------------------------------------------
sc_multi = MinMaxScaler(feature_range=(0, 1))
train_scaled_multi = sc_multi.fit_transform(train_df[FEATURES].values)
test_scaled_multi = sc_multi.transform(test_df[FEATURES].values)


def make_sequences_multi(values, n_steps, horizon, target_col=0):
    X, y = [], []
    for i in range(len(values) - n_steps - horizon + 1):
        X.append(values[i:i + n_steps])
        y.append(values[i + n_steps:i + n_steps + horizon, target_col])
    return np.array(X), np.array(y)


X_train_multi, y_train_multi = make_sequences_multi(train_scaled_multi, N_STEPS, HORIZON)
X_test_multi, y_test_multi = make_sequences_multi(test_scaled_multi, N_STEPS, HORIZON)

lstm_multi = Sequential([
    LSTM(units=100, activation="tanh", input_shape=(N_STEPS, len(FEATURES))),
    Dense(units=HORIZON),
])
lstm_multi.compile(optimizer=RMSprop(), loss="mse")
history_multi = lstm_multi.fit(
    X_train_multi, y_train_multi,
    validation_split=0.1,
    epochs=100, batch_size=32, verbose=0,
    callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
)
results["multivariate_epochs_trained"] = len(history_multi.history["loss"])

pred_scaled_multi = lstm_multi.predict(X_test_multi, verbose=0)
# Inverse transform: only the temperature column was scaled per-feature, so
# reconstruct a full-width array to invert correctly, then slice temperature back out.
pad_pred = np.zeros((pred_scaled_multi.shape[0], HORIZON, len(FEATURES)))
pad_true = np.zeros((y_test_multi.shape[0], HORIZON, len(FEATURES)))
for h in range(HORIZON):
    pad_pred[:, h, 0] = pred_scaled_multi[:, h]
    pad_true[:, h, 0] = y_test_multi[:, h]
pred_multi = np.stack([sc_multi.inverse_transform(pad_pred[:, h, :])[:, 0] for h in range(HORIZON)], axis=1)
y_test_multi_inv = np.stack([sc_multi.inverse_transform(pad_true[:, h, :])[:, 0] for h in range(HORIZON)], axis=1)
evaluate(y_test_multi_inv, pred_multi, "lstm_multivariate")

# ---------------------------------------------------------------------------
# 6. Charts
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 4.5))
days = np.arange(1, HORIZON + 1)
plt.plot(days, results["naive_baseline"]["mae_per_day"], marker="o", label="Naive baseline", color=PALETTE[3])
plt.plot(days, results["lstm_univariate"]["mae_per_day"], marker="o", label="LSTM (temperature only)", color=PALETTE[1])
plt.plot(days, results["lstm_multivariate"]["mae_per_day"], marker="o", label="LSTM (+ humidity, wind, pressure)", color=PALETTE[0])
plt.xlabel("Forecast day ahead")
plt.ylabel("MAE (C)")
plt.title("Forecast error by horizon day")
plt.legend()
savefig("03_mae_by_horizon.png")

plt.figure(figsize=(8, 4.5))
labels = ["Naive baseline", "LSTM\n(temperature only)", "LSTM\n(+ weather features)"]
avgs = [results["naive_baseline"]["mae_avg"], results["lstm_univariate"]["mae_avg"], results["lstm_multivariate"]["mae_avg"]]
plt.bar(labels, avgs, color=[PALETTE[3], PALETTE[1], PALETTE[0]])
for i, v in enumerate(avgs):
    plt.text(i, v + 0.03, f"{v:.2f}", ha="center")
plt.ylabel("Average MAE across 14-day horizon (C)")
plt.title("Average forecast error by model")
savefig("04_model_comparison.png")

sample_index = 6
plt.figure(figsize=(8, 4.5))
history_x = np.arange(1, N_STEPS + 1)
future_x = np.arange(N_STEPS, N_STEPS + HORIZON)
plt.plot(history_x, sc_uni.inverse_transform(X_test_uni[sample_index]), label="Observed history", color="gray")
plt.plot(future_x, y_test_uni_inv[sample_index], label="Actual", color=PALETTE[3], marker="o", markersize=3)
plt.plot(future_x, pred_uni[sample_index], label="LSTM forecast (temperature only)", color=PALETTE[1], marker="o", markersize=3)
plt.plot(future_x, pred_multi[sample_index], label="LSTM forecast (+ weather features)", color=PALETTE[0], marker="o", markersize=3)
plt.title(f"Example 14-day forecast (test window {sample_index})")
plt.ylabel("Mean temperature (C)")
plt.xlabel("Day")
plt.legend(fontsize=8)
savefig("05_example_forecast.png")

with open(ROOT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print("\nDone. Figures in ./figures, full results in results.json")
