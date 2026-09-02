"""
Agricultural Pest Image Classification
=======================================
Classifies images of five agricultural pests (ants, bees, grasshoppers, moths,
wasps) using two convolutional networks trained from scratch, then a transfer
learning model built on a pretrained MobileNetV2, to see how much a small
image dataset (~2,500 images) benefits from starting with pretrained weights.

Run with: python analysis.py
Figures go to ./figures/, metrics to results.json.
"""
import json
import ssl
import warnings
from pathlib import Path

import certifi
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping

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
tf.random.set_seed(311)
np.random.seed(311)

ROOT = Path(__file__).parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR = ROOT / "data"
PALETTE = ["#2E5A87", "#5B9BD5", "#9DC3E6", "#1F3B57", "#C0392B"]
CLASSES = ["ants", "bees", "grasshopper", "moth", "wasp"]
IMG_SIZE = 128
MAX_EPOCHS = 100

results = {}


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 1. Load images
# ---------------------------------------------------------------------------
images, labels = [], []
counts = {}
for class_idx, cls in enumerate(CLASSES):
    files = [f for f in (DATA_DIR / cls).iterdir() if f.suffix.lower() in (".jpg", ".jpeg")]
    counts[cls] = len(files)
    for f in files:
        img = tf.io.read_file(str(f))
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img.numpy())
        labels.append(class_idx)

X = np.array(images, dtype="float32") / 255.0
y = np.array(labels)
results["class_counts"] = counts
results["n_images"] = int(len(y))

# Stratified split: 70% train, 15% validation, 15% test, so every split
# keeps roughly the same class balance as the full dataset. The original
# notebook used a single unstratified 80/20 shuffle split with no separate
# validation set held out in advance of training.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=311, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=311, stratify=y_temp
)
results["split_sizes"] = {"train": int(len(y_train)), "val": int(len(y_val)), "test": int(len(y_test))}

fig, axes = plt.subplots(5, 5, figsize=(9, 9))
rng = np.random.default_rng(311)
sample_idx = rng.choice(len(X_train), 25, replace=False)
for ax, idx in zip(axes.flat, sample_idx):
    ax.imshow(X_train[idx])
    ax.set_title(CLASSES[y_train[idx]], fontsize=9)
    ax.axis("off")
savefig("01_sample_images.png")

plt.figure(figsize=(6, 4))
plt.bar(counts.keys(), counts.values(), color=PALETTE[0])
plt.ylabel("Number of images")
plt.title("Images per pest class")
savefig("02_class_balance.png")


def evaluate_model(model, name, history=None):
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    report = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)
    kappa = cohen_kappa_score(y_test, y_pred)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    results[name] = {
        "test_accuracy": round(float(test_acc), 4),
        "cohen_kappa": round(float(kappa), 4),
        "per_class_f1": {c: round(report[c]["f1-score"], 3) for c in CLASSES},
        "epochs_trained": len(history.history["loss"]) if history else None,
    }
    cm = confusion_matrix(y_test, y_pred)
    return cm, y_pred


def plot_confusion(cm, title, filename):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    plt.yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    savefig(filename)


early_stop = lambda: EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

# ---------------------------------------------------------------------------
# 2. Model 1: simple CNN (matches original architecture)
# ---------------------------------------------------------------------------
model_1 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(CLASSES), activation="softmax"),
])
model_1.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_1 = model_1.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=MAX_EPOCHS, batch_size=32, verbose=0, callbacks=[early_stop()],
)
cm1, _ = evaluate_model(model_1, "model_1_simple_cnn", history_1)
print("Model 1 done:", results["model_1_simple_cnn"])

# ---------------------------------------------------------------------------
# 3. Model 2: deeper CNN with dropout (matches original architecture)
# ---------------------------------------------------------------------------
model_2 = models.Sequential([
    layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(CLASSES), activation="softmax"),
])
model_2.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_2 = model_2.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=MAX_EPOCHS, batch_size=32, verbose=0, callbacks=[early_stop()],
)
cm2, _ = evaluate_model(model_2, "model_2_deeper_cnn", history_2)
print("Model 2 done:", results["model_2_deeper_cnn"])

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history_1.history["accuracy"], label="Model 1 train")
plt.plot(history_1.history["val_accuracy"], label="Model 1 val")
plt.plot(history_2.history["accuracy"], label="Model 2 train")
plt.plot(history_2.history["val_accuracy"], label="Model 2 val")
plt.title("Training accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(fontsize=8)
plt.subplot(1, 2, 2)
plt.plot(history_1.history["loss"], label="Model 1 train")
plt.plot(history_1.history["val_loss"], label="Model 1 val")
plt.plot(history_2.history["loss"], label="Model 2 train")
plt.plot(history_2.history["val_loss"], label="Model 2 val")
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(fontsize=8)
savefig("03_training_curves_scratch_models.png")

plot_confusion(cm1, "Confusion matrix: Model 1 (simple CNN)", "04_confusion_model1.png")
plot_confusion(cm2, "Confusion matrix: Model 2 (deeper CNN)", "05_confusion_model2.png")

# ---------------------------------------------------------------------------
# 4. Model 3: transfer learning on MobileNetV2, with light augmentation.
#    With ~1,700 training images, a from-scratch CNN has very little to
#    learn general visual features from; a network pretrained on ImageNet
#    already knows edges, textures, and shapes, so only a small classifier
#    head needs training on top of it.
# ---------------------------------------------------------------------------
augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

for attempt in range(5):
    try:
        base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
        break
    except Exception as e:
        print(f"MobileNetV2 weights download attempt {attempt + 1} failed: {e}")
        if attempt == 4:
            raise
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = augmentation(inputs)
x = layers.Rescaling(scale=2.0, offset=-1.0)(x)  # MobileNetV2 expects [-1, 1], data is already [0, 1]
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(CLASSES), activation="softmax")(x)
model_3 = models.Model(inputs, outputs)
model_3.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history_3 = model_3.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=MAX_EPOCHS, batch_size=32, verbose=0, callbacks=[early_stop()],
)
cm3, _ = evaluate_model(model_3, "model_3_transfer_learning", history_3)

plot_confusion(cm3, "Confusion matrix: Model 3 (transfer learning)", "06_confusion_model3.png")

plt.figure(figsize=(7, 4))
names = ["Model 1\n(simple CNN)", "Model 2\n(deeper CNN)", "Model 3\n(transfer learning)"]
accs = [results["model_1_simple_cnn"]["test_accuracy"],
        results["model_2_deeper_cnn"]["test_accuracy"],
        results["model_3_transfer_learning"]["test_accuracy"]]
plt.bar(names, accs, color=[PALETTE[1], PALETTE[0], PALETTE[4]])
for i, v in enumerate(accs):
    plt.text(i, v + 0.01, f"{v:.1%}", ha="center")
plt.ylabel("Test accuracy")
plt.ylim(0, 1)
plt.title("Test accuracy by model")
savefig("07_model_comparison.png")

with open(ROOT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print("\nDone. Figures in ./figures, full results in results.json")
