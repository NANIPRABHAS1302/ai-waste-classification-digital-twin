"""
Run this AFTER 02a_extract_features.py has cached features for a backbone
(mobilenetv2 or efficientnetb0). Trains a small dense head on the cached
features, then rebuilds the full backbone+head model to get a realistic
model size and inference time for the deployable pipeline.

Usage:
    python 02a_extract_features.py mobilenetv2
    python 02b_train_head.py mobilenetv2

    python 02a_extract_features.py efficientnetb0
    python 02b_train_head.py efficientnetb0
"""
import os, json, time, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix

tf.random.set_seed(42)
np.random.seed(42)

backbone_name = sys.argv[1]
BACKBONES = {"mobilenetv2": MobileNetV2, "efficientnetb0": EfficientNetB0}
IMG_SIZE = (160, 160)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

CACHE_DIR = "/home/claude/project/features"
RESULTS_DIR = "/home/claude/project/results"
MODEL_DIR = "/home/claude/project/models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load(split):
    X = np.load(os.path.join(CACHE_DIR, f"{backbone_name}_{split}_X.npy"))
    y = np.load(os.path.join(CACHE_DIR, f"{backbone_name}_{split}_y.npy"))
    return X, y

X_train, y_train = load("train")
X_val, y_val = load("val")
X_test, y_test = load("test")

# Class weights (same TrashNet train-split counts as the baseline CNN run)
train_counts = {"cardboard": 282, "glass": 350, "metal": 287, "paper": 415, "plastic": 337, "trash": 95}
total = sum(train_counts.values())
class_weight_dict = {i: total / (6 * train_counts[c]) for i, c in enumerate(CLASS_NAMES)}

feat_dim = X_train.shape[1]
head = models.Sequential([
    layers.Input(shape=(feat_dim,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(6, activation="softmax"),
])
head.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
t0 = time.time()
history = head.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50,
                    class_weight=class_weight_dict, callbacks=[es], verbose=2, batch_size=32)
head_train_time = time.time() - t0

# Rebuild full deployable model: frozen backbone + trained head
base = BACKBONES[backbone_name](input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet", pooling="avg")
base.trainable = False
inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = base(inputs, training=False)
outputs = head(x)
full_model = models.Model(inputs, outputs, name=backbone_name)
full_model.save(os.path.join(MODEL_DIR, f"{backbone_name}.keras"))
model_size_mb = os.path.getsize(os.path.join(MODEL_DIR, f"{backbone_name}.keras")) / (1024*1024)

# Inference time on the FULL pipeline (backbone + head), matching real deployment
dummy = tf.random.uniform((1, *IMG_SIZE, 3))
full_model.predict(dummy, verbose=0)
t1 = time.time()
for _ in range(20):
    full_model.predict(dummy, verbose=0)
inf_ms = (time.time() - t1) / 20 * 1000

# Evaluate on test features (head only — equivalent to full model on real images)
preds = np.argmax(head.predict(X_test, verbose=0), axis=1)
report = classification_report(y_test, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
cm = confusion_matrix(y_test, preds).tolist()

summary = {
    "model": backbone_name,
    "head_train_time_sec": round(head_train_time, 1),
    "epochs_trained": len(history.history["loss"]),
    "model_size_mb": round(model_size_mb, 3),
    "inference_time_ms": round(inf_ms, 2),
    "test_accuracy": round(report["accuracy"], 4),
    "macro_f1": round(report["macro avg"]["f1-score"], 4),
    "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
    "per_class": {cls: {"precision": round(report[cls]["precision"],3), "recall": round(report[cls]["recall"],3), "f1": round(report[cls]["f1-score"],3), "support": int(report[cls]["support"])} for cls in CLASS_NAMES},
    "confusion_matrix": cm,
}
with open(os.path.join(RESULTS_DIR, f"{backbone_name}_results.json"), "w") as f:
    json.dump(summary, f, indent=2)
with open(os.path.join(RESULTS_DIR, f"{backbone_name}_history.json"), "w") as f:
    json.dump(history.history, f)

print(json.dumps(summary, indent=2))
