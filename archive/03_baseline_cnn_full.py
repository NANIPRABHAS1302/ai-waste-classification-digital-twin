import os, json, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

tf.random.set_seed(42)
np.random.seed(42)

DATA_DIR = "/home/claude/project/dataset"
OUT_DIR = "/home/claude/project/results"
MODEL_DIR = "/home/claude/project/models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 15
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def make_ds(split, shuffle):
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, split), labels="inferred", label_mode="int",
        class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=shuffle, seed=42,
    )
    return ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

train_ds = make_ds("train", True)
val_ds = make_ds("val", False)
test_ds = make_ds("test", False)

train_labels = np.concatenate([y.numpy() for _, y in make_ds("train", False).unbatch().batch(1000)])
cw = compute_class_weight("balanced", classes=np.arange(6), y=train_labels)
class_weight_dict = {i: w for i, w in enumerate(cw)}
print("Class weights:", {CLASS_NAMES[i]: round(w, 3) for i, w in class_weight_dict.items()})

model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(6, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

t0 = time.time()
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=class_weight_dict, callbacks=[es], verbose=2)
train_time = time.time() - t0

model.save(os.path.join(MODEL_DIR, "baseline_cnn.keras"))
model_size_mb = os.path.getsize(os.path.join(MODEL_DIR, "baseline_cnn.keras")) / (1024*1024)

# Inference time
sample = next(iter(test_ds))[0][:1]
model.predict(sample, verbose=0)
t1 = time.time()
for _ in range(30):
    model.predict(sample, verbose=0)
inf_ms = (time.time()-t1)/30*1000

# Evaluation
y_true, y_pred = [], []
for x, y in test_ds:
    preds = model.predict(x, verbose=0)
    y_true.extend(y.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
cm = confusion_matrix(y_true, y_pred).tolist()

summary = {
    "model": "baseline_cnn",
    "img_size": IMG_SIZE,
    "epochs_trained": len(history.history["loss"]),
    "train_time_sec": round(train_time, 1),
    "model_size_mb": round(model_size_mb, 3),
    "inference_time_ms": round(inf_ms, 2),
    "test_accuracy": round(report["accuracy"], 4),
    "macro_f1": round(report["macro avg"]["f1-score"], 4),
    "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
    "per_class": {cls: {"precision": round(report[cls]["precision"],3), "recall": round(report[cls]["recall"],3), "f1": round(report[cls]["f1-score"],3), "support": report[cls]["support"]} for cls in CLASS_NAMES},
    "confusion_matrix": cm,
}

with open(os.path.join(OUT_DIR, "baseline_cnn_results.json"), "w") as f:
    json.dump(summary, f, indent=2)
with open(os.path.join(OUT_DIR, "baseline_cnn_history.json"), "w") as f:
    json.dump(history.history, f)

print(json.dumps(summary, indent=2))
