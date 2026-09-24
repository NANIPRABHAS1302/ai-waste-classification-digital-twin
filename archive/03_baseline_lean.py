import os, json, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

tf.random.set_seed(42)
np.random.seed(42)

DATA_DIR = "/home/claude/project/dataset"
MODEL_DIR = "/home/claude/project/models"
os.makedirs(MODEL_DIR, exist_ok=True)
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 6
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Known train counts from the stratified split (avoids re-scanning the dataset for weights)
train_counts = {"cardboard": 282, "glass": 350, "metal": 287, "paper": 415, "plastic": 337, "trash": 95}
total = sum(train_counts.values())
class_weight_dict = {i: total / (6 * train_counts[c]) for i, c in enumerate(CLASS_NAMES)}
print("Class weights:", {CLASS_NAMES[i]: round(w, 3) for i, w in class_weight_dict.items()})

def make_ds(split, shuffle):
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, split), labels="inferred", label_mode="int",
        class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=shuffle, seed=42,
    )
    return ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

train_ds = make_ds("train", True)
val_ds = make_ds("val", False)

model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Conv2D(16, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(6, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

ckpt = tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "baseline_cnn.keras"), save_best_only=False)

t0 = time.time()
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=class_weight_dict, callbacks=[ckpt], verbose=2)
train_time = time.time() - t0

with open("/home/claude/project/results_train_time.json", "w") as f:
    json.dump({"train_time_sec": train_time, "epochs": EPOCHS}, f)
with open("/home/claude/project/results_history.json", "w") as f:
    json.dump(history.history, f)

print("TRAIN_TIME:", train_time)
