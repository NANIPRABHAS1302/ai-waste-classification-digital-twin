
"""
02b_train_efficientnet.py

Experiment 2: Improved EfficientNet-B0 Training
"""

import os
import json
import tensorflow as tf
import numpy as np

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

tf.random.set_seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = (160,160)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 5

CLASS_NAMES = [
    "cardboard","glass","metal",
    "paper","plastic","trash"
]

def make_dataset(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, split),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES,
        label_mode="int",
        shuffle=shuffle,
        seed=42,
    )

train_raw = make_dataset("train", True)
val_raw = make_dataset("val", False)

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

AUTOTUNE = tf.data.AUTOTUNE

def prepare(ds, augment=False):
    def process(x, y):
        x = tf.cast(x, tf.float32)
        if augment:
            x = augmentation(x, training=True)
        x = preprocess_input(x)
        return x, y
    return ds.map(process, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

train_ds = prepare(train_raw, True)
val_ds = prepare(val_raw, False)

labels=[]
for _,y in train_raw.unbatch():
    labels.append(int(y.numpy()))
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASS_NAMES)),
    y=labels
)
class_weight = {i:float(w) for i,w in enumerate(weights)}

base = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMG_SIZE,3)
)
base.trainable=False

inputs = layers.Input(shape=(*IMG_SIZE,3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

cbs = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2
    ),
    callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR,"efficientnetb0.keras"),
        save_best_only=True,
        monitor="val_accuracy"
    )
]

print("\n========== STAGE 1 ==========\n")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weight,
    callbacks=cbs,
    verbose=2
)

print("\n========== STAGE 2 ==========\n")

base.trainable=True

for layer in base.layers[:-20]:
    layer.trainable=False

model.compile(
    optimizer=optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weight,
    callbacks=cbs,
    verbose=2
)

history = {}
for k,v in history1.history.items():
    history[k] = v + history2.history.get(k,[])

with open(
    os.path.join(
        RESULTS_DIR,
        "history_efficientnetb0.json"
    ),
    "w"
) as f:
    json.dump(history,f,indent=4)

print("\nTraining complete.")
print("Model saved to:", os.path.join(MODEL_DIR,"efficientnetb0.keras"))
print("History saved.")
