"""
Phase 2: Train baseline CNN, MobileNetV2 (transfer learning), EfficientNet-B0 (transfer learning).
CPU-only run. Class weights applied to offset TrashNet's imbalance (trash: 95 train images
vs paper: 415). Fixed seed for reproducibility.
"""
import os, json, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from sklearn.utils.class_weight import compute_class_weight

tf.random.set_seed(42)
np.random.seed(42)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "dataset")
OUT_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 15
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def make_dataset(split, shuffle):
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, split),
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=42,
    )
    return ds

train_ds_raw = make_dataset("train", shuffle=True)
val_ds_raw = make_dataset("val", shuffle=False)
test_ds_raw = make_dataset("test", shuffle=False)

# Compute class weights from training labels
train_labels = []
for _, y in train_ds_raw.unbatch():
    train_labels.append(int(y.numpy()))
class_weights_arr = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=train_labels)
class_weight_dict = {i: w for i, w in enumerate(class_weights_arr)}
print("Class weights:", {CLASS_NAMES[i]: round(w, 3) for i, w in class_weight_dict.items()})

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

AUTOTUNE = tf.data.AUTOTUNE

def prep(ds, augment):
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

train_ds = prep(train_ds_raw, augment=True)
val_ds = prep(val_ds_raw, augment=False)
test_ds = prep(test_ds_raw, augment=False)

def build_baseline_cnn():
    m = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ])
    return m

def build_transfer_model(base_class, name):
    base = base_class(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    m = models.Model(inputs, outputs, name=name)
    return m

MODEL_BUILDERS = {
    "baseline_cnn": build_baseline_cnn,
    "mobilenetv2": lambda: build_transfer_model(MobileNetV2, "mobilenetv2"),
    "efficientnetb0": lambda: build_transfer_model(EfficientNetB0, "efficientnetb0"),
}

results_summary = {}

for name, builder in MODEL_BUILDERS.items():
    print(f"\n{'='*60}\nTraining: {name}\n{'='*60}")
    tf.keras.backend.clear_session()
    model = builder()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ]

    start = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS,
        class_weight=class_weight_dict, callbacks=callbacks, verbose=2,
    )
    train_time = time.time() - start

    model.save(os.path.join(MODEL_DIR, f"{name}.keras"))
    model_size_mb = os.path.getsize(os.path.join(MODEL_DIR, f"{name}.keras")) / (1024 * 1024)

    # Inference time: average over test set, single-image batches, CPU
    sample_batch = next(iter(test_ds))[0][:1]
    _ = model.predict(sample_batch, verbose=0)  # warmup
    n_trials = 30
    t0 = time.time()
    for _ in range(n_trials):
        model.predict(sample_batch, verbose=0)
    inf_time_ms = (time.time() - t0) / n_trials * 1000

    with open(os.path.join(OUT_DIR, f"history_{name}.json"), "w") as f:
        json.dump(history.history, f)

    results_summary[name] = {
        "train_time_sec": round(train_time, 1),
        "model_size_mb": round(model_size_mb, 2),
        "inference_time_ms": round(inf_time_ms, 2),
        "epochs_trained": len(history.history["loss"]),
    }
    print(f"{name} -> train_time={train_time:.1f}s, size={model_size_mb:.2f}MB, inference={inf_time_ms:.2f}ms")

with open(os.path.join(OUT_DIR, "training_summary.json"), "w") as f:
    json.dump(results_summary, f, indent=2)

print("\n\nDONE. Summary:")
print(json.dumps(results_summary, indent=2))
