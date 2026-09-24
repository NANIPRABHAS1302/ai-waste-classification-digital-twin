"""
Extract bottleneck features once via frozen MobileNetV2 / EfficientNet-B0 (ImageNet weights).
This is standard feature-extraction transfer learning: a single forward pass through the
frozen backbone, cached to disk, so the classifier head can be trained fast on top of it.
"""
import os, time, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

DATA_DIR = "/home/claude/project/dataset"
CACHE_DIR = "/home/claude/project/features"
os.makedirs(CACHE_DIR, exist_ok=True)

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

backbone_name = sys.argv[1]  # "mobilenetv2" or "efficientnetb0"
BACKBONES = {"mobilenetv2": MobileNetV2, "efficientnetb0": EfficientNetB0}

def make_dataset(split):
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, split), labels="inferred", label_mode="int",
        class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

base = BACKBONES[backbone_name](input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet", pooling="avg")
base.trainable = False

for split in ["train", "val", "test"]:
    ds = make_dataset(split)
    t0 = time.time()
    feats, labels = [], []
    for x, y in ds:
        feats.append(base.predict(x, verbose=0))
        labels.append(y.numpy())
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    np.save(os.path.join(CACHE_DIR, f"{backbone_name}_{split}_X.npy"), feats)
    np.save(os.path.join(CACHE_DIR, f"{backbone_name}_{split}_y.npy"), labels)
    print(f"{backbone_name} {split}: {feats.shape} extracted in {time.time()-t0:.1f}s")

print(f"DONE extracting {backbone_name}")
