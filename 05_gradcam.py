"""
Phase 5
Grad-CAM Visualization

Creates publication-quality Grad-CAM visualizations
for EfficientNet-B0.

Expected input:
    models/efficientnetb0.keras
    dataset/test/<class_name>/*

Outputs:
    results/gradcam/
        cardboard_gradcam.png
        glass_gradcam.png
        metal_gradcam.png
        paper_gradcam.png
        plastic_gradcam.png
        trash_gradcam.png
        GradCAM_Summary.json
"""

import os
import json
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "efficientnetb0.keras"
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "test"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "gradcam"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

IMG_SIZE = (160, 160)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("GRAD-CAM GENERATION")
print("=" * 60)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "\nEfficientNet-B0 model not found.\n"
        f"Expected location:\n{MODEL_PATH}"
    )


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_DIR):

    raise FileNotFoundError(
        "\nTest dataset not found.\n"
        f"Expected location:\n{DATASET_DIR}\n\n"
        "Recover the test dataset before running Grad-CAM."
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

print("\nModel Summary:")
model.summary()


# ============================================================
# LOCATE EFFICIENTNET BACKBONE
# ============================================================

print("\nLocating EfficientNet backbone...")

backbone = None

for layer in model.layers:

    if isinstance(layer, tf.keras.Model):

        backbone = layer
        break


if backbone is None:

    raise RuntimeError(
        "Could not locate EfficientNet backbone."
    )


print(
    "Backbone found:",
    backbone.name
)


# ============================================================
# LOCATE LAST CONVOLUTION LAYER
# ============================================================

print("\nLocating last convolution layer...")

last_conv_layer = None

for layer in reversed(backbone.layers):

    if isinstance(
        layer,
        tf.keras.layers.Conv2D
    ):

        last_conv_layer = layer
        break


if last_conv_layer is None:

    raise RuntimeError(
        "No Conv2D layer found inside EfficientNet backbone."
    )


print(
    "Last Conv2D layer:",
    last_conv_layer.name
)


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL INFORMATION")
print("=" * 60)

print(
    "Model Name      :",
    model.name
)

print(
    "Backbone        :",
    backbone.name
)

print(
    "Grad-CAM Layer  :",
    last_conv_layer.name
)

print("=" * 60)


# ============================================================
# BUILD CONNECTED GRAD-CAM MODEL
# ============================================================

print("\nBuilding connected Grad-CAM model...")

target_layer = last_conv_layer

grad_model = tf.keras.models.Model(
    inputs=backbone.input,
    outputs=[
        target_layer.output,
        backbone.output
    ]
)


# ============================================================
# FIND CLASSIFIER LAYERS
# ============================================================

backbone_index = None

for i, layer in enumerate(model.layers):

    if layer.name == backbone.name:

        backbone_index = i
        break


if backbone_index is None:

    raise RuntimeError(
        "Could not locate EfficientNet backbone "
        "inside the main model."
    )


classifier_layers = model.layers[
    backbone_index + 1:
]


print("\nClassifier layers:")

for layer in classifier_layers:

    print(
        "  -",
        layer.name
    )


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def make_gradcam(
    image_tensor,
    target_class=None
):

    """
    Generate Grad-CAM heatmap for one image.
    """

    image_tensor = tf.cast(
        image_tensor,
        tf.float32
    )

    with tf.GradientTape() as tape:

        conv_output, backbone_output = grad_model(
            image_tensor,
            training=False
        )

        x = backbone_output

        for layer in classifier_layers:

            x = layer(
                x,
                training=False
            )

        predictions = x

        if target_class is None:

            target_class = tf.argmax(
                predictions[0]
            )

        class_score = predictions[
            :,
            target_class
        ]

    gradients = tape.gradient(
        class_score,
        conv_output
    )

    if gradients is None:

        raise RuntimeError(
            "Gradients are None. "
            "The Grad-CAM graph is disconnected."
        )

    # Global average pooling of gradients
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_output = conv_output[0]

    pooled_gradients = pooled_gradients[0]

    # Weighted combination of feature maps
    heatmap = tf.reduce_sum(
        conv_output * pooled_gradients,
        axis=-1
    )

    # Remove negative values
    heatmap = tf.maximum(
        heatmap,
        0
    )

    maximum = tf.reduce_max(
        heatmap
    )

    # Normalize safely
    heatmap = tf.where(
        maximum > 0,
        heatmap / maximum,
        heatmap
    )

    return (
        heatmap.numpy(),
        predictions.numpy()[0],
        int(target_class)
    )


# ============================================================
# FIND REPRESENTATIVE TEST IMAGES
# ============================================================

print("\n" + "=" * 60)
print("SEARCHING FOR TEST IMAGES")
print("=" * 60)

representative_images = {}


for class_name in CLASS_NAMES:

    class_dir = os.path.join(
        DATASET_DIR,
        class_name
    )

    if not os.path.isdir(class_dir):

        print(
            f"WARNING: Missing class directory: "
            f"{class_dir}"
        )

        continue


    image_files = []


    for filename in os.listdir(class_dir):

        if filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        ):

            image_files.append(
                os.path.join(
                    class_dir,
                    filename
                )
            )


    image_files.sort()


    if len(image_files) == 0:

        print(
            f"WARNING: No images found for "
            f"{class_name}"
        )

        continue


    # First deterministic image
    representative_images[
        class_name
    ] = image_files[0]

    print(
        f"{class_name}:",
        image_files[0]
    )


# ============================================================
# VERIFY REPRESENTATIVE IMAGES
# ============================================================

if len(representative_images) == 0:

    raise RuntimeError(
        "No test images were found."
    )


print(
    "\nRepresentative images found:",
    len(representative_images)
)


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

summary = []


print("\n" + "=" * 60)
print("GENERATING GRAD-CAM VISUALIZATIONS")
print("=" * 60)


for class_name, image_path in representative_images.items():

    print(
        f"\nProcessing: {class_name}"
    )


    # --------------------------------------------------------
    # Load original image
    # --------------------------------------------------------

    original = cv2.imread(
        image_path
    )


    if original is None:

        print(
            "WARNING: Could not read:",
            image_path
        )

        continue


    original_rgb = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Resize for model
    # --------------------------------------------------------

    resized = cv2.resize(
        original_rgb,
        IMG_SIZE
    )


    image_array = (
        resized.astype(
            np.float32
        ) / 255.0
    )


    image_tensor = np.expand_dims(
        image_array,
        axis=0
    )


    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    heatmap, predictions, predicted_class = (
        make_gradcam(
            image_tensor
        )
    )


    predicted_name = CLASS_NAMES[
        predicted_class
    ]


    confidence = float(
        predictions[
            predicted_class
        ]
    )


    # --------------------------------------------------------
    # Resize heatmap
    # --------------------------------------------------------

    heatmap_resized = cv2.resize(
        heatmap,
        (
            original_rgb.shape[1],
            original_rgb.shape[0]
        )
    )


    heatmap_uint8 = np.uint8(
        255 * heatmap_resized
    )


    # --------------------------------------------------------
    # Color heatmap
    # --------------------------------------------------------

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )


    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        original_rgb,
        0.60,
        colored_heatmap,
        0.40,
        0
    )


    # ========================================================
    # SAVE PUBLICATION FIGURE
    # ========================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{class_name}_gradcam.png"
    )


    fig = plt.figure(
        figsize=(12, 4)
    )


    # Original
    ax1 = fig.add_subplot(
        1,
        3,
        1
    )

    ax1.imshow(
        original_rgb
    )

    ax1.set_title(
        f"Original\n{class_name}"
    )

    ax1.axis("off")


    # Heatmap
    ax2 = fig.add_subplot(
        1,
        3,
        2
    )

    ax2.imshow(
        heatmap_resized,
        cmap="jet"
    )

    ax2.set_title(
        "Grad-CAM"
    )

    ax2.axis("off")


    # Overlay
    ax3 = fig.add_subplot(
        1,
        3,
        3
    )

    ax3.imshow(
        overlay
    )

    ax3.set_title(
        f"Prediction: {predicted_name}\n"
        f"Confidence: {confidence:.3f}"
    )

    ax3.axis("off")


    fig.tight_layout()


    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close(fig)


    print(
        "Saved:",
        output_path
    )


    # ========================================================
    # SAVE SUMMARY ENTRY
    # ========================================================

    summary.append(
        {
            "true_class": class_name,
            "image": image_path,
            "predicted_class": predicted_name,
            "confidence": round(
                confidence,
                4
            ),
            "output": output_path,
            "gradcam_layer": target_layer.name
        }
    )


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "GradCAM_Summary.json"
)


with open(
    summary_path,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("GRAD-CAM COMPLETE")
print("=" * 60)

print(
    "Images generated:",
    len(summary)
)

print(
    "Summary:",
    summary_path
)

print("=" * 60)