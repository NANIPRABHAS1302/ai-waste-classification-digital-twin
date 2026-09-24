"""
Real-time MobileNetV2 waste classifier module.
Performs image preprocessing, model loading, and six-class waste inference.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


DEFAULT_CLASSES: List[str] = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]


class WasteClassifier:
    """Reusable classifier using MobileNetV2 trained on the 6 waste categories."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/mobilenetv2.keras",
        input_size: Tuple[int, int] = (160, 160),
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.class_names = list(class_names) if class_names is not None else list(DEFAULT_CLASSES)
        self.num_classes = len(self.class_names)
        self.model: Optional[Any] = None

    def load_model(self) -> Any:
        """
        Loads the trained Keras model from disk.
        Does not retrain or alter weights.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path.resolve()}")

        try:
            import keras
            self.model = keras.models.load_model(str(self.model_path.resolve()), compile=False)
            return self.model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model from {self.model_path}: {exc}"
            ) from exc

    def preprocess(self, frame: np.ndarray, input_color: str = "BGR") -> np.ndarray:
        """
        Preprocesses a raw image frame for MobileNetV2 inference:
        1. Validates frame is not None and has 3 dimensions (H, W, 3).
        2. Converts color space (BGR -> RGB if input is BGR).
        3. Resizes to (input_size[0], input_size[1]) using bilinear interpolation.
        4. Casts to float32 and scales pixel values: pixel / 255.0.
        5. Adds batch dimension: (1, H, W, 3).
        """
        if frame is None:
            raise ValueError("Input frame cannot be None.")

        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Input frame must be a numpy.ndarray, got {type(frame).__name__}.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected frame with shape (height, width, 3), got shape {frame.shape}."
            )

        if frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError(f"Frame has invalid empty dimensions: {frame.shape}.")

        # Color conversion
        input_color_upper = input_color.upper()
        if input_color_upper == "BGR":
            # Convert BGR to RGB (flip channels)
            rgb_frame = frame[:, :, ::-1]
        elif input_color_upper == "RGB":
            rgb_frame = frame
        else:
            raise ValueError(f"Unsupported input_color '{input_color}'. Expected 'BGR' or 'RGB'.")

        target_h, target_w = self.input_size

        # Resize image using cv2 if available, or PIL, or pure NumPy bilinear fallback
        if rgb_frame.shape[0] == target_h and rgb_frame.shape[1] == target_w:
            resized = rgb_frame
        else:
            try:
                import cv2
                resized = cv2.resize(rgb_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            except ImportError:
                try:
                    from PIL import Image
                    pil_img = Image.fromarray(rgb_frame.astype(np.uint8))
                    resized = np.array(pil_img.resize((target_w, target_h), resample=Image.BILINEAR))
                except ImportError:
                    # Pure NumPy bilinear interpolation fallback (zero external dependencies)
                    src_h, src_w = rgb_frame.shape[:2]
                    scale_y = src_h / target_h
                    scale_x = src_w / target_w
                    y_indices = (np.arange(target_h) + 0.5) * scale_y - 0.5
                    x_indices = (np.arange(target_w) + 0.5) * scale_x - 0.5

                    y_indices = np.clip(y_indices, 0, src_h - 1)
                    x_indices = np.clip(x_indices, 0, src_w - 1)

                    y0 = np.floor(y_indices).astype(int)
                    y1 = np.clip(y0 + 1, 0, src_h - 1)
                    x0 = np.floor(x_indices).astype(int)
                    x1 = np.clip(x0 + 1, 0, src_w - 1)

                    dy = (y_indices - y0)[:, np.newaxis, np.newaxis]
                    dx = (x_indices - x0)[np.newaxis, :, np.newaxis]

                    top_left = rgb_frame[y0[:, np.newaxis], x0[np.newaxis, :]]
                    top_right = rgb_frame[y0[:, np.newaxis], x1[np.newaxis, :]]
                    bottom_left = rgb_frame[y1[:, np.newaxis], x0[np.newaxis, :]]
                    bottom_right = rgb_frame[y1[:, np.newaxis], x1[np.newaxis, :]]

                    top = top_left * (1.0 - dx) + top_right * dx
                    bottom = bottom_left * (1.0 - dx) + bottom_right * dx
                    resized = (top * (1.0 - dy) + bottom * dy).astype(rgb_frame.dtype)

        # Cast to float32 and normalize [0, 1] matching training (02_train_models.py: x / 255.0)
        norm_img = resized.astype(np.float32) / 255.0

        # Ensure values stay bounded in [0.0, 1.0]
        norm_img = np.clip(norm_img, 0.0, 1.0)

        # Add batch dimension -> (1, 160, 160, 3)
        batched = np.expand_dims(norm_img, axis=0)
        return batched

    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Runs inference on an input frame:
        1. Preprocesses the frame.
        2. Executes MobileNetV2 model prediction.
        3. Validates probability vector.
        4. Returns dictionary containing class_id, class_name, confidence, and probabilities.
        """
        if self.model is None:
            self.load_model()

        preprocessed = self.preprocess(frame, input_color="BGR")

        # Run model inference
        raw_output = self.model(preprocessed, training=False)
        if hasattr(raw_output, "numpy"):
            probs = raw_output.numpy().squeeze()
        else:
            probs = np.array(raw_output).squeeze()

        probs = np.asarray(probs, dtype=np.float32)

        # Validation: check shape
        if probs.ndim != 1 or len(probs) != self.num_classes:
            raise ValueError(
                f"Model output shape mismatch: expected {self.num_classes} classes, got shape {probs.shape}."
            )

        # Validation: check finite values
        if not np.all(np.isfinite(probs)):
            raise ValueError(f"Model output contains non-finite values (NaN or Inf): {probs}")

        # Validation: check sum approximately 1.0
        prob_sum = float(np.sum(probs))
        if not np.isclose(prob_sum, 1.0, atol=1e-2):
            raise ValueError(f"Probabilities do not sum to ~1.0: sum is {prob_sum:.4f}")

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        class_name = self.class_names[class_id]

        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "probabilities": probs.tolist(),
        }
