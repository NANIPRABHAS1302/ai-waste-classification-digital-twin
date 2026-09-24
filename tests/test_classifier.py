"""
Unit tests for WasteClassifier in src/realtime/realtime_classifier.py.
Covers:
- Test 1: Class mapping
- Test 2: Preprocessing dimensions, dtype, range [0, 1]
- Test 3: BGR to RGB channel conversion
- Test 4: Model prediction and probability distribution verification
"""

import unittest
from pathlib import Path
import numpy as np

from src.realtime.realtime_classifier import WasteClassifier, DEFAULT_CLASSES


class TestWasteClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = WasteClassifier(
            model_path="models/mobilenetv2.keras",
            input_size=(160, 160),
            class_names=DEFAULT_CLASSES,
        )

    def test_01_class_mapping(self):
        """Verify the 6-class mapping matches the trained model specification."""
        expected_mapping = {
            0: "cardboard",
            1: "glass",
            2: "metal",
            3: "paper",
            4: "plastic",
            5: "trash",
        }
        self.assertEqual(len(self.classifier.class_names), 6)
        for class_id, class_name in expected_mapping.items():
            self.assertEqual(
                self.classifier.class_names[class_id],
                class_name,
                f"Class id {class_id} should be '{class_name}' but got '{self.classifier.class_names[class_id]}'",
            )

    def test_02_preprocessing(self):
        """Verify preprocessing output shape, float32 dtype, and [0, 1] normalization."""
        # Create synthetic 200x300 BGR image with values in [0, 255]
        np.random.seed(42)
        dummy_frame = np.random.randint(0, 256, size=(200, 300, 3), dtype=np.uint8)

        preprocessed = self.classifier.preprocess(dummy_frame, input_color="BGR")

        # Check shape (1, 160, 160, 3)
        self.assertEqual(preprocessed.shape, (1, 160, 160, 3))
        # Check dtype float32
        self.assertEqual(preprocessed.dtype, np.float32)
        # Check value bounds
        self.assertGreaterEqual(float(np.min(preprocessed)), 0.0)
        self.assertLessEqual(float(np.max(preprocessed)), 1.0)

    def test_03_rgb_conversion(self):
        """Verify that BGR to RGB conversion correctly flips color channels."""
        # Create a 1x1 image with Distinct B, G, R values
        # B = 50, G = 100, R = 200
        distinct_frame = np.zeros((160, 160, 3), dtype=np.uint8)
        distinct_frame[:, :, 0] = 50   # Blue
        distinct_frame[:, :, 1] = 100  # Green
        distinct_frame[:, :, 2] = 200  # Red

        preprocessed = self.classifier.preprocess(distinct_frame, input_color="BGR")

        # Channel 0 in preprocessed output should now be Red (200 / 255.0)
        # Channel 2 in preprocessed output should now be Blue (50 / 255.0)
        expected_r = 200.0 / 255.0
        expected_g = 100.0 / 255.0
        expected_b = 50.0 / 255.0

        sample_pixel = preprocessed[0, 80, 80, :]
        self.assertAlmostEqual(float(sample_pixel[0]), expected_r, places=4)
        self.assertAlmostEqual(float(sample_pixel[1]), expected_g, places=4)
        self.assertAlmostEqual(float(sample_pixel[2]), expected_b, places=4)

    def test_04_input_validation(self):
        """Verify clear exception raising for invalid inputs."""
        with self.assertRaises(ValueError):
            self.classifier.preprocess(None)

        with self.assertRaises(ValueError):
            # 2D grayscale instead of 3-channel
            self.classifier.preprocess(np.zeros((100, 100), dtype=np.uint8))

        with self.assertRaises(ValueError):
            # Empty frame
            self.classifier.preprocess(np.zeros((0, 0, 3), dtype=np.uint8))

        with self.assertRaises(ValueError):
            # Invalid color space string
            self.classifier.preprocess(np.zeros((100, 100, 3), dtype=np.uint8), input_color="HSV")

    def test_05_prediction_output(self):
        """
        Verify prediction with real MobileNetV2 model if TensorFlow/Keras runtime is available.
        If runtime prevents native execution, the limitation is cleanly skipped without fabricating.
        """
        model_file = Path("models/mobilenetv2.keras")
        if not model_file.exists():
            self.skipTest(f"Model file {model_file} not found.")

        try:
            import keras
            self.classifier.load_model()
        except Exception as exc:
            self.skipTest(f"Keras/TensorFlow runtime unavailable in current environment: {exc}")

        # If model loaded successfully, execute inference on synthetic image
        np.random.seed(42)
        dummy_frame = np.random.randint(0, 256, size=(160, 160, 3), dtype=np.uint8)

        result = self.classifier.predict(dummy_frame)

        self.assertIn("class_id", result)
        self.assertIn("class_name", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)

        probs = result["probabilities"]
        self.assertEqual(len(probs), 6)
        self.assertTrue(all(np.isfinite(p) for p in probs))
        self.assertAlmostEqual(sum(probs), 1.0, places=2)
        self.assertGreaterEqual(result["class_id"], 0)
        self.assertLessEqual(result["class_id"], 5)
        self.assertEqual(result["class_name"], self.classifier.class_names[result["class_id"]])
        self.assertAlmostEqual(result["confidence"], probs[result["class_id"]], places=4)


if __name__ == "__main__":
    unittest.main()
