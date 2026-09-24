"""
Unit tests for YOLO Detector and Detection Utilities.

Covers:
1. Detector initialization
2. Detection representation
3. No detection handling
4. Single detection parsing
5. Multiple detections parsing
6. Best-confidence selection
7. ROI padding calculations
8. ROI clipping at image boundaries
9. Invalid / inverted bounding boxes
10. Empty / degenerate ROI handling
11. Detector failure and graceful fallback handling
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.detection.detection_utils import Detection, extract_roi, pad_and_clip_box
from src.detection.yolo_detector import YOLODetector


class TestDetectionRepresentation(unittest.TestCase):
    """Tests for the Detection dataclass representation."""

    def test_01_detection_fields_and_properties(self):
        det = Detection(
            class_id=39,
            class_name="bottle",
            confidence=0.88,
            x1=100.0,
            y1=150.0,
            x2=200.0,
            y2=350.0,
        )
        self.assertEqual(det.class_id, 39)
        self.assertEqual(det.class_name, "bottle")
        self.assertAlmostEqual(det.confidence, 0.88)
        self.assertEqual(det.x1, 100.0)
        self.assertEqual(det.y1, 150.0)
        self.assertEqual(det.x2, 200.0)
        self.assertEqual(det.y2, 350.0)
        self.assertEqual(det.width, 100.0)
        self.assertEqual(det.height, 200.0)
        self.assertEqual(det.area, 20000.0)
        self.assertEqual(det.box_xyxy, (100.0, 150.0, 200.0, 350.0))

    def test_02_detection_frozen(self):
        det = Detection(0, "person", 0.9, 10, 10, 50, 50)
        with self.assertRaises(Exception):
            det.confidence = 0.95  # Dataclass is frozen


class TestRoiUtilities(unittest.TestCase):
    """Tests for pad_and_clip_box and extract_roi functions."""

    def test_01_pad_and_clip_normal(self):
        # 100x100 box in 640x480 frame with 10% padding
        # orig: [100, 100, 200, 200], pad_w = 10, pad_h = 10
        # expected: [90, 90, 210, 210]
        clipped = pad_and_clip_box(
            x1=100, y1=100, x2=200, y2=200,
            frame_width=640, frame_height=480,
            padding_fraction=0.10,
        )
        self.assertIsNotNone(clipped)
        self.assertEqual(clipped, (90, 90, 210, 210))

    def test_02_clipping_at_boundaries(self):
        # Box touching top-left: [5, 5, 50, 50], 10% pad -> [-0.5, -0.5, 54.5, 54.5]
        # Must clip at 0: [0, 0, 55, 55]
        clipped = pad_and_clip_box(
            x1=5, y1=5, x2=50, y2=50,
            frame_width=640, frame_height=480,
            padding_fraction=0.10,
        )
        self.assertIsNotNone(clipped)
        cx1, cy1, cx2, cy2 = clipped
        self.assertEqual(cx1, 0)
        self.assertEqual(cy1, 0)
        self.assertGreaterEqual(cx2, 50)
        self.assertGreaterEqual(cy2, 50)

        # Box near bottom-right
        clipped_br = pad_and_clip_box(
            x1=600, y1=450, x2=638, y2=478,
            frame_width=640, frame_height=480,
            padding_fraction=0.20,
        )
        self.assertIsNotNone(clipped_br)
        cx1, cy1, cx2, cy2 = clipped_br
        self.assertLessEqual(cx2, 640)
        self.assertLessEqual(cy2, 480)

    def test_03_invalid_bounding_box(self):
        # Zero width or height
        self.assertIsNone(pad_and_clip_box(100, 100, 100, 200, 640, 480))
        self.assertIsNone(pad_and_clip_box(100, 100, 200, 100, 640, 480))

        # NaN / Inf coordinates
        self.assertIsNone(pad_and_clip_box(float("nan"), 100, 200, 200, 640, 480))
        self.assertIsNone(pad_and_clip_box(100, float("inf"), 200, 200, 640, 480))

        # Invalid frame dimensions
        self.assertIsNone(pad_and_clip_box(10, 10, 50, 50, 0, 480))
        self.assertIsNone(pad_and_clip_box(10, 10, 50, 50, 640, -10))

    def test_04_inverted_coordinates_handled(self):
        # x1 > x2 and y1 > y2 should be normalized safely
        clipped = pad_and_clip_box(
            x1=200, y1=200, x2=100, y2=100,
            frame_width=640, frame_height=480,
            padding_fraction=0.10,
        )
        self.assertIsNotNone(clipped)
        self.assertEqual(clipped, (90, 90, 210, 210))

    def test_05_extract_roi_valid(self):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        det = Detection(class_id=1, class_name="cup", confidence=0.85, x1=50, y1=60, x2=150, y2=160)
        res = extract_roi(frame, det, padding_fraction=0.10)
        self.assertIsNotNone(res)
        roi, coords = res
        self.assertEqual(roi.ndim, 3)
        self.assertEqual(roi.shape[2], 3)
        self.assertEqual(roi.shape[0], coords[3] - coords[1])
        self.assertEqual(roi.shape[1], coords[2] - coords[0])

    def test_06_extract_roi_empty_or_invalid_frame(self):
        det = Detection(class_id=1, class_name="cup", confidence=0.85, x1=50, y1=60, x2=150, y2=160)
        self.assertIsNone(extract_roi(None, det))
        self.assertIsNone(extract_roi(np.zeros((0, 0, 3), dtype=np.uint8), det))
        self.assertIsNone(extract_roi(np.zeros((100, 100), dtype=np.uint8), det))  # ndim != 3


class TestYOLODetector(unittest.TestCase):
    """Tests for YOLODetector with mocked Ultralytics model."""

    def test_01_initialization(self):
        detector = YOLODetector(model_path="yolov8n.pt", confidence_threshold=0.35, device="cpu")
        self.assertEqual(detector.model_path, "yolov8n.pt")
        self.assertEqual(detector.confidence_threshold, 0.35)
        self.assertEqual(detector.device, "cpu")
        self.assertFalse(detector.is_loaded)

    def test_02_no_detection(self):
        detector = YOLODetector(confidence_threshold=0.50)
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_model.return_value = [mock_result]
        detector.model = mock_model
        detector._is_loaded = True

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        self.assertEqual(len(detections), 0)
        self.assertIsNone(detector.get_best_detection(frame))
        self.assertIsNone(detector.detect_and_crop(frame))

    def test_03_single_detection(self):
        detector = YOLODetector(confidence_threshold=0.40)
        mock_model = MagicMock()
        mock_model.names = {39: "bottle"}

        mock_box = MagicMock()
        mock_box.conf = [0.82]
        mock_box.cls = [39]
        mock_box.xyxy = [[100.0, 120.0, 250.0, 320.0]]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]

        detector.model = mock_model
        detector._is_loaded = True

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        self.assertEqual(len(detections), 1)
        det = detections[0]
        self.assertEqual(det.class_id, 39)
        self.assertEqual(det.class_name, "bottle")
        self.assertAlmostEqual(det.confidence, 0.82)
        self.assertEqual(det.x1, 100.0)

    def test_04_multiple_detections_and_best_selection(self):
        detector = YOLODetector(confidence_threshold=0.30)
        mock_model = MagicMock()
        mock_model.names = {0: "person", 39: "bottle", 41: "cup"}

        # Box 1: person (conf 0.45)
        box1 = MagicMock()
        box1.conf = [0.45]
        box1.cls = [0]
        box1.xyxy = [[10.0, 10.0, 100.0, 200.0]]

        # Box 2: bottle (conf 0.91) -> HIGHEST
        box2 = MagicMock()
        box2.conf = [0.91]
        box2.cls = [39]
        box2.xyxy = [[200.0, 200.0, 350.0, 400.0]]

        # Box 3: cup (conf 0.65)
        box3 = MagicMock()
        box3.conf = [0.65]
        box3.cls = [41]
        box3.xyxy = [[400.0, 300.0, 480.0, 380.0]]

        mock_result = MagicMock()
        mock_result.boxes = [box1, box2, box3]
        mock_model.return_value = [mock_result]

        detector.model = mock_model
        detector._is_loaded = True

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        self.assertEqual(len(detections), 3)

        best = detector.get_best_detection(frame)
        self.assertIsNotNone(best)
        self.assertEqual(best.class_name, "bottle")
        self.assertAlmostEqual(best.confidence, 0.91)

    def test_05_detect_and_crop(self):
        detector = YOLODetector(confidence_threshold=0.30)
        mock_model = MagicMock()
        mock_model.names = {39: "bottle"}

        box = MagicMock()
        box.conf = [0.85]
        box.cls = [39]
        box.xyxy = [[100.0, 100.0, 200.0, 200.0]]

        mock_result = MagicMock()
        mock_result.boxes = [box]
        mock_model.return_value = [mock_result]
        detector.model = mock_model
        detector._is_loaded = True

        frame = np.full((480, 640, 3), 75, dtype=np.uint8)
        crop_res = detector.detect_and_crop(frame, padding_fraction=0.10)
        self.assertIsNotNone(crop_res)
        det, roi, coords = crop_res
        self.assertEqual(det.class_name, "bottle")
        self.assertEqual(coords, (90, 90, 210, 210))
        self.assertEqual(roi.shape, (120, 120, 3))

    def test_06_inference_failure_handling(self):
        detector = YOLODetector()
        mock_model = MagicMock()
        mock_model.side_effect = RuntimeError("Inference kernel crashed")
        detector.model = mock_model
        detector._is_loaded = True

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Must catch exception safely and return empty list / None without crashing
        self.assertEqual(detector.detect(frame), [])
        self.assertIsNone(detector.get_best_detection(frame))
        self.assertIsNone(detector.detect_and_crop(frame))

    def test_07_invalid_frame_inputs(self):
        detector = YOLODetector()
        detector.model = MagicMock()
        detector._is_loaded = True

        self.assertEqual(detector.detect(None), [])
        self.assertEqual(detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)), [])
        self.assertEqual(detector.detect("not_an_array"), [])


if __name__ == "__main__":
    unittest.main()
