"""
YOLOv8 Object Detector Wrapper.

Provides clean modular inference for object localization and presence detection
using Ultralytics YOLO models (e.g. YOLOv8n pretrained on COCO).

Designed to decouple detection from classification:
- Accepts OpenCV BGR frames
- Safely handles inference failures and empty detections
- Returns list of Detection instances
- Identifies highest-confidence (best) detection
- Does not hardcode waste classification labels
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import numpy as np

from src.detection.detection_utils import Detection, extract_roi

logger = logging.getLogger(__name__)


class YOLODetector:
    """
    Object detector wrapper around Ultralytics YOLO models.

    Parameters
    ----------
    model_path : str or Path, default "yolov8n.pt"
        Path to YOLO weights file or model name.
    confidence_threshold : float, default 0.25
        Minimum confidence required to retain a detection.
    device : str, default "cpu"
        Device to run inference on ('cpu', 'cuda', etc.).
    warmup : bool, default True
        If True, run one silent dummy inference immediately after model
        loading to pay the one-time PyTorch CPU JIT compilation cost
        (~1-2 s on first call) during startup rather than on the first
        live frame.  The dummy frame is a synthetic black image; no
        webcam is required and no sorting decision, UDP message, or
        telemetry event is generated.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        device: str = "cpu",
        warmup: bool = True,
    ) -> None:
        self.model_path = str(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.device = device
        self.warmup = warmup
        self.model: Optional[Any] = None
        self._is_loaded = False
        self._warmup_done = False

    def load_model(self) -> Any:
        """
        Loads the YOLO model into memory.

        If ``warmup=True`` (the default), a single silent inference is
        executed on a synthetic black frame immediately after loading.
        This pays the one-time PyTorch CPU JIT compilation cost during
        startup so the first live frame sees warm-path latency only.

        Returns
        -------
        The loaded YOLO model instance.
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self._is_loaded = True
        except Exception as exc:
            self.model = None
            self._is_loaded = False
            logger.error("Failed to load YOLO model from %s: %s", self.model_path, exc)
            raise RuntimeError(f"Failed to load YOLO model from {self.model_path}: {exc}") from exc

        if self.warmup:
            self._run_warmup()

        return self.model

    def _run_warmup(self) -> None:
        """
        Runs one silent dummy inference to trigger PyTorch JIT kernel
        compilation.  Uses a synthetic black frame (no webcam needed).
        Results are discarded.  Called once by load_model() when
        warmup=True; never called again for subsequent frames.
        """
        if self._warmup_done or self.model is None:
            return
        try:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model(dummy, verbose=False)
            self._warmup_done = True
            logger.info(
                "YOLODetector warm-up inference completed for model '%s'.",
                self.model_path,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "YOLODetector warm-up inference failed (non-fatal): %s", exc
            )

    @property
    def is_loaded(self) -> bool:
        """Whether the model is successfully loaded."""
        return self._is_loaded and self.model is not None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Runs object detection on a BGR image frame.

        Parameters
        ----------
        frame : np.ndarray
            Input image in OpenCV BGR format (H, W, 3).

        Returns
        -------
        List[Detection]
            List of detected objects satisfying confidence_threshold.
            Returns empty list if no objects found, invalid frame, or on inference failure.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            return []

        if frame.ndim != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
            return []

        if not self.is_loaded:
            try:
                self.load_model()
            except Exception as exc:
                logger.error("YOLODetector auto-load failed: %s", exc)
                return []

        try:
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            logger.error("YOLO inference failed: %s", exc)
            return []

        if not results:
            return []

        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(self.model, "names", {})
        detections: List[Detection] = []

        for box in boxes:
            try:
                # Extract confidence
                raw_conf = box.conf[0]
                conf = float(raw_conf.item() if hasattr(raw_conf, "item") else raw_conf)
                if conf < self.confidence_threshold:
                    continue

                # Extract class id and name
                raw_cls = box.cls[0]
                cls_id = int(raw_cls.item() if hasattr(raw_cls, "item") else raw_cls)
                cls_name = names.get(cls_id, str(cls_id))

                # Extract bounding box coordinates [x1, y1, x2, y2]
                raw_xyxy = box.xyxy[0]
                if hasattr(raw_xyxy, "tolist"):
                    coords = raw_xyxy.tolist()
                elif hasattr(raw_xyxy, "__iter__"):
                    coords = list(raw_xyxy)
                else:
                    coords = [float(c) for c in raw_xyxy]

                det = Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    x1=float(coords[0]),
                    y1=float(coords[1]),
                    x2=float(coords[2]),
                    y2=float(coords[3]),
                )
                detections.append(det)
            except Exception as exc:
                logger.warning("Error parsing YOLO box: %s", exc)
                continue

        return detections

    def get_best_detection(self, frame: np.ndarray) -> Optional[Detection]:
        """
        Detects objects and returns the single detection with the highest confidence.

        Parameters
        ----------
        frame : np.ndarray
            Input image in OpenCV BGR format.

        Returns
        -------
        Detection or None
            The detection with highest confidence, or None if no objects detected.
        """
        detections = self.detect(frame)
        if not detections:
            return None
        return max(detections, key=lambda d: d.confidence)

    def detect_and_crop(
        self,
        frame: np.ndarray,
        padding_fraction: float = 0.10,
        min_size: int = 4,
    ) -> Optional[Tuple[Detection, np.ndarray, Tuple[int, int, int, int]]]:
        """
        Convenience method: detects objects, selects the best detection,
        and extracts the padded, clipped ROI.

        Parameters
        ----------
        frame : np.ndarray
            Input BGR frame.
        padding_fraction : float, default 0.10
            Padding around bounding box.
        min_size : int, default 4
            Minimum width/height for valid ROI.

        Returns
        -------
        Tuple[Detection, np.ndarray, Tuple[int, int, int, int]] or None
            (best_detection, roi_crop, (x1, y1, x2, y2)), or None if no object detected
            or if ROI extraction failed.
        """
        best_det = self.get_best_detection(frame)
        if best_det is None:
            return None

        roi_res = extract_roi(
            frame=frame,
            detection=best_det,
            padding_fraction=padding_fraction,
            min_size=min_size,
        )
        if roi_res is None:
            return None

        roi_crop, box_coords = roi_res
        return best_det, roi_crop, box_coords
