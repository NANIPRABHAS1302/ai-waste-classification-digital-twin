"""
Detection utilities for bounding box transformations and ROI extraction.

Designed for robust bounding box processing:
- Configurable padding
- Boundary clipping
- Safe handling of invalid, zero-size, or out-of-bounds bounding boxes
- Extraction of valid BGR regions of interest for downstream classifiers (MobileNetV2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class Detection:
    """
    Clean internal representation of a detected object.

    Attributes
    ----------
    class_id : int
        Class index (e.g. COCO class ID from detector).
    class_name : str
        Human-readable class label.
    confidence : float
        Detection confidence score in [0.0, 1.0].
    x1 : float
        Top-left X coordinate.
    y1 : float
        Top-left Y coordinate.
    x2 : float
        Bottom-right X coordinate.
    y2 : float
        Bottom-right Y coordinate.
    """
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Bounding box width."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Bounding box height."""
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Bounding box area."""
        return self.width * self.height

    @property
    def box_xyxy(self) -> Tuple[float, float, float, float]:
        """Returns tuple of (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)


def pad_and_clip_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int,
    frame_height: int,
    padding_fraction: float = 0.10,
    min_size: int = 4,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Applies configurable padding to bounding box coordinates and clips to image boundaries.

    Parameters
    ----------
    x1, y1, x2, y2 : float
        Original unclipped bounding box coordinates.
    frame_width : int
        Width of the parent frame in pixels.
    frame_height : int
        Height of the parent frame in pixels.
    padding_fraction : float, default 0.10 (10%)
        Fractional expansion applied to width and height around the box center.
    min_size : int, default 4
        Minimum width and height in pixels for an ROI to be considered valid.

    Returns
    -------
    Tuple[int, int, int, int] or None
        Clipped integer bounding box (x1, y1, x2, y2) within [0, frame_width]
        and [0, frame_height], or None if the box is invalid/degenerate.
    """
    if frame_width <= 0 or frame_height <= 0:
        return None

    # Check for NaN / Inf
    coords = (x1, y1, x2, y2)
    if any(not np.isfinite(c) for c in coords):
        return None

    # Sort coordinates to ensure x1 <= x2 and y1 <= y2
    bx1 = float(min(x1, x2))
    bx2 = float(max(x1, x2))
    by1 = float(min(y1, y2))
    by2 = float(max(y1, y2))

    orig_w = bx2 - bx1
    orig_h = by2 - by1

    # Reject degenerate boxes with non-positive dimensions
    if orig_w <= 0.0 or orig_h <= 0.0:
        return None

    pad_w = orig_w * max(0.0, padding_fraction)
    pad_h = orig_h * max(0.0, padding_fraction)

    px1 = bx1 - pad_w
    py1 = by1 - pad_h
    px2 = bx2 + pad_w
    py2 = by2 + pad_h

    # Clip to frame boundary
    cx1 = int(max(0, min(round(px1), frame_width)))
    cy1 = int(max(0, min(round(py1), frame_height)))
    cx2 = int(max(0, min(round(px2), frame_width)))
    cy2 = int(max(0, min(round(py2), frame_height)))

    clipped_w = cx2 - cx1
    clipped_h = cy2 - cy1

    if clipped_w < min_size or clipped_h < min_size:
        return None

    return (cx1, cy1, cx2, cy2)


def extract_roi(
    frame: np.ndarray,
    detection: Detection,
    padding_fraction: float = 0.10,
    min_size: int = 4,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Extracts a padded, clipped Region of Interest (ROI) from an OpenCV frame.

    Preserves BGR format and original pixel data without resizing.
    Downstream classifiers (e.g. MobileNetV2) perform their own required
    resizing (160x160), color conversion (BGR->RGB), and normalization.

    Parameters
    ----------
    frame : np.ndarray
        Source image frame in BGR format (height, width, channels).
    detection : Detection
        Detection containing the bounding box coordinates.
    padding_fraction : float, default 0.10
        Padding factor around bounding box.
    min_size : int, default 4
        Minimum pixel width/height required for valid ROI.

    Returns
    -------
    Tuple[np.ndarray, Tuple[int, int, int, int]] or None
        (roi_crop, (x1, y1, x2, y2)) where roi_crop is a sub-array of frame,
        or None if frame or detection box is invalid.
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return None

    if frame.ndim != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
        return None

    h, w = frame.shape[:2]
    coords = pad_and_clip_box(
        x1=detection.x1,
        y1=detection.y1,
        x2=detection.x2,
        y2=detection.y2,
        frame_width=w,
        frame_height=h,
        padding_fraction=padding_fraction,
        min_size=min_size,
    )

    if coords is None:
        return None

    x1, y1, x2, y2 = coords
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0 or roi.shape[0] < min_size or roi.shape[1] < min_size:
        return None

    return roi, (x1, y1, x2, y2)
