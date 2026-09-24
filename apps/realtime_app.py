"""
Real-time Waste Classification & Decision Application.

Orchestrates:
1. Video capture from webcam (with graceful failure fallback)
2. YOLOv8n object detection (background rejection / object localization)
3. Real-time inference using MobileNetV2 (models/mobilenetv2.keras)
4. Confidence & entropy reliability evaluation (ConfidenceGate)
5. Human-in-the-loop gesture supervisor / fallback
6. UDP JSON dispatch to the digital twin simulator (Port 5005)
7. Telemetry logging (results/telemetry/)

Pipeline:
    Webcam
    → YOLOv8n  (object localization / background rejection)
        NO object:  frame silently skipped — no MobileNet, no UDP, no telemetry
        YES object: best detection chosen, padded ROI cropped
    → MobileNetV2  (six-class waste classification: cardboard, glass, metal, paper, plastic, trash)
    → ConfidenceGate
    → Gesture supervisor / fallback
    → UDP dispatch
    → Telemetry
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.communication.udp_sender import UdpSender, build_message
from src.reliability.confidence_gate import ConfidenceGate
from src.telemetry.telemetry_logger import TelemetryLogger, TelemetryRecord

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False

try:
    from src.realtime.realtime_classifier import WasteClassifier
    CLASSIFIER_AVAILABLE = True
except ImportError:
    WasteClassifier = None  # type: ignore
    CLASSIFIER_AVAILABLE = False

try:
    from src.gesture.gesture_supervisor import GestureSupervisor
    GESTURE_AVAILABLE = True
except ImportError:
    GestureSupervisor = None  # type: ignore
    GESTURE_AVAILABLE = False

try:
    from src.detection.yolo_detector import YOLODetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLODetector = None  # type: ignore
    YOLO_AVAILABLE = False


# ---------------------------------------------------------------------------
# YOLO Configuration defaults
# Values can be overridden via constructor or CLI flags.
# ---------------------------------------------------------------------------
YOLO_DEFAULT_MODEL: str = "yolov8n.pt"
YOLO_DEFAULT_CONFIDENCE_THRESHOLD: float = 0.25
YOLO_DEFAULT_ROI_PADDING: float = 0.10


class RealtimeApp:
    """
    Coordinates webcam capture, YOLO object detection, MobileNetV2 classification,
    reliability gating, UDP communication, and telemetry logging.

    YOLO acts as a background-rejection layer:
    - If no object is detected in a frame, the frame is silently skipped.
    - Only frames with a detected object are forwarded to MobileNetV2.
    - "No object detected" is a frame-level skip, NOT a waste classification result.
    """

    def __init__(
        self,
        camera_index: int = 0,
        model_path: str = "models/mobilenetv2.keras",
        confidence_threshold: float = 0.70,
        entropy_threshold: float = 1.20,
        udp_host: str = "127.0.0.1",
        udp_port: int = 5005,
        telemetry_csv: Optional[str] = "results/telemetry/telemetry.csv",
        telemetry_jsonl: Optional[str] = None,
        warmup: bool = True,
        yolo_enabled: bool = True,
        yolo_model: str = YOLO_DEFAULT_MODEL,
        yolo_confidence_threshold: float = YOLO_DEFAULT_CONFIDENCE_THRESHOLD,
        yolo_roi_padding: float = YOLO_DEFAULT_ROI_PADDING,
    ) -> None:
        self.camera_index = camera_index
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.udp_host = udp_host
        self.udp_port = udp_port
        self.yolo_enabled = yolo_enabled and YOLO_AVAILABLE
        self.yolo_model_name = yolo_model
        self.yolo_confidence_threshold = yolo_confidence_threshold
        self.yolo_roi_padding = yolo_roi_padding

        # Initialize subcomponents
        if CLASSIFIER_AVAILABLE:
            self.classifier = WasteClassifier(model_path=self.model_path)
        else:
            self.classifier = None

        self.gate = ConfidenceGate(
            confidence_threshold=self.confidence_threshold,
            entropy_threshold=self.entropy_threshold,
        )
        self.sender = UdpSender(host=self.udp_host, port=self.udp_port)
        self.logger = TelemetryLogger(csv_path=telemetry_csv, jsonl_path=telemetry_jsonl)

        # Human supervisor
        if GESTURE_AVAILABLE:
            self.supervisor = GestureSupervisor()
        else:
            self.supervisor = None

        # YOLO detector
        self.detector: Optional[Any] = None
        if self.yolo_enabled:
            self.detector = YOLODetector(
                model_path=self.yolo_model_name,
                confidence_threshold=self.yolo_confidence_threshold,
                device="cpu",
            )

        self.cap = None
        self._warmup_done = False
        if warmup and (self.classifier is not None or (self.yolo_enabled and self.detector is not None)):
            self.warmup()

    def warmup(self) -> None:
        """Warms up model inference so cold-start overhead does not bias timings."""
        if self.classifier is not None:
            dummy = np.zeros((160, 160, 3), dtype=np.uint8)
            try:
                self.classifier.predict(dummy)
                self._warmup_done = True
            except Exception:
                pass

        if self.yolo_enabled and self.detector is not None:
            try:
                self.detector.load_model()
            except Exception as exc:
                logger.warning("YOLODetector startup load/warmup failed (safe fallback active): %s", exc)

    def open_camera(self) -> bool:
        """Opens the specified webcam gracefully."""
        if not CV2_AVAILABLE:
            return False
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            return self.cap.isOpened()
        except Exception:
            self.cap = None
            return False

    def close_camera(self) -> None:
        """Closes webcam capture device."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def process_frame(
        self, frame: np.ndarray, object_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes the real-time classification, reliability, supervision, and UDP pipeline
        for a single video frame.

        When YOLO is enabled:
        - If no object is detected, returns early with action="skipped" (no MobileNet,
          no UDP, no telemetry).
        - If an object is detected, the padded ROI is extracted and passed to MobileNetV2.

        Parameters
        ----------
        frame : np.ndarray
            BGR image array from camera or test source.
        object_id : int, optional
            Identifier for waste object if tracked.

        Returns
        -------
        dict
            On YOLO skip: {"action": "skipped", "reason": "no_object_detected", ...}
            On classification: complete execution payload including prediction,
            reliability, supervision, UDP status, and latency measurements.
        """
        if self.classifier is None:
            raise RuntimeError("WasteClassifier is not available.")

        # ------------------------------------------------------------------
        # YOLO object detection / background rejection
        # ------------------------------------------------------------------
        yolo_latency_ms: Optional[float] = None
        yolo_detection = None
        roi_coords: Optional[tuple] = None

        if self.yolo_enabled and self.detector is not None:
            t_yolo_start = time.perf_counter()
            try:
                crop_result = self.detector.detect_and_crop(
                    frame,
                    padding_fraction=self.yolo_roi_padding,
                )
            except Exception:
                crop_result = None
            yolo_latency_ms = (time.perf_counter() - t_yolo_start) * 1000.0

            if crop_result is None:
                # No object detected — silently skip this frame
                return {
                    "action": "skipped",
                    "reason": "no_object_detected",
                    "yolo_latency_ms": yolo_latency_ms,
                }

            yolo_detection, frame_roi, roi_coords = crop_result
            # MobileNetV2 receives the ROI crop (BGR); its preprocess() handles
            # resize to 160x160, BGR→RGB conversion, float32, and normalization.
            classify_frame = frame_roi
        else:
            # YOLO disabled: pass full frame to classifier (legacy behavior)
            classify_frame = frame

        # ------------------------------------------------------------------
        # MobileNetV2 classification
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        pred = self.classifier.predict(classify_frame)
        t1 = time.perf_counter()
        classifier_latency_ms = (t1 - t0) * 1000.0

        # Evaluate reliability
        rel = self.gate.evaluate(pred["probabilities"])
        decision = rel["decision"]
        confidence = rel["max_confidence"]
        entropy = rel["entropy"]
        pred_class_id = rel["predicted_class_id"]
        pred_class_name = rel["predicted_class_name"]

        # Human supervision logic
        if decision == "ACCEPT":
            final_class_id = pred_class_id
            final_class_name = pred_class_name
            decision_source = "classifier"
            override = False
        else:
            # UNCERTAIN: check if gesture supervisor is active and functional
            if self.supervisor is not None and getattr(self.supervisor, "is_available", False):
                sup_res = self.supervisor.supervise(
                    classifier_decision=decision,
                    classifier_class_id=pred_class_id,
                    frame=frame,
                )
                final_class_id = sup_res.get("class_id", pred_class_id)
                final_class_name = sup_res.get("class_name", pred_class_name)
                decision_source = sup_res.get("source", "unresolved")
                override = sup_res.get("override", False)
            else:
                # MediaPipe not available: fallback to unresolved
                final_class_id = pred_class_id
                final_class_name = pred_class_name
                decision_source = "unresolved"
                override = False

        # Build UDP message
        udp_msg = build_message(
            source=decision_source,
            decision=decision,
            class_id=final_class_id,
            class_name=final_class_name,
            confidence=confidence,
            entropy=entropy,
            override=override,
        )

        t2 = time.perf_counter()
        udp_res = self.sender.send_decision(udp_msg)
        t3 = time.perf_counter()
        udp_latency_ms = (t3 - t2) * 1000.0

        e2e_latency_ms = (yolo_latency_ms or 0.0) + classifier_latency_ms + udp_latency_ms

        # Record telemetry
        rec = TelemetryRecord(
            event_type="classification",
            object_id=object_id,
            class_id=final_class_id,
            class_name=final_class_name,
            confidence=confidence,
            entropy=entropy,
            reliability_decision=decision,
            decision_source=decision_source,
            override=override,
            classifier_latency_ms=classifier_latency_ms,
            udp_latency_ms=udp_latency_ms,
            e2e_latency_ms=e2e_latency_ms,
        )
        self.logger.log(rec)

        return {
            "action": "classified",
            "prediction": pred,
            "reliability": rel,
            "final_class_id": final_class_id,
            "final_class_name": final_class_name,
            "decision_source": decision_source,
            "override": override,
            "udp_status": udp_res,
            "classifier_latency_ms": classifier_latency_ms,
            "yolo_latency_ms": yolo_latency_ms,
            "udp_latency_ms": udp_latency_ms,
            "e2e_latency_ms": e2e_latency_ms,
            "telemetry_record": rec.to_dict(),
            # YOLO detection context
            "yolo_detection": {
                "class_id": yolo_detection.class_id,
                "class_name": yolo_detection.class_name,
                "confidence": yolo_detection.confidence,
                "box_xyxy": yolo_detection.box_xyxy,
                "roi_coords": roi_coords,
            } if yolo_detection is not None else None,
        }

    def run_loop(self, max_frames: Optional[int] = None) -> int:
        """
        Runs the real-time classification loop.
        Press 'q' in OpenCV window or terminate after max_frames.

        Returns the number of frames that were classified (YOLO-accepted frames only).
        """
        opened = self.open_camera()
        if not opened:
            print(f"[RealtimeApp] Warning: Camera {self.camera_index} could not be opened.")
            return 0

        frames_captured = 0
        frames_classified = 0
        frames_skipped = 0

        try:
            while True:
                if max_frames is not None and frames_captured >= max_frames:
                    break

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break

                frames_captured += 1
                res = self.process_frame(frame, object_id=frames_classified)

                if res.get("action") == "skipped":
                    frames_skipped += 1
                    # HUD: show YOLO "no object" state if cv2 available
                    try:
                        yolo_lat = res.get("yolo_latency_ms", 0.0) or 0.0
                        text = f"YOLO: No object detected | YOLO Lat: {yolo_lat:.1f}ms"
                        cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (80, 80, 80), 2)
                        cv2.imshow("Waste Classification Twin - Webcam Stream", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except Exception:
                        pass
                    continue

                frames_classified += 1

                # Visual HUD overlay if cv2 gui available
                try:
                    cname = res["final_class_name"]
                    conf = res["reliability"]["max_confidence"]
                    dec = res["reliability"]["decision"]
                    src = res["decision_source"]
                    lat = res["classifier_latency_ms"]
                    yolo_lat = res.get("yolo_latency_ms") or 0.0
                    yolo_det = res.get("yolo_detection")
                    yolo_info = ""
                    if yolo_det:
                        yolo_info = f" | YOLO: {yolo_det['class_name']} ({yolo_det['confidence']*100:.0f}%) {yolo_lat:.1f}ms"

                    text = f"Class: {cname.upper()} ({conf*100:.1f}%) | Gate: {dec} | Src: {src} | MNet: {lat:.1f}ms{yolo_info}"
                    color = (0, 255, 0) if dec == "ACCEPT" else (0, 165, 255)
                    cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                    # Draw YOLO bounding box if available
                    if yolo_det and "roi_coords" in yolo_det and yolo_det["roi_coords"]:
                        x1, y1, x2, y2 = yolo_det["roi_coords"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    cv2.imshow("Waste Classification Twin - Webcam Stream", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except Exception:
                    pass

        finally:
            self.close_camera()
            if CV2_AVAILABLE:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            self.logger.close()

        print(f"[RealtimeApp] Finished. Captured: {frames_captured} | Classified: {frames_classified} | Skipped (no object): {frames_skipped}")
        return frames_classified


def main():
    parser = argparse.ArgumentParser(description="Realtime AI Waste Classification App")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--model", type=str, default="models/mobilenetv2.keras", help="Model path")
    parser.add_argument("--conf-thresh", type=float, default=0.70, help="MobileNetV2 confidence threshold")
    parser.add_argument("--ent-thresh", type=float, default=1.20, help="MobileNetV2 entropy threshold")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="UDP host")
    parser.add_argument("--port", type=int, default=5005, help="UDP port")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to capture")
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO (legacy full-frame mode)")
    parser.add_argument("--yolo-model", type=str, default=YOLO_DEFAULT_MODEL, help="YOLO model path/name")
    parser.add_argument("--yolo-conf", type=float, default=YOLO_DEFAULT_CONFIDENCE_THRESHOLD, help="YOLO detection confidence threshold")
    parser.add_argument("--yolo-padding", type=float, default=YOLO_DEFAULT_ROI_PADDING, help="YOLO ROI padding fraction")
    args = parser.parse_args()

    app = RealtimeApp(
        camera_index=args.camera,
        model_path=args.model,
        confidence_threshold=args.conf_thresh,
        entropy_threshold=args.ent_thresh,
        udp_host=args.host,
        udp_port=args.port,
        yolo_enabled=not args.no_yolo,
        yolo_model=args.yolo_model,
        yolo_confidence_threshold=args.yolo_conf,
        yolo_roi_padding=args.yolo_padding,
    )
    processed = app.run_loop(max_frames=args.max_frames)
    print(f"[RealtimeApp] Total classified frames: {processed}")


# Backward compatibility alias
RealtimeClassifierApp = RealtimeApp

if __name__ == "__main__":
    main()
