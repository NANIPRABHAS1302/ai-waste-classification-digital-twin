# AI-Powered Intelligent Waste Classification and Real-Time Sorting Digital Twin

### A Computer Vision Approach for Automated Waste Recognition, Human-in-the-Loop Control and Simulation

---

## 📌 Overview

This project presents an **AI-powered intelligent waste classification and real-time sorting system** that combines Computer Vision, Deep Learning, object detection, reliability-aware decision making, human-in-the-loop supervision, UDP communication, and Digital Twin simulation.

The system is designed to recognize waste objects from a live camera feed, classify them into six waste categories, evaluate prediction reliability, communicate the sorting decision, and simulate the corresponding physical sorting process through a Digital Twin.

### Waste Categories

| Class ID | Category |
|---:|---|
| 0 | Cardboard |
| 1 | Glass |
| 2 | Metal |
| 3 | Paper |
| 4 | Plastic |
| 5 | Trash |

---

# 🎯 Objectives

- Develop an automated waste classification system using Computer Vision.
- Detect objects from live webcam frames.
- Classify waste into six predefined categories.
- Use a lightweight deep-learning model for real-time classification.
- Introduce confidence and entropy-based reliability assessment.
- Provide human-in-the-loop intervention for uncertain predictions.
- Communicate sorting decisions through UDP.
- Build a real-time Digital Twin of the sorting process.
- Record system telemetry and latency measurements.
- Provide explainability using Grad-CAM.
- Evaluate multiple CNN architectures.

---

# 🏗️ System Architecture

```text
                         WEBCAM
                            │
                            ▼
                    ┌───────────────┐
                    │    YOLOv8n    │
                    │ Object Detect │
                    └───────┬───────┘
                            │
                     Object Detected?
                       /           \
                     NO             YES
                     │               │
                     ▼               ▼
                  SKIP FRAME      ROI CROP
                                     │
                                     ▼
                             ┌───────────────┐
                             │  MobileNetV2  │
                             │ Waste Class.  │
                             └───────┬───────┘
                                     │
                                     ▼
                         ┌─────────────────────┐
                         │ Reliability Layer   │
                         │ Confidence +Entropy │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                      ACCEPT               UNCERTAIN
                         │                     │
                         │                     ▼
                         │              Human Gesture
                         │                Override
                         │                     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                              UDP Communication
                                    │
                                    ▼
                           ┌─────────────────┐
                           │   Digital Twin  │
                           │ Conveyor + Gate │
                           └────────┬────────┘
                                    │
                                    ▼
                            Six Waste Bins
                                    │
                                    ▼
                               Telemetry
