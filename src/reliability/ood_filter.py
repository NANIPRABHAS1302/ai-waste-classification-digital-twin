"""
Threshold-based prediction rejection filter.

WHAT THIS MODULE DOES
---------------------
OodFilter wraps ConfidenceGate and re-exposes its ACCEPT / UNCERTAIN output
under names consistent with the existing scaffold (ood_filter.py).

WHAT THIS MODULE DOES NOT DO
------------------------------
This module does NOT implement genuine out-of-distribution (OOD) detection.
It does not use feature-space density estimation, Mahalanobis distance,
energy scoring, normalising flows, or any method that models the in-distribution
manifold of training data.

The UNCERTAIN / REJECT decision is driven solely by:
  - Maximum softmax confidence falling below a configurable threshold (default 0.70)
  - Shannon entropy exceeding a configurable threshold (default 1.20)

These are known-class confidence heuristics. They are not validated OOD detectors.
Do not interpret an UNCERTAIN result as a guarantee that the input is out-of-
distribution; high entropy may also arise from difficult in-distribution samples.

USAGE
-----
OodFilter is intended as a facade over ConfidenceGate for callers that reference
the ood_filter module by name. All substantive logic lives in ConfidenceGate.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np

from src.reliability.confidence_gate import (
    ConfidenceGate,
    DEFAULT_CLASS_NAMES,
    DECISION_ACCEPT,
    DECISION_UNCERTAIN,
)

# Re-export constants so callers can import from this module
__all__ = ["OodFilter", "DECISION_ACCEPT", "DECISION_UNCERTAIN"]


class OodFilter:
    """
    Thin facade over ConfidenceGate that provides confidence-and-entropy-based
    prediction rejection.

    This is NOT a generalised OOD detector. See module docstring above.

    Parameters
    ----------
    confidence_threshold : float
        Minimum max-class probability required to accept a prediction.
        Default 0.70 — from configs/config.yaml (reliability.confidence_threshold).
    entropy_threshold : float
        Maximum Shannon entropy (nats) allowed for an accepted prediction.
        Default 1.20 — from configs/config.yaml (reliability.entropy_threshold).
    class_names : sequence of str, optional
        Class names aligned with probability vector indices.
        Defaults to the canonical 6-class mapping.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        entropy_threshold: float = 1.20,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self._gate = ConfidenceGate(
            confidence_threshold=confidence_threshold,
            entropy_threshold=entropy_threshold,
            class_names=class_names,
        )

    @property
    def confidence_threshold(self) -> float:
        return self._gate.confidence_threshold

    @property
    def entropy_threshold(self) -> float:
        return self._gate.entropy_threshold

    @property
    def class_names(self):
        return self._gate.class_names

    def filter(
        self,
        probabilities: Union[np.ndarray, Sequence[float]],
    ) -> Dict:
        """
        Applies confidence + entropy thresholding to a probability vector.

        Returns the same structured result as ConfidenceGate.evaluate(),
        with an additional key:

            rejected : bool — True if decision is UNCERTAIN, False if ACCEPT.

        Parameters
        ----------
        probabilities : array-like of float
            6-element softmax probability vector.

        Returns
        -------
        dict — see ConfidenceGate.evaluate() for key definitions.
        """
        result = self._gate.evaluate(probabilities)
        result["rejected"] = result["decision"] == DECISION_UNCERTAIN
        return result

    def is_rejected(
        self,
        probabilities: Union[np.ndarray, Sequence[float]],
    ) -> bool:
        """
        Returns True if the prediction should be rejected (UNCERTAIN).

        Parameters
        ----------
        probabilities : array-like of float

        Returns
        -------
        bool
        """
        return self.filter(probabilities)["rejected"]
