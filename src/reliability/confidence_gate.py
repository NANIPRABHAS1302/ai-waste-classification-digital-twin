"""
Confidence and entropy reliability gate for the waste classification pipeline.

MECHANISM
---------
Given the softmax probability vector p produced by MobileNetV2:

1. Max confidence:
       max_confidence = max(p)

2. Shannon entropy:
       H(p) = -sum(p_i * log(p_i))   for p_i > 0

   (p_i = 0 terms are skipped to avoid 0 * log(0), which is 0 by convention.)

3. Decision:
       ACCEPT   if max_confidence >= confidence_threshold
                AND entropy <= entropy_threshold
       UNCERTAIN  otherwise

TERMINOLOGY NOTE
----------------
This is a confidence-and-entropy reliability/rejection mechanism. It applies
two scalar thresholds to the model's own softmax output. It does NOT constitute
a scientifically validated out-of-distribution (OOD) detector. It will not
reliably detect novel categories or distribution-shifted inputs not seen during
training. The thresholds (0.70 / 1.20) are initial engineering values, not
calibrated bounds derived from held-out experiments.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

# Canonical six-class mapping — must match training order in 02_train_models.py
DEFAULT_CLASS_NAMES: List[str] = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

# Decision string constants
DECISION_ACCEPT = "ACCEPT"
DECISION_UNCERTAIN = "UNCERTAIN"


class ConfidenceGate:
    """
    Applies a confidence + Shannon entropy reliability check to a softmax
    probability vector.

    Parameters
    ----------
    confidence_threshold : float
        Minimum max-class probability required for ACCEPT.
        Default 0.70 (engineering initial value from configs/config.yaml).
    entropy_threshold : float
        Maximum Shannon entropy (nats) allowed for ACCEPT.
        Default 1.20 (engineering initial value from configs/config.yaml).
    class_names : sequence of str, optional
        Class name labels aligned with probability vector indices.
        Defaults to the canonical 6-class mapping.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        entropy_threshold: float = 1.20,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}."
            )
        if entropy_threshold < 0.0:
            raise ValueError(
                f"entropy_threshold must be non-negative, got {entropy_threshold}."
            )

        self.confidence_threshold = float(confidence_threshold)
        self.entropy_threshold = float(entropy_threshold)
        self.class_names: List[str] = (
            list(class_names) if class_names is not None else list(DEFAULT_CLASS_NAMES)
        )
        self.num_classes = len(self.class_names)

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

    def _validate_probabilities(self, probabilities: np.ndarray) -> None:
        """
        Validates a probability vector before evaluation.

        Raises
        ------
        TypeError  : if probabilities is not a numpy array.
        ValueError : if shape, values, or sum are invalid.
        """
        if not isinstance(probabilities, np.ndarray):
            raise TypeError(
                f"probabilities must be a numpy.ndarray, got {type(probabilities).__name__}."
            )
        if probabilities.ndim != 1:
            raise ValueError(
                f"probabilities must be 1-dimensional, got shape {probabilities.shape}."
            )
        if len(probabilities) != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} probabilities, got {len(probabilities)}."
            )
        if not np.all(np.isfinite(probabilities)):
            raise ValueError(
                "probabilities contain non-finite values (NaN or Inf)."
            )
        if np.any(probabilities < 0.0):
            raise ValueError(
                "probabilities contain negative values."
            )
        prob_sum = float(np.sum(probabilities))
        if not math.isclose(prob_sum, 1.0, abs_tol=1e-2):
            raise ValueError(
                f"probabilities must sum to approximately 1.0, got sum = {prob_sum:.6f}."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def compute_entropy(probabilities: np.ndarray) -> float:
        """
        Computes Shannon entropy H(p) = -sum(p_i * log(p_i)) in nats.

        p_i = 0 terms are skipped (0 * log(0) = 0 by convention).
        Input is not validated here; call _validate_probabilities first.

        Parameters
        ----------
        probabilities : np.ndarray
            1-D array of non-negative floats summing to ~1.

        Returns
        -------
        float : Shannon entropy >= 0 (in nats).
        """
        p = probabilities.astype(np.float64)
        # Mask out zero entries to avoid log(0)
        nonzero = p[p > 0.0]
        entropy = -float(np.sum(nonzero * np.log(nonzero)))
        return max(entropy, 0.0)  # Clamp numerical noise near 0

    def evaluate(
        self,
        probabilities: Union[np.ndarray, Sequence[float]],
    ) -> Dict:
        """
        Evaluates the reliability of a softmax probability vector.

        Parameters
        ----------
        probabilities : array-like of float
            6-element softmax probability vector from MobileNetV2.

        Returns
        -------
        dict with keys:
            decision           : "ACCEPT" or "UNCERTAIN"
            max_confidence     : float, max(p)
            entropy            : float, Shannon entropy H(p) in nats
            predicted_class_id : int, argmax(p)
            predicted_class_name : str, class label for predicted_class_id
            confidence_threshold : float, threshold used
            entropy_threshold  : float, threshold used
        """
        # Type guard: reject non-sequence types before np.asarray() is called.
        # np.asarray() on a plain string raises ValueError (not TypeError),
        # which would be a misleading error for the caller.
        if not isinstance(probabilities, (np.ndarray, list, tuple)):
            raise TypeError(
                f"probabilities must be a numpy.ndarray, list, or tuple, "
                f"got {type(probabilities).__name__}."
            )
        probs = np.asarray(probabilities, dtype=np.float64)
        self._validate_probabilities(probs)

        max_confidence = float(np.max(probs))
        predicted_class_id = int(np.argmax(probs))
        predicted_class_name = self.class_names[predicted_class_id]
        entropy = self.compute_entropy(probs)

        if (
            max_confidence >= self.confidence_threshold
            and entropy <= self.entropy_threshold
        ):
            decision = DECISION_ACCEPT
        else:
            decision = DECISION_UNCERTAIN

        return {
            "decision": decision,
            "max_confidence": max_confidence,
            "entropy": entropy,
            "predicted_class_id": predicted_class_id,
            "predicted_class_name": predicted_class_name,
            "confidence_threshold": self.confidence_threshold,
            "entropy_threshold": self.entropy_threshold,
        }

    def is_accepted(
        self,
        probabilities: Union[np.ndarray, Sequence[float]],
    ) -> bool:
        """
        Convenience method. Returns True if the prediction is ACCEPT.

        Parameters
        ----------
        probabilities : array-like of float

        Returns
        -------
        bool
        """
        return self.evaluate(probabilities)["decision"] == DECISION_ACCEPT
