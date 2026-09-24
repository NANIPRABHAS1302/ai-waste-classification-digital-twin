"""
UDP JSON sender for the waste-classification digital twin pipeline.

ROLE
----
Serialises a sorting-decision payload to JSON, encodes it as UTF-8,
and transmits it over a single UDP datagram to a configured localhost
address and port.

The sender is stateless: every call to send_decision() opens a fresh
socket, sends one datagram, and closes the socket immediately. This
avoids leaking sockets and keeps the lifecycle explicit and testable.

SECURITY / SAFETY
-----------------
- Default bind target is 127.0.0.1 (loopback only).
- Maximum payload is enforced before transmission.
- eval() is never called.
- Message contents are never executed.
- No authentication or encryption is implemented (local research prototype).

CONFIGURATION
-------------
Default values match configs/config.yaml:
    host : "127.0.0.1"
    port : 5005

MESSAGE SCHEMA
--------------
See MESSAGE_SCHEMA_FIELDS and build_message() for field definitions.
"""

from __future__ import annotations

import json
import math
import socket
import time
from typing import Any, Dict, Optional, Union

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

# Maximum UDP payload size (bytes). Standard safe MTU minus IP/UDP headers.
# 65 507 bytes is the theoretical UDP maximum; 8 192 bytes is a conservative
# limit for JSON text payloads, sufficient for any realistic sorting message.
MAX_PAYLOAD_BYTES: int = 8_192

# Canonical valid values for enum-like fields
VALID_DECISIONS = frozenset({"ACCEPT", "UNCERTAIN"})
VALID_SOURCES = frozenset({"classifier", "human_gesture", "unresolved"})
VALID_CLASS_NAMES = frozenset({
    "cardboard", "glass", "metal", "paper", "plastic", "trash"
})
VALID_CLASS_IDS = frozenset(range(6))  # 0–5

# Required top-level fields in every outgoing message
REQUIRED_FIELDS = (
    "timestamp",
    "message_type",
    "class_id",
    "class_name",
    "decision",
    "source",
    "confidence",
    "entropy",
    "override",
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5005
DEFAULT_MESSAGE_TYPE = "sorting_decision"


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def build_message(
    *,
    class_id: int,
    class_name: str,
    decision: str,
    source: str,
    confidence: Optional[float],
    entropy: Optional[float],
    override: bool,
    timestamp: Optional[float] = None,
    message_type: str = DEFAULT_MESSAGE_TYPE,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Builds a validated sorting-decision message dictionary.

    Parameters
    ----------
    class_id : int
        Predicted class index (0–5).
    class_name : str
        Predicted class label (must match VALID_CLASS_NAMES).
    decision : str
        Reliability gate decision: "ACCEPT" or "UNCERTAIN".
    source : str
        Decision source: "classifier", "human_gesture", or "unresolved".
    confidence : float or None
        Max softmax confidence in [0, 1], or None when unresolved.
    entropy : float or None
        Shannon entropy >= 0, or None when unresolved.
    override : bool
        True if a human gesture overrode the classifier.
    timestamp : float, optional
        Unix timestamp. Defaults to time.time() if not supplied.
    message_type : str
        Protocol message type string. Default "sorting_decision".
    extra : dict, optional
        Optional additional key-value pairs merged into the message.
        Keys must not overlap with required fields.

    Returns
    -------
    dict — the validated message.

    Raises
    ------
    ValueError : if any required field has an invalid value.
    TypeError  : if a field has the wrong Python type.
    """
    if timestamp is None:
        timestamp = time.time()

    # --- Type checks ---
    if not isinstance(class_id, int):
        raise TypeError(f"class_id must be int, got {type(class_id).__name__}.")
    if not isinstance(class_name, str):
        raise TypeError(f"class_name must be str, got {type(class_name).__name__}.")
    if not isinstance(decision, str):
        raise TypeError(f"decision must be str, got {type(decision).__name__}.")
    if not isinstance(source, str):
        raise TypeError(f"source must be str, got {type(source).__name__}.")
    if not isinstance(override, bool):
        raise TypeError(f"override must be bool, got {type(override).__name__}.")
    if confidence is not None and not isinstance(confidence, (int, float)):
        raise TypeError(f"confidence must be numeric or None, got {type(confidence).__name__}.")
    if entropy is not None and not isinstance(entropy, (int, float)):
        raise TypeError(f"entropy must be numeric or None, got {type(entropy).__name__}.")
    if not isinstance(timestamp, (int, float)):
        raise TypeError(f"timestamp must be numeric, got {type(timestamp).__name__}.")

    # --- Value checks ---
    if class_id not in VALID_CLASS_IDS:
        raise ValueError(f"class_id={class_id} is not in valid range 0–5.")
    if class_name not in VALID_CLASS_NAMES:
        raise ValueError(
            f"class_name={class_name!r} is not a valid class name. "
            f"Valid: {sorted(VALID_CLASS_NAMES)}."
        )
    if decision not in VALID_DECISIONS:
        raise ValueError(
            f"decision={decision!r} is invalid. Valid: {sorted(VALID_DECISIONS)}."
        )
    if source not in VALID_SOURCES:
        raise ValueError(
            f"source={source!r} is invalid. Valid: {sorted(VALID_SOURCES)}."
        )
    if confidence is not None:
        if not math.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
            raise ValueError(
                f"confidence={confidence} must be finite and in [0, 1]."
            )
    if entropy is not None:
        if not math.isfinite(entropy) or entropy < 0.0:
            raise ValueError(
                f"entropy={entropy} must be finite and >= 0."
            )

    msg: Dict[str, Any] = {
        "timestamp": float(timestamp),
        "message_type": message_type,
        "class_id": class_id,
        "class_name": class_name,
        "decision": decision,
        "source": source,
        "confidence": confidence,
        "entropy": entropy,
        "override": override,
    }

    if extra:
        overlap = set(extra.keys()) & set(REQUIRED_FIELDS)
        if overlap:
            raise ValueError(
                f"'extra' keys overlap with required fields: {sorted(overlap)}."
            )
        msg.update(extra)

    return msg


# ---------------------------------------------------------------------------
# UDP Sender
# ---------------------------------------------------------------------------

class UdpSender:
    """
    Sends a single JSON-encoded sorting-decision datagram over UDP.

    Parameters
    ----------
    host : str
        Destination host. Default "127.0.0.1".
    port : int
        Destination port. Default 5005.
    timeout_s : float
        Socket send timeout in seconds. Default 1.0.
    max_payload_bytes : int
        Maximum allowed payload size. Rejects messages exceeding this.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout_s: float = 1.0,
        max_payload_bytes: int = MAX_PAYLOAD_BYTES,
    ) -> None:
        if not isinstance(port, int) or not (1 <= port <= 65535):
            raise ValueError(f"port must be in [1, 65535], got {port!r}.")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}.")
        if max_payload_bytes <= 0:
            raise ValueError(
                f"max_payload_bytes must be positive, got {max_payload_bytes}."
            )

        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.max_payload_bytes = max_payload_bytes

    def send_decision(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates, serialises, and sends one sorting-decision message.

        Does NOT perform classification or reliability calculations.
        Transports the supplied values unchanged.

        Parameters
        ----------
        message : dict
            A validated message dict (typically from build_message()).
            Must contain all REQUIRED_FIELDS.

        Returns
        -------
        dict with keys:
            success      : bool
            bytes_sent   : int (0 on failure)
            host         : str
            port         : int
            error        : str or None
        """
        # Verify required fields are present
        missing = [f for f in REQUIRED_FIELDS if f not in message]
        if missing:
            return {
                "success": False,
                "bytes_sent": 0,
                "host": self.host,
                "port": self.port,
                "error": f"Missing required fields: {sorted(missing)}",
            }

        # Serialize
        try:
            payload_str = json.dumps(message, ensure_ascii=True)
            payload_bytes = payload_str.encode("utf-8")
        except (TypeError, ValueError) as exc:
            return {
                "success": False,
                "bytes_sent": 0,
                "host": self.host,
                "port": self.port,
                "error": f"JSON serialisation error: {exc}",
            }

        # Enforce payload size limit
        if len(payload_bytes) > self.max_payload_bytes:
            return {
                "success": False,
                "bytes_sent": 0,
                "host": self.host,
                "port": self.port,
                "error": (
                    f"Payload too large: {len(payload_bytes)} bytes "
                    f"> limit {self.max_payload_bytes} bytes."
                ),
            }

        # Send
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(self.timeout_s)
                bytes_sent = sock.sendto(payload_bytes, (self.host, self.port))
        except OSError as exc:
            return {
                "success": False,
                "bytes_sent": 0,
                "host": self.host,
                "port": self.port,
                "error": f"Socket error: {exc}",
            }

        return {
            "success": True,
            "bytes_sent": bytes_sent,
            "host": self.host,
            "port": self.port,
            "error": None,
        }
