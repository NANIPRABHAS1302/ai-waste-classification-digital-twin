"""
UDP JSON receiver for the waste-classification digital twin pipeline.

ROLE
----
Binds to a configured UDP port, receives one datagram, decodes it
from UTF-8, parses it as JSON, validates the required fields and value
ranges, and returns a structured result.

The receiver is stateless and designed to be called inside a loop
(e.g., the digital twin's event loop). Every call to receive_one()
performs one blocking recv() with a configurable timeout.

SECURITY / SAFETY
-----------------
- Binds only to the configured interface (default 127.0.0.1).
- Enforces a maximum accepted payload size; oversized datagrams are
  truncated by the OS and rejected by length check.
- eval() is never called.
- Message contents are never executed.
- Malformed UTF-8, invalid JSON, and structurally invalid messages
  are all caught and returned as structured error results — they do
  not crash the receiver.
- No authentication or encryption (local research prototype).

VALIDATION
----------
The receiver independently validates every received message against
the same field and value constraints as the sender. A receiver must
not silently accept a structurally invalid message.

Validated fields (see REQUIRED_FIELDS, validate_message()):
    timestamp, message_type, class_id, class_name, decision,
    source, confidence, entropy, override
"""

from __future__ import annotations

import json
import math
import socket
from typing import Any, Dict, Optional, Tuple

from src.communication.udp_sender import (
    MAX_PAYLOAD_BYTES,
    REQUIRED_FIELDS,
    VALID_CLASS_IDS,
    VALID_CLASS_NAMES,
    VALID_DECISIONS,
    VALID_SOURCES,
    DEFAULT_HOST,
    DEFAULT_PORT,
)

# ---------------------------------------------------------------------------
# Receive result status codes
# ---------------------------------------------------------------------------

STATUS_OK = "ok"
STATUS_TIMEOUT = "timeout"
STATUS_DECODE_ERROR = "decode_error"
STATUS_JSON_ERROR = "json_error"
STATUS_VALIDATION_ERROR = "validation_error"
STATUS_PAYLOAD_TOO_LARGE = "payload_too_large"
STATUS_SOCKET_ERROR = "socket_error"
STATUS_EMPTY_DATAGRAM = "empty_datagram"


# ---------------------------------------------------------------------------
# Message validator
# ---------------------------------------------------------------------------

def validate_message(msg: Dict[str, Any]) -> Optional[str]:
    """
    Validates a parsed message dictionary against the protocol schema.

    Returns None if valid, or an error string describing the first
    violation found.

    Parameters
    ----------
    msg : dict
        Parsed JSON object from a received datagram.

    Returns
    -------
    str or None
    """
    # Check all required fields are present
    missing = [f for f in REQUIRED_FIELDS if f not in msg]
    if missing:
        return f"Missing required fields: {sorted(missing)}"

    # timestamp: numeric
    ts = msg["timestamp"]
    if not isinstance(ts, (int, float)) or not math.isfinite(float(ts)):
        return f"timestamp must be a finite numeric value, got {ts!r}."

    # message_type: string
    if not isinstance(msg["message_type"], str):
        return f"message_type must be str, got {type(msg['message_type']).__name__}."

    # class_id: int in 0–5
    class_id = msg["class_id"]
    if not isinstance(class_id, int) or isinstance(class_id, bool):
        return f"class_id must be int, got {type(class_id).__name__}."
    if class_id not in VALID_CLASS_IDS:
        return f"class_id={class_id} is not in valid range 0–5."

    # class_name: valid string
    class_name = msg["class_name"]
    if not isinstance(class_name, str):
        return f"class_name must be str, got {type(class_name).__name__}."
    if class_name not in VALID_CLASS_NAMES:
        return (
            f"class_name={class_name!r} is not a valid class name. "
            f"Valid: {sorted(VALID_CLASS_NAMES)}."
        )

    # decision: "ACCEPT" or "UNCERTAIN"
    decision = msg["decision"]
    if not isinstance(decision, str):
        return f"decision must be str, got {type(decision).__name__}."
    if decision not in VALID_DECISIONS:
        return f"decision={decision!r} is invalid. Valid: {sorted(VALID_DECISIONS)}."

    # source: "classifier", "human_gesture", or "unresolved"
    source = msg["source"]
    if not isinstance(source, str):
        return f"source must be str, got {type(source).__name__}."
    if source not in VALID_SOURCES:
        return f"source={source!r} is invalid. Valid: {sorted(VALID_SOURCES)}."

    # confidence: None or numeric in [0, 1]
    confidence = msg["confidence"]
    if confidence is not None:
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            return f"confidence must be numeric or null, got {type(confidence).__name__}."
        if not math.isfinite(float(confidence)) or not (0.0 <= float(confidence) <= 1.0):
            return f"confidence={confidence} must be finite and in [0, 1]."

    # entropy: None or numeric >= 0
    entropy = msg["entropy"]
    if entropy is not None:
        if not isinstance(entropy, (int, float)) or isinstance(entropy, bool):
            return f"entropy must be numeric or null, got {type(entropy).__name__}."
        if not math.isfinite(float(entropy)) or float(entropy) < 0.0:
            return f"entropy={entropy} must be finite and >= 0."

    # override: bool (not int/float)
    override = msg["override"]
    if not isinstance(override, bool):
        return f"override must be bool, got {type(override).__name__}."

    return None  # Valid


# ---------------------------------------------------------------------------
# UDP Receiver
# ---------------------------------------------------------------------------

class UdpReceiver:
    """
    Receives and validates one JSON-encoded sorting-decision datagram.

    Parameters
    ----------
    host : str
        Interface to bind to. Default "127.0.0.1".
    port : int
        Port to bind to. Default 5005.
    timeout_s : float
        Socket recv timeout in seconds. Default 2.0.
    max_payload_bytes : int
        Maximum accepted datagram size. Datagrams truncated to this size
        are detected and rejected. Default MAX_PAYLOAD_BYTES.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout_s: float = 2.0,
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

    def receive_one(self) -> Dict[str, Any]:
        """
        Binds to the configured port, waits for one datagram, validates it,
        and returns a structured result. The socket is closed after one
        receive regardless of outcome.

        Returns
        -------
        dict with keys:
            status   : str (see STATUS_* constants)
            message  : dict or None — parsed and validated message
            error    : str or None — description of any failure
            addr     : (host, port) tuple or None — sender address
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.settimeout(self.timeout_s)
                sock.bind((self.host, self.port))
                return self._recv(sock)
        except OSError as exc:
            return _error_result(STATUS_SOCKET_ERROR, f"Socket error: {exc}")

    def receive_one_on_socket(self, sock: socket.socket) -> Dict[str, Any]:
        """
        Receives one datagram from an already-bound socket.

        Used by tests that manage the socket lifecycle externally
        (e.g., to avoid repeated bind/unbind overhead in tight loops).

        Parameters
        ----------
        sock : socket.socket
            An already-bound UDP socket.

        Returns
        -------
        dict — same structure as receive_one().
        """
        return self._recv(sock)

    def _recv(self, sock: socket.socket) -> Dict[str, Any]:
        """Internal: receive one datagram from the given socket."""
        addr: Optional[Tuple[str, int]] = None
        try:
            # Receive up to max_payload_bytes + 1 to detect oversized datagrams
            raw_bytes, addr = sock.recvfrom(self.max_payload_bytes + 1)
        except socket.timeout:
            return _error_result(STATUS_TIMEOUT, "Receive timed out.")
        except OSError as exc:
            return _error_result(STATUS_SOCKET_ERROR, f"recvfrom error: {exc}")

        # Empty datagram
        if len(raw_bytes) == 0:
            return _error_result(STATUS_EMPTY_DATAGRAM, "Received empty datagram.", addr)

        # Oversized datagram (OS truncated it)
        if len(raw_bytes) > self.max_payload_bytes:
            return _error_result(
                STATUS_PAYLOAD_TOO_LARGE,
                f"Datagram exceeds max payload size ({self.max_payload_bytes} bytes).",
                addr,
            )

        # Decode UTF-8
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            return _error_result(STATUS_DECODE_ERROR, f"UTF-8 decode error: {exc}", addr)

        # Parse JSON
        try:
            msg = json.loads(text)
        except json.JSONDecodeError as exc:
            return _error_result(STATUS_JSON_ERROR, f"JSON parse error: {exc}", addr)

        if not isinstance(msg, dict):
            return _error_result(
                STATUS_VALIDATION_ERROR,
                f"Expected JSON object, got {type(msg).__name__}.",
                addr,
            )

        # Validate fields and values
        validation_error = validate_message(msg)
        if validation_error:
            return _error_result(STATUS_VALIDATION_ERROR, validation_error, addr)

        return {
            "status": STATUS_OK,
            "message": msg,
            "error": None,
            "addr": addr,
        }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _error_result(
    status: str,
    error: str,
    addr: Optional[Tuple[str, int]] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "message": None,
        "error": error,
        "addr": addr,
    }
