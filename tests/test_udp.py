"""
Unit and integration tests for the UDP communication layer — Phase 4.

Test strategy:
  - Serialisation/deserialisation: pure Python, no network I/O
  - Validation: pure Python, no network I/O
  - Round-trip: localhost UDP using an ephemeral port allocated by the
    OS (port=0), avoiding conflicts with any running process or with
    the project's default port 5005
  - All tests are self-contained and require no internet access

Test groups:
  A. build_message() — construction and validation
  B. validate_message() — receiver-side field validation
  C. UdpSender serialisation (no socket)
  D. UdpReceiver validation edge cases (no socket)
  E. Localhost round-trip integration tests (ephemeral port)
  F. Receiver timeout test
  G. Regression: Phase 1 + Phase 2 + Phase 3 smoke imports
"""

import json
import socket
import threading
import time
import unittest

from src.communication.udp_sender import (
    UdpSender,
    build_message,
    MAX_PAYLOAD_BYTES,
    REQUIRED_FIELDS,
    VALID_CLASS_IDS,
    VALID_CLASS_NAMES,
    VALID_DECISIONS,
    VALID_SOURCES,
    DEFAULT_HOST,
    DEFAULT_PORT,
)
from src.communication.udp_receiver import (
    UdpReceiver,
    validate_message,
    STATUS_OK,
    STATUS_TIMEOUT,
    STATUS_DECODE_ERROR,
    STATUS_JSON_ERROR,
    STATUS_VALIDATION_ERROR,
    STATUS_PAYLOAD_TOO_LARGE,
    STATUS_EMPTY_DATAGRAM,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _valid_message(**overrides) -> dict:
    """Returns a fully valid message dict with optional field overrides."""
    base = dict(
        class_id=4,
        class_name="plastic",
        decision="ACCEPT",
        source="classifier",
        confidence=0.91,
        entropy=0.42,
        override=False,
        timestamp=1_700_000_000.0,
    )
    base.update(overrides)
    return build_message(**base)


def _find_free_udp_port() -> int:
    """Binds an ephemeral UDP socket to get an OS-allocated free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# A. build_message() tests
# ---------------------------------------------------------------------------

class TestBuildMessage(unittest.TestCase):
    """Tests for the message builder and sender-side validation."""

    def test_01_valid_accept_message_serialises(self):
        """A fully valid ACCEPT message can be built and JSON-encoded."""
        msg = _valid_message()
        self.assertEqual(msg["decision"], "ACCEPT")
        self.assertEqual(msg["class_name"], "plastic")
        self.assertEqual(msg["class_id"], 4)
        self.assertFalse(msg["override"])
        encoded = json.dumps(msg)
        self.assertIsInstance(encoded, str)

    def test_02_valid_uncertain_message_serialises(self):
        """A fully valid UNCERTAIN message can be built."""
        msg = build_message(
            class_id=2, class_name="metal", decision="UNCERTAIN",
            source="unresolved", confidence=None, entropy=None,
            override=False, timestamp=1_700_000_001.0,
        )
        self.assertEqual(msg["decision"], "UNCERTAIN")
        self.assertIsNone(msg["confidence"])
        self.assertIsNone(msg["entropy"])

    def test_03_valid_human_gesture_override_message(self):
        """Human-gesture override message serialises correctly."""
        msg = build_message(
            class_id=2, class_name="metal", decision="UNCERTAIN",
            source="human_gesture", confidence=0.52, entropy=1.05,
            override=True, timestamp=1_700_000_002.0,
        )
        self.assertTrue(msg["override"])
        self.assertEqual(msg["source"], "human_gesture")

    def test_04_valid_unresolved_message(self):
        """Unresolved source with null confidence/entropy is valid."""
        msg = build_message(
            class_id=0, class_name="cardboard", decision="UNCERTAIN",
            source="unresolved", confidence=None, entropy=None,
            override=False, timestamp=1_700_000_003.0,
        )
        self.assertEqual(msg["source"], "unresolved")

    def test_07_all_six_class_names_accepted(self):
        """All six valid class names can be used in a message."""
        pairs = [(0, "cardboard"), (1, "glass"), (2, "metal"),
                 (3, "paper"), (4, "plastic"), (5, "trash")]
        for class_id, class_name in pairs:
            msg = build_message(
                class_id=class_id, class_name=class_name,
                decision="ACCEPT", source="classifier",
                confidence=0.85, entropy=0.3, override=False,
            )
            self.assertEqual(msg["class_id"], class_id)
            self.assertEqual(msg["class_name"], class_name)

    def test_08_invalid_class_id_raises(self):
        """class_id outside 0–5 raises ValueError."""
        with self.assertRaises(ValueError):
            build_message(
                class_id=6, class_name="plastic", decision="ACCEPT",
                source="classifier", confidence=0.9, entropy=0.3, override=False,
            )

    def test_08b_negative_class_id_raises(self):
        with self.assertRaises(ValueError):
            build_message(
                class_id=-1, class_name="plastic", decision="ACCEPT",
                source="classifier", confidence=0.9, entropy=0.3, override=False,
            )

    def test_09_invalid_class_name_raises(self):
        """Unknown class name raises ValueError."""
        with self.assertRaises(ValueError):
            build_message(
                class_id=0, class_name="banana", decision="ACCEPT",
                source="classifier", confidence=0.9, entropy=0.3, override=False,
            )

    def test_10_invalid_decision_raises(self):
        """Unknown decision string raises ValueError."""
        with self.assertRaises(ValueError):
            build_message(
                class_id=0, class_name="cardboard", decision="MAYBE",
                source="classifier", confidence=0.9, entropy=0.3, override=False,
            )

    def test_11_invalid_source_raises(self):
        """Unknown source string raises ValueError."""
        with self.assertRaises(ValueError):
            build_message(
                class_id=0, class_name="cardboard", decision="ACCEPT",
                source="robot_arm", confidence=0.9, entropy=0.3, override=False,
            )

    def test_12_invalid_confidence_above_one_raises(self):
        """confidence > 1.0 raises ValueError."""
        with self.assertRaises(ValueError):
            build_message(
                class_id=0, class_name="cardboard", decision="ACCEPT",
                source="classifier", confidence=1.5, entropy=0.3, override=False,
            )

    def test_12b_negative_confidence_raises(self):
        with self.assertRaises(ValueError):
            build_message(
                class_id=0, class_name="cardboard", decision="ACCEPT",
                source="classifier", confidence=-0.1, entropy=0.3, override=False,
            )

    def test_13_negative_entropy_raises(self):
        """Negative entropy raises ValueError."""
        with self.assertRaises(ValueError):
            build_message(
                class_id=0, class_name="cardboard", decision="ACCEPT",
                source="classifier", confidence=0.9, entropy=-1.0, override=False,
            )

    def test_14_override_not_bool_raises(self):
        """Non-bool override raises TypeError."""
        with self.assertRaises(TypeError):
            build_message(
                class_id=0, class_name="cardboard", decision="ACCEPT",
                source="classifier", confidence=0.9, entropy=0.3, override=1,
            )

    def test_timestamp_auto_assigned_if_none(self):
        """If timestamp is omitted, time.time() is used."""
        before = time.time()
        msg = build_message(
            class_id=0, class_name="cardboard", decision="ACCEPT",
            source="classifier", confidence=0.9, entropy=0.3, override=False,
        )
        after = time.time()
        self.assertGreaterEqual(msg["timestamp"], before)
        self.assertLessEqual(msg["timestamp"], after)

    def test_required_fields_all_present(self):
        """All REQUIRED_FIELDS appear in a built message."""
        msg = _valid_message()
        for field in REQUIRED_FIELDS:
            self.assertIn(field, msg, f"Missing required field: {field}")


# ---------------------------------------------------------------------------
# B. validate_message() tests (receiver-side)
# ---------------------------------------------------------------------------

class TestValidateMessage(unittest.TestCase):

    def test_valid_message_returns_none(self):
        msg = _valid_message()
        self.assertIsNone(validate_message(msg))

    def test_15_missing_required_field_returns_error(self):
        """Each missing required field must produce a validation error."""
        for field in REQUIRED_FIELDS:
            msg = _valid_message()
            del msg[field]
            error = validate_message(msg)
            self.assertIsNotNone(error, f"Expected error for missing field: {field}")
            self.assertIn(field, error)

    def test_invalid_class_id_returns_error(self):
        msg = _valid_message()
        msg["class_id"] = 99
        self.assertIsNotNone(validate_message(msg))

    def test_invalid_class_name_returns_error(self):
        msg = _valid_message()
        msg["class_name"] = "unknown"
        self.assertIsNotNone(validate_message(msg))

    def test_invalid_decision_returns_error(self):
        msg = _valid_message()
        msg["decision"] = "REJECTED"
        self.assertIsNotNone(validate_message(msg))

    def test_invalid_source_returns_error(self):
        msg = _valid_message()
        msg["source"] = "autopilot"
        self.assertIsNotNone(validate_message(msg))

    def test_invalid_confidence_type_returns_error(self):
        msg = _valid_message()
        msg["confidence"] = "high"
        self.assertIsNotNone(validate_message(msg))

    def test_invalid_entropy_negative_returns_error(self):
        msg = _valid_message()
        msg["entropy"] = -0.5
        self.assertIsNotNone(validate_message(msg))

    def test_invalid_override_not_bool_returns_error(self):
        """Integer 1 must be rejected for override (not bool)."""
        msg = _valid_message()
        msg["override"] = 1  # JSON int, not bool
        self.assertIsNotNone(validate_message(msg))

    def test_class_id_as_bool_rejected(self):
        """bool is a subtype of int in Python; validator must reject it."""
        msg = _valid_message()
        msg["class_id"] = True  # would be 1 as int — must reject bool
        self.assertIsNotNone(validate_message(msg))

    def test_null_confidence_is_valid(self):
        msg = _valid_message()
        msg["confidence"] = None
        self.assertIsNone(validate_message(msg))

    def test_null_entropy_is_valid(self):
        msg = _valid_message()
        msg["entropy"] = None
        self.assertIsNone(validate_message(msg))

    def test_22_deterministic_validation(self):
        """Same message validates identically across 10 calls."""
        msg = _valid_message()
        results = [validate_message(msg) for _ in range(10)]
        self.assertTrue(all(r is None for r in results))


# ---------------------------------------------------------------------------
# C. UdpSender serialisation (no socket needed)
# ---------------------------------------------------------------------------

class TestUdpSenderNoSocket(unittest.TestCase):
    """Tests for sender behaviour that does not require a live socket."""

    def test_01_send_missing_field_returns_failure(self):
        """Sending a message that is missing a required field → success=False."""
        sender = UdpSender(host="127.0.0.1", port=_find_free_udp_port())
        incomplete = {"timestamp": 1.0, "decision": "ACCEPT"}
        result = sender.send_decision(incomplete)
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])

    def test_23_oversized_payload_rejected_before_send(self):
        """Payload exceeding max_payload_bytes is rejected without touching the socket."""
        sender = UdpSender(
            host="127.0.0.1",
            port=_find_free_udp_port(),
            max_payload_bytes=10,  # absurdly small
        )
        msg = _valid_message()
        result = sender.send_decision(msg)
        self.assertFalse(result["success"])
        self.assertIn("too large", result["error"].lower())


# ---------------------------------------------------------------------------
# D. UdpReceiver validation edge cases (injected raw bytes, no live socket)
# ---------------------------------------------------------------------------

class _FakeSocket:
    """Minimal fake socket for injecting raw bytes into UdpReceiver._recv()."""
    def __init__(self, raw_bytes: bytes, addr=("127.0.0.1", 9999)):
        self._data = raw_bytes
        self._addr = addr

    def recvfrom(self, bufsize: int):
        return self._data[:bufsize], self._addr


class TestUdpReceiverValidation(unittest.TestCase):

    def setUp(self):
        self.receiver = UdpReceiver(host="127.0.0.1", port=_find_free_udp_port())

    def _recv_raw(self, raw_bytes: bytes) -> dict:
        return self.receiver._recv(_FakeSocket(raw_bytes))

    def test_16_malformed_json_returns_json_error(self):
        """Non-JSON text → STATUS_JSON_ERROR."""
        result = self._recv_raw(b"not json at all {{{")
        self.assertEqual(result["status"], STATUS_JSON_ERROR)
        self.assertIsNone(result["message"])

    def test_17_invalid_utf8_returns_decode_error(self):
        """Invalid UTF-8 bytes → STATUS_DECODE_ERROR."""
        result = self._recv_raw(b"\xff\xfe garbage")
        self.assertEqual(result["status"], STATUS_DECODE_ERROR)
        self.assertIsNone(result["message"])

    def test_18_empty_datagram_returns_empty_error(self):
        """Zero-length datagram → STATUS_EMPTY_DATAGRAM."""
        result = self._recv_raw(b"")
        self.assertEqual(result["status"], STATUS_EMPTY_DATAGRAM)

    def test_23b_oversized_datagram_rejected(self):
        """Datagram larger than max_payload_bytes → STATUS_PAYLOAD_TOO_LARGE."""
        tiny_receiver = UdpReceiver(
            host="127.0.0.1",
            port=_find_free_udp_port(),
            max_payload_bytes=5,
        )
        # FakeSocket returns exactly bufsize+1 bytes (simulates OS returning
        # max_payload_bytes+1 to signal truncation)
        big = b"A" * 6
        result = tiny_receiver._recv(_FakeSocket(big))
        self.assertEqual(result["status"], STATUS_PAYLOAD_TOO_LARGE)

    def test_valid_bytes_accepted(self):
        """A correctly formed JSON datagram passes validation."""
        msg = _valid_message()
        payload = json.dumps(msg).encode("utf-8")
        result = self._recv_raw(payload)
        self.assertEqual(result["status"], STATUS_OK)
        self.assertIsNotNone(result["message"])

    def test_validation_error_for_bad_field(self):
        """A JSON message with an invalid field → STATUS_VALIDATION_ERROR."""
        msg = _valid_message()
        msg["decision"] = "INVALID"
        payload = json.dumps(msg).encode("utf-8")
        result = self._recv_raw(payload)
        self.assertEqual(result["status"], STATUS_VALIDATION_ERROR)
        self.assertIsNone(result["message"])


# ---------------------------------------------------------------------------
# E. Localhost round-trip integration tests (ephemeral port)
# ---------------------------------------------------------------------------

class TestUdpRoundTrip(unittest.TestCase):
    """
    Actual localhost UDP sender → receiver round-trips.

    Uses an OS-allocated ephemeral port to avoid conflicts with port 5005
    or any other process. The receiver socket is bound once per test and
    reused to avoid repeated bind/unbind overhead.
    """

    def setUp(self):
        self.port = _find_free_udp_port()
        self.host = "127.0.0.1"
        self.sender = UdpSender(host=self.host, port=self.port, timeout_s=2.0)
        self.receiver = UdpReceiver(host=self.host, port=self.port, timeout_s=2.0)

        # Pre-bind the receiver socket for all tests in this class
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(2.0)
        self._sock.bind((self.host, self.port))

    def tearDown(self):
        self._sock.close()

    def _roundtrip(self, msg: dict) -> dict:
        """Send msg and receive on the pre-bound socket."""
        send_result = self.sender.send_decision(msg)
        self.assertTrue(send_result["success"], f"Send failed: {send_result['error']}")
        return self.receiver.receive_one_on_socket(self._sock)

    def test_19_accept_message_round_trip(self):
        """ACCEPT message travels from sender to receiver intact."""
        msg = _valid_message(
            class_id=4, class_name="plastic", decision="ACCEPT",
            source="classifier", confidence=0.91, entropy=0.42,
            override=False,
        )
        recv_result = self._roundtrip(msg)
        self.assertEqual(recv_result["status"], STATUS_OK)
        received = recv_result["message"]
        self.assertEqual(received["class_id"], 4)
        self.assertEqual(received["class_name"], "plastic")
        self.assertEqual(received["decision"], "ACCEPT")
        self.assertEqual(received["source"], "classifier")
        self.assertAlmostEqual(received["confidence"], 0.91, places=5)
        self.assertAlmostEqual(received["entropy"], 0.42, places=5)
        self.assertFalse(received["override"])

    def test_04_uncertain_message_round_trip(self):
        """UNCERTAIN unresolved message round-trips correctly."""
        msg = build_message(
            class_id=0, class_name="cardboard", decision="UNCERTAIN",
            source="unresolved", confidence=None, entropy=None, override=False,
        )
        recv_result = self._roundtrip(msg)
        self.assertEqual(recv_result["status"], STATUS_OK)
        received = recv_result["message"]
        self.assertEqual(received["decision"], "UNCERTAIN")
        self.assertIsNone(received["confidence"])
        self.assertIsNone(received["entropy"])
        self.assertFalse(received["override"])

    def test_05_human_gesture_override_round_trip(self):
        """Human gesture override message round-trips correctly."""
        msg = build_message(
            class_id=2, class_name="metal", decision="UNCERTAIN",
            source="human_gesture", confidence=0.52, entropy=1.05, override=True,
        )
        recv_result = self._roundtrip(msg)
        self.assertEqual(recv_result["status"], STATUS_OK)
        received = recv_result["message"]
        self.assertEqual(received["source"], "human_gesture")
        self.assertTrue(received["override"])
        self.assertEqual(received["class_name"], "metal")

    def test_06_unresolved_message_round_trip(self):
        """Unresolved message round-trips correctly."""
        msg = build_message(
            class_id=5, class_name="trash", decision="UNCERTAIN",
            source="unresolved", confidence=None, entropy=None, override=False,
        )
        recv_result = self._roundtrip(msg)
        self.assertEqual(recv_result["status"], STATUS_OK)
        self.assertEqual(recv_result["message"]["source"], "unresolved")

    def test_20_multiple_sequential_messages(self):
        """Five messages sent and received sequentially all return STATUS_OK."""
        class_cycle = [
            (0, "cardboard"), (1, "glass"), (2, "metal"),
            (3, "paper"), (4, "plastic"),
        ]
        for class_id, class_name in class_cycle:
            msg = build_message(
                class_id=class_id, class_name=class_name,
                decision="ACCEPT", source="classifier",
                confidence=0.88, entropy=0.35, override=False,
            )
            recv_result = self._roundtrip(msg)
            self.assertEqual(
                recv_result["status"], STATUS_OK,
                f"Failed for class {class_name}: {recv_result['error']}",
            )
            self.assertEqual(recv_result["message"]["class_id"], class_id)

    def test_semantic_preservation(self):
        """
        Receiver must return the same semantic values the sender transmitted.
        Validates the protocol safety rule: no field is reinterpreted.
        """
        msg = build_message(
            class_id=4, class_name="plastic",
            decision="ACCEPT", source="classifier",
            confidence=0.94, entropy=0.28, override=False,
        )
        recv_result = self._roundtrip(msg)
        received = recv_result["message"]
        # All semantic fields must be byte-for-byte equal after JSON round-trip
        self.assertEqual(received["class_id"], msg["class_id"])
        self.assertEqual(received["class_name"], msg["class_name"])
        self.assertEqual(received["decision"], msg["decision"])
        self.assertEqual(received["source"], msg["source"])
        self.assertAlmostEqual(received["confidence"], msg["confidence"], places=8)
        self.assertAlmostEqual(received["entropy"], msg["entropy"], places=8)
        self.assertEqual(received["override"], msg["override"])


# ---------------------------------------------------------------------------
# F. Receiver timeout test (ephemeral port, no sender)
# ---------------------------------------------------------------------------

class TestUdpReceiverTimeout(unittest.TestCase):

    def test_21_receiver_times_out_gracefully(self):
        """Receiver returns STATUS_TIMEOUT when no datagram arrives."""
        port = _find_free_udp_port()
        receiver = UdpReceiver(host="127.0.0.1", port=port, timeout_s=0.1)
        t0 = time.monotonic()
        result = receiver.receive_one()
        elapsed = time.monotonic() - t0
        self.assertEqual(result["status"], STATUS_TIMEOUT)
        self.assertIsNone(result["message"])
        # Should time out in roughly 0.1 s, give 1 s of margin
        self.assertLess(elapsed, 1.0)


# ---------------------------------------------------------------------------
# G. Regression: Phase 1 + Phase 2 + Phase 3 smoke imports
# ---------------------------------------------------------------------------

class TestRegressionImports(unittest.TestCase):

    def test_phase1_classifier_imports(self):
        from src.realtime.realtime_classifier import WasteClassifier, DEFAULT_CLASSES
        self.assertEqual(len(DEFAULT_CLASSES), 6)

    def test_phase2_confidence_gate_imports(self):
        from src.reliability.confidence_gate import ConfidenceGate
        gate = ConfidenceGate()
        self.assertAlmostEqual(gate.confidence_threshold, 0.70)

    def test_phase2_ood_filter_imports(self):
        from src.reliability.ood_filter import OodFilter
        self.assertIsNotNone(OodFilter())

    def test_phase3_gesture_detector_imports(self):
        from src.gesture.gesture_detector import GESTURE_CLASS_NAMES, FINGER_COUNT_TO_CLASS_ID
        self.assertEqual(len(GESTURE_CLASS_NAMES), 6)

    def test_phase3_gesture_supervisor_imports(self):
        from src.gesture.gesture_supervisor import GestureSupervisor, SOURCE_CLASSIFIER
        self.assertEqual(SOURCE_CLASSIFIER, "classifier")


if __name__ == "__main__":
    unittest.main(verbosity=2)
