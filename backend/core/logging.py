"""
Phase 5 - structured (JSON-lines) request logging.
AI Fraud-Spike & Risk Detection System (Razorpay Buildathon, Track 02)

No print statements anywhere in backend/. Every HTTP request produces exactly
one JSON log line with: timestamp, endpoint, method, status_code, latency_ms,
and - for /transactions/score - operating_mode and risk_band. No PII or raw
transaction identifiers (nameOrig/nameDest) are logged; only the
non-sensitive decision fields already present in the risk_engine response.
"""
import json
import logging
import sys

LOG_RECORD_EXTRA_FIELDS = (
    "endpoint", "method", "status_code", "latency_ms", "operating_mode", "risk_band",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in LOG_RECORD_EXTRA_FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload)


_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    root = logging.getLogger("backend")
    root.setLevel(level)
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str = "backend") -> logging.Logger:
    return logging.getLogger("backend" if name == "backend" else f"backend.{name}")
