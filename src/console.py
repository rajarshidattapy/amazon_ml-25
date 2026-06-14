"""Console compatibility helpers."""

import sys


def configure_utf8_output() -> None:
    """Prefer UTF-8 console output when the host stream supports it."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
