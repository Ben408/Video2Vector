"""Optional SSL workaround for corporate/broken CA stores on Windows."""
from __future__ import annotations

import os
import ssl


def allow_insecure_downloads_if_needed() -> None:
    if os.environ.get("VIDEO2VECTOR_INSECURE_SSL", "1") == "1":
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except Exception:
            pass
