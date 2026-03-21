"""
Disk Cache
----------
Persists fetched data to app/data/cache/ as JSON files.
Each key maps to a .json file. TTL is checked on read.

Sits alongside the existing in-memory SimpleCache in app/core/cache.py —
this handles cross-restart persistence specifically for yfinance data.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from app.core.logger import get_logger

log = get_logger(__name__)


class DiskCache:
    """
    Simple file-based cache with TTL.

    Layout:
        app/data/cache/
            AAPL_20240101.json
            MSFT_20240101.json
            ...

    Each file contains:
        {
            "ts": <unix timestamp of write>,
            "data": { ... }
        }
    """

    def __init__(self, cache_dir: str = "app/data/cache", ttl_hours: int = 24):
        self._dir     = Path(cache_dir)
        self._ttl_s   = ttl_hours * 3600
        self._dir.mkdir(parents=True, exist_ok=True)
        log.info(f"DiskCache initialised at {self._dir} (TTL={ttl_hours}h)")

    def _path(self, key: str) -> Path:
        # Sanitise key to be filesystem-safe
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe_key}.json"

    def get(self, key: str) -> Optional[Any]:
        """Return cached data if present and fresh, else None."""
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
            age = time.time() - entry["ts"]
            if age > self._ttl_s:
                log.info(f"disk_cache_stale  {key} (age={age/3600:.1f}h)")
                return None
            return entry["data"]
        except Exception as e:
            log.warning(f"disk_cache_read_error  {key}: {e}")
            return None

    def set(self, key: str, data: Any) -> None:
        """Write data to disk cache."""
        path = self._path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"ts": time.time(), "data": data}, f, default=str)
        except Exception as e:
            log.warning(f"disk_cache_write_error  {key}: {e}")

    def invalidate(self, key: str) -> None:
        """Delete a single cache entry."""
        path = self._path(key)
        if path.exists():
            path.unlink()
            log.info(f"disk_cache_invalidated  {key}")

    def clear_all(self) -> None:
        """Delete all cache files."""
        for f in self._dir.glob("*.json"):
            f.unlink()
        log.info("disk_cache_cleared")

    def stats(self) -> dict:
        files      = list(self._dir.glob("*.json"))
        now        = time.time()
        fresh      = sum(
            1 for f in files
            if now - json.load(open(f))["ts"] <= self._ttl_s
        )
        return {
            "total_files": len(files),
            "fresh":       fresh,
            "stale":       len(files) - fresh,
            "cache_dir":   str(self._dir),
            "ttl_hours":   self._ttl_s / 3600,
        }