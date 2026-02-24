import time
import hashlib
import json
from app.core.logger import get_logger

log = get_logger(__name__)

class SimpleCache:
    """
    In-memory TTL cache.
    Tradeoff: fast, zero dependencies, but lost on restart
    and not shared across multiple workers.
    For multi-worker production: replace with Redis.
    """
    def __init__(self, ttl_seconds: int = 3600):
        self._store: dict = {}
        self._ttl         = ttl_seconds
        self._hits        = 0
        self._misses      = 0

    def _make_key(self, *args, **kwargs) -> str:
        raw = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, key: str):
        entry = self._store.get(key)
        if entry and time.time() - entry['ts'] < self._ttl:
            self._hits += 1
            log.info("cache_hit", key=key[:8])
            return entry['data']
        self._misses += 1
        return None

    def set(self, key: str, data):
        self._store[key] = {'data': data, 'ts': time.time()}

    def invalidate(self):
        self._store.clear()
        log.info("cache_cleared")

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            'hits':      self._hits,
            'misses':    self._misses,
            'hit_rate':  round(self._hits / total, 3) if total else 0,
            'size':      len(self._store),
        }

cache = SimpleCache(ttl_seconds=3600)