"""
Tests for app/core/cache.py (SimpleCache) and app/core/disk_cache.py (DiskCache)
"""

import pytest
import time
from unittest.mock import patch, MagicMock

# Patch logger before importing to avoid structlog kwarg issues
with patch('app.core.cache.log'), patch('app.core.disk_cache.log'):
    from app.core.cache import SimpleCache
    from app.core.disk_cache import DiskCache


# ── SimpleCache fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def cache():
    with patch('app.core.cache.log'):
        return SimpleCache(ttl_seconds=60)


@pytest.fixture
def short_ttl_cache():
    with patch('app.core.cache.log'):
        return SimpleCache(ttl_seconds=1)


# ── SimpleCache tests ─────────────────────────────────────────────────────────

def test_cache_set_and_get(cache):
    """set then get should return the same value."""
    with patch('app.core.cache.log'):
        cache.set('key1', {'data': 42})
        result = cache.get('key1')
    assert result == {'data': 42}


def test_cache_miss_returns_none(cache):
    """get on missing key should return None."""
    with patch('app.core.cache.log'):
        assert cache.get('nonexistent') is None


def test_cache_ttl_expiry(short_ttl_cache):
    """Entry should expire after TTL seconds."""
    with patch('app.core.cache.log'):
        short_ttl_cache.set('key1', 'value')
        time.sleep(1.1)
        assert short_ttl_cache.get('key1') is None


def test_cache_ttl_not_expired(cache):
    """Entry should still be available before TTL expires."""
    with patch('app.core.cache.log'):
        cache.set('key1', 'value')
        assert cache.get('key1') == 'value'


def test_cache_overwrite(cache):
    """Setting same key twice should overwrite."""
    with patch('app.core.cache.log'):
        cache.set('key1', 'first')
        cache.set('key1', 'second')
        assert cache.get('key1') == 'second'


def test_cache_invalidate_clears_all(cache):
    """invalidate should remove all entries."""
    with patch('app.core.cache.log'):
        cache.set('key1', 'a')
        cache.set('key2', 'b')
        cache.invalidate()
        assert cache.get('key1') is None
        assert cache.get('key2') is None


def test_cache_stats_hits_and_misses(cache):
    """stats should track hits and misses correctly."""
    with patch('app.core.cache.log'):
        cache.set('key1', 'value')
        cache.get('key1')     # hit
        cache.get('key1')     # hit
        cache.get('missing')  # miss
    stats = cache.stats
    assert stats['hits']   == 2
    assert stats['misses'] == 1


def test_cache_stats_hit_rate(cache):
    """hit_rate should be hits / total."""
    with patch('app.core.cache.log'):
        cache.set('key1', 'value')
        cache.get('key1')    # hit
        cache.get('missing') # miss
    assert cache.stats['hit_rate'] == 0.5


def test_cache_stats_size(cache):
    """stats size should reflect number of stored entries."""
    with patch('app.core.cache.log'):
        cache.set('key1', 'a')
        cache.set('key2', 'b')
    assert cache.stats['size'] == 2


def test_cache_stores_various_types(cache):
    """cache should store lists, dicts, strings, and numbers."""
    with patch('app.core.cache.log'):
        cache.set('list',   [1, 2, 3])
        cache.set('dict',   {'a': 1})
        cache.set('string', 'hello')
        cache.set('number', 3.14)
        assert cache.get('list')   == [1, 2, 3]
        assert cache.get('dict')   == {'a': 1}
        assert cache.get('string') == 'hello'
        assert cache.get('number') == 3.14


def test_cache_make_key_consistent(cache):
    """Same args should produce same cache key."""
    k1 = cache._make_key('a', 'b', x=1)
    k2 = cache._make_key('a', 'b', x=1)
    assert k1 == k2


def test_cache_make_key_different_args(cache):
    """Different args should produce different keys."""
    k1 = cache._make_key('a', x=1)
    k2 = cache._make_key('a', x=2)
    assert k1 != k2


# ── DiskCache fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def disk_cache(tmp_path):
    with patch('app.core.disk_cache.log'):
        return DiskCache(cache_dir=str(tmp_path / 'cache'), ttl_hours=1)


@pytest.fixture
def short_ttl_disk_cache(tmp_path):
    with patch('app.core.disk_cache.log'):
        return DiskCache(cache_dir=str(tmp_path / 'cache_short'), ttl_hours=0.0003)


# ── DiskCache tests ───────────────────────────────────────────────────────────

def test_disk_cache_set_and_get(disk_cache):
    """set then get should return the same value."""
    disk_cache.set('AAPL_20240101', {'pe_ratio': 28.0})
    assert disk_cache.get('AAPL_20240101') == {'pe_ratio': 28.0}


def test_disk_cache_miss_returns_none(disk_cache):
    """get on missing key should return None."""
    assert disk_cache.get('nonexistent') is None


def test_disk_cache_creates_file(disk_cache, tmp_path):
    """set should create a JSON file on disk."""
    disk_cache.set('AAPL_20240101', {'data': 1})
    files = list((tmp_path / 'cache').glob('*.json'))
    assert len(files) == 1


def test_disk_cache_ttl_expiry(short_ttl_disk_cache):
    """Entry should expire after TTL."""
    short_ttl_disk_cache.set('key1', 'value')
    time.sleep(1.2)
    assert short_ttl_disk_cache.get('key1') is None


def test_disk_cache_invalidate_single(disk_cache):
    """invalidate should remove a single key's file."""
    disk_cache.set('key1', 'a')
    disk_cache.set('key2', 'b')
    disk_cache.invalidate('key1')
    assert disk_cache.get('key1') is None
    assert disk_cache.get('key2') == 'b'


def test_disk_cache_clear_all(disk_cache, tmp_path):
    """clear_all should remove all cache files."""
    disk_cache.set('key1', 'a')
    disk_cache.set('key2', 'b')
    disk_cache.clear_all()
    assert len(list((tmp_path / 'cache').glob('*.json'))) == 0


def test_disk_cache_stats(disk_cache):
    """stats should return correct file counts."""
    disk_cache.set('key1', 'a')
    disk_cache.set('key2', 'b')
    stats = disk_cache.stats()
    assert stats['total_files'] == 2
    assert stats['fresh']       == 2
    assert stats['stale']       == 0


def test_disk_cache_stale_in_stats(short_ttl_disk_cache):
    """Expired files should appear as stale in stats."""
    short_ttl_disk_cache.set('key1', 'a')
    time.sleep(1.2)
    stats = short_ttl_disk_cache.stats()
    assert stats['stale'] == 1
    assert stats['fresh'] == 0


def test_disk_cache_overwrites_existing(disk_cache):
    """Setting same key twice should overwrite."""
    disk_cache.set('key1', 'first')
    disk_cache.set('key1', 'second')
    assert disk_cache.get('key1') == 'second'


def test_disk_cache_stores_nested_dict(disk_cache):
    """DiskCache should handle nested dicts correctly."""
    data = {'ticker': 'AAPL', 'metrics': {'pe': 28.0, 'beta': 1.2}}
    disk_cache.set('complex', data)
    assert disk_cache.get('complex') == data


def test_disk_cache_key_sanitization(disk_cache):
    """Keys with slashes should be sanitized for filesystem."""
    disk_cache.set('key/with/slash', 'value')
    assert disk_cache.get('key/with/slash') == 'value'


def test_disk_cache_creates_dir_if_missing(tmp_path):
    """DiskCache should create cache directory if it does not exist."""
    new_dir = tmp_path / 'new' / 'nested' / 'cache'
    with patch('app.core.disk_cache.log'):
        DiskCache(cache_dir=str(new_dir), ttl_hours=1)
    assert new_dir.exists()


def test_disk_cache_handles_corrupt_file(disk_cache, tmp_path):
    """Corrupt JSON file should return None gracefully."""
    corrupt = (tmp_path / 'cache' / 'corrupt.json')
    corrupt.write_text('{ not valid json }')
    assert disk_cache.get('corrupt') is None