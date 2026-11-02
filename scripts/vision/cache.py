"""Local caches that make repeat work free: match videos and Gemini answers.

    ~/.cache/spartans-vision/videos/<youtube id>/   downloaded sections, with an index of the seconds each one covers
    ~/.cache/spartans-vision/gemini/<ab>/<hash>.json  one successful Gemini answer per exact request

Videos: a run asks for seconds [start, end] of a YouTube video; any cached file that covers them is reused (an auto
run after a scouting run of the same match downloads nothing). Gemini: the key is a hash of everything that shapes the
answer (model, thinking level, media resolution, schema, token limit, and every part, with image bytes hashed), so a
re-run of the same match with the same frames returns the saved answers instead of paying for them again; any change
to a prompt, a frame or the model is a different key. Only parsed, successful answers are stored.

VISION_CACHE=0 turns both off. VISION_CACHE_DIR moves them. VISION_VIDEO_CACHE_GB caps the video cache (default 20).
"""
from contextlib import contextmanager
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None
from common import valid_youtube_id

_lock = threading.RLock()
_held_file_locks: dict[str, list[int]] = {}


def enabled():
    return os.environ.get('VISION_CACHE', '1') not in ('0', 'false', 'no')


def root():
    return Path(os.environ.get('VISION_CACHE_DIR') or Path.home() / '.cache' / 'spartans-vision')


@contextmanager
def _file_lock(name: str = 'cache.lock'):
    """Acquire the in-process RLock and a cross-process POSIX flock on root() / name."""
    with _lock:
        if fcntl is None:
            yield
            return
        lock_dir = root()
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_key = str((lock_dir / name).resolve())
        state = _held_file_locks.get(lock_key)
        if state is not None:
            state[1] += 1
            try:
                yield
            finally:
                state[1] -= 1
            return
        fd = os.open(str(lock_dir / name), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            _held_file_locks[lock_key] = [fd, 1]
            yield
        finally:
            _held_file_locks.pop(lock_key, None)
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass


# ---- Gemini answers

def _canonical_part(part):
    if 'image' in part:
        return {**{k: v for k, v in part.items() if k != 'image'},
                'mime': part.get('mime', 'image/jpeg'),
                'image_sha256': hashlib.sha256(part['image']).hexdigest()}
    return part


def request_key(model, parts, schema, **settings):
    """Stable hash of a generate_json request. Images are hashed by content, so identical frames hit."""
    payload = {'model': model, 'schema': schema, 'settings': settings, 'parts': [_canonical_part(p) for p in parts]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _answer_path(key):
    return root() / 'gemini' / key[:2] / f'{key}.json'


def get_answer(key):
    if not enabled():
        return None
    path = _answer_path(key)
    try:
        raw = path.read_text()
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    try:
        os.utime(path)
    except OSError:
        pass
    return data


def put_answer(key, value):
    if not enabled():
        return
    _maybe_prune_answers()
    path = _answer_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value))
    tmp.replace(path)


_last_answer_prune = [0.0]


def _maybe_prune_answers(every_s=600):
    """At most every ten minutes, keep the answer cache under VISION_ANSWER_CACHE_GB (default 2), oldest first."""
    with _lock:
        now = time.time()
        if now - _last_answer_prune[0] < every_s:
            return
        _last_answer_prune[0] = now
        prune_answers()


def _stat_entries(paths):
    """Snapshot (path, last_used, size) for each path, skipping files another worker removed."""
    entries = []
    for path in paths:
        try:
            st = path.stat()
            entries.append((path, max(st.st_atime, st.st_mtime), st.st_size))
        except OSError:
            continue
    return entries


def prune_answers(limit_gb=None):
    limit = float(limit_gb if limit_gb is not None else os.environ.get('VISION_ANSWER_CACHE_GB', 2)) * 1e9
    base = root() / 'gemini'
    with _lock:
        if not base.exists():
            return
        entries = _stat_entries(base.glob('*/*.json'))
        total = sum(size for _, _, size in entries)
        for path, _, size in sorted(entries, key=lambda item: item[1]):
            if total <= limit:
                break
            total -= size
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


# ---- videos

def _write_atomic(path, text):
    """Write then rename, so another worker process never reads a half-written index."""
    tmp = path.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def _video_dir(youtube_id):
    return root() / 'videos' / valid_youtube_id(youtube_id)


def _index(youtube_id):
    folder = _video_dir(youtube_id)
    try:
        return json.loads((folder / 'index.json').read_text())
    except (OSError, ValueError):
        return []


def find_video(youtube_id, start_s, end_s, max_height, exact=False):
    """(path, offset_s) of a cached file covering [start_s, end_s] at >= max_height, else None."""
    youtube_id = valid_youtube_id(youtube_id)
    if not enabled():
        return None
    folder = _video_dir(youtube_id)
    folder_resolved = folder.resolve()
    candidates = []
    for entry in _index(youtube_id):
        name = entry.get('file') if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name or name in ('.', '..') or Path(name).name != name:
            continue
        path = folder / name
        try:
            if path.resolve().parent != folder_resolved:
                continue
        except OSError:
            continue
        covers = (entry['start_s'] == start_s and entry['end_s'] == end_s and entry['offset_s'] == start_s) if exact else (entry['start_s'] <= start_s and end_s <= entry['end_s'])
        if entry['max_height'] < max_height or not covers:
            continue
        try:
            if not path.exists() or path.stat().st_size <= 0:
                continue
        except OSError:
            continue
        candidates.append((entry, path))
    if not exact and len(candidates) > 1:
        candidates.sort(key=lambda item: (float(item[0]['end_s']) - float(item[0]['start_s']), -float(item[0].get('saved_at', 0.0))))
    for entry, path in candidates:
        try:
            os.utime(path)                              # recently used: pruned last
        except OSError:
            continue
        return path, entry['offset_s']
    return None


def forget_video(youtube_id, path):
    """Drop one cached file from the index (it stays on disk until pruned): used when it turns out not to decode."""
    youtube_id = valid_youtube_id(youtube_id)
    if not enabled():
        return
    name = Path(path).name
    with _file_lock('videos.lock'):
        folder = _video_dir(youtube_id)
        entries = [e for e in _index(youtube_id) if e['file'] != name]
        if folder.exists():
            _write_atomic(folder / 'index.json', json.dumps(entries, indent=1))


_last_video_prune = [0.0]


def _maybe_prune_videos(every_s=600):
    """At most every ten minutes, keep the video cache under VISION_VIDEO_CACHE_GB (default 20), oldest first."""
    with _lock:
        now = time.time()
        if now - _last_video_prune[0] < every_s:
            return
        _last_video_prune[0] = now
    prune_videos()


def store_video(youtube_id, path, offset_s, start_s, end_s, max_height):
    """Move a downloaded file into the cache and record the seconds it covers. Returns the cached path."""
    youtube_id = valid_youtube_id(youtube_id)
    if not enabled():
        return Path(path)
    folder = _video_dir(youtube_id)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / Path(path).name
    with _file_lock('videos.lock'):
        if Path(path).resolve() != target.resolve():
            tmp = target.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp')
            shutil.move(str(path), tmp)
            tmp.replace(target)
        try:
            os.utime(target)
        except OSError:
            pass
        entries = [e for e in _index(youtube_id) if e['file'] != target.name]
        entries.append({'file': target.name, 'offset_s': offset_s, 'start_s': start_s, 'end_s': end_s, 'max_height': max_height, 'saved_at': time.time()})
        _write_atomic(folder / 'index.json', json.dumps(entries, indent=1))
    _maybe_prune_videos()
    return target


def prune_videos(limit_gb=None):
    """Keep the video cache under its size cap, removing the least recently used files first."""
    limit = float(limit_gb if limit_gb is not None else os.environ.get('VISION_VIDEO_CACHE_GB', 20)) * 1e9
    base = root() / 'videos'
    with _file_lock('videos.lock'):
        _last_video_prune[0] = time.time()
        if not base.exists():
            return
        entries = _stat_entries(p for p in base.glob('*/*') if p.suffix in ('.mp4', '.mkv', '.webm'))
        total = sum(size for _, _, size in entries)
        for path, _, size in sorted(entries, key=lambda item: item[1]):
            if total <= limit:
                break
            total -= size
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            index = path.parent / 'index.json'
            try:
                kept = [e for e in json.loads(index.read_text()) if e['file'] != path.name]
                _write_atomic(index, json.dumps(kept, indent=1))
            except (OSError, ValueError):
                pass
