"""Getting video: download only the seconds that are needed, and find when the match starts."""
import concurrent.futures as cf
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cache
from common import AUTO_END_MS, AUTO_GRACE_MS, log, valid_youtube_id

TIMING_SCHEMA = {'type': 'object', 'properties': {'match_start_s': {'type': 'number'}, 'match_end_s': {'type': 'number'}, 'evidence': {'type': 'string'}},
                 'required': ['match_start_s', 'match_end_s']}
PARALLEL_SPLIT_THRESHOLD_S = 60.0
AUTO_PART_SECONDS = 26.0
PARALLEL_DOWNLOAD_WORKERS = 3
VIDEO_END_SLACK_S = 3.0      # a part this much shorter than requested means the video ended inside it
YTDLP_TIMEOUT_S = 600
FFMPEG_TIMEOUT_S = 180
MAX_FULL_VIDEO_FALLBACK_S = 1800.0
SAFE_MEDIA_ENV_KEYS = ('PATH', 'HOME', 'TMPDIR', 'TEMP', 'TMP', 'LANG', 'LC_ALL', 'SSL_CERT_FILE', 'SSL_CERT_DIR', 'REQUESTS_CA_BUNDLE', 'SYSTEMROOT')


def _media_env():
    """Minimal allowlisted environment for yt-dlp / node / ffmpeg subprocesses (no API keys or runner tokens)."""
    return {k: os.environ[k] for k in SAFE_MEDIA_ENV_KEYS if k in os.environ}


def _run_media_cmd(cmd, timeout):
    env = _media_env()
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except TypeError:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


_YTDLP_VERSIONS: dict[str, str] = {}


def managed_ytdlp_path() -> Path:
    """Managed yt-dlp binary inside the vision cache venv."""
    return cache.root() / 'ytenv' / 'bin' / 'yt-dlp'


def ytdlp_binary() -> str | None:
    """yt-dlp to run: $YTDLP when set, then ~/.cache/spartans-vision/ytenv/bin/yt-dlp when executable, else PATH."""
    env_bin = os.environ.get('YTDLP')
    if env_bin:
        return env_bin
    managed = managed_ytdlp_path()
    try:
        if managed.is_file() and os.access(managed, os.X_OK):
            return str(managed)
    except OSError as error:  # e.g. the cache's ytenv belongs to another user: use yt-dlp on PATH instead
        log(f'warning: cannot use {managed} ({error}); trying yt-dlp on PATH')
    return shutil.which('yt-dlp')


def ytdlp_version(binary=None) -> str:
    """Version string for a yt-dlp binary (cached by binary path), or 'unknown' on failure."""
    target = binary or ytdlp_binary()
    if not target:
        return 'unknown'
    key = str(target)
    if key in _YTDLP_VERSIONS:
        return _YTDLP_VERSIONS[key]
    try:
        run = subprocess.run([key, '--version'], capture_output=True, text=True, timeout=5)
        version = (run.stdout or '').strip().splitlines()[0].strip() if run.returncode == 0 and (run.stdout or '').strip() else 'unknown'
    except (OSError, subprocess.SubprocessError):
        version = 'unknown'
    _YTDLP_VERSIONS[key] = version
    return version


def extract_ytdlp_error(output: str, limit: int = 300) -> str:
    """Keep root-cause ERROR:/HTTP Error/nsig/bot lines instead of letting trailing lines push them out."""
    text = (output or '').strip()
    if not text:
        return ''
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    needles = ('ERROR:', 'HTTP Error', 'Sign in to confirm', 'nsig', 'Signature extraction')
    needles_lower = tuple(n.lower() for n in needles)
    matched = [line for line in lines if any(n in line.lower() for n in needles_lower)]
    if matched:
        picked = list(matched[-2:])
        if lines[-1] not in picked:
            picked.append(lines[-1])
        joined = ' | '.join(picked)
        if len(joined) <= limit:
            return joined
        return ' | '.join(matched[-2:])[:limit]
    return text[-limit:]


def is_permanent_ytdlp_error(error_text: str) -> bool:
    """True when retrying uncut or full-video modes on the same YouTube video will fail the same way."""
    text = str(error_text or '').lower()
    if not text or 'ffmpeg' in text or 'concat' in text:
        return False
    permanent_patterns = (
        'http error 403',
        '403: forbidden',
        '403 forbidden',
        'sign in to confirm',
        'not a bot',
        'video unavailable',
        'video is unavailable',        # yt-dlp's wording for a removed TBA video (2026hop_qm110)
        'has been removed',
        'private video',
        'this video is private',
        'nsig extraction failed',
        'signature extraction failed',
    )
    return any(pattern in text for pattern in permanent_patterns)


def format_download_error(youtube_id: str, errors: list[str], binary: str | None = None) -> str:
    """Format a SystemExit message with actionable guidance for YouTube 403 / nsig / bot errors and --file."""
    joined = '; '.join(str(e) for e in errors if e)
    lower = joined.lower()
    hint = ''
    if any(token in lower for token in ('403', 'nsig', 'signature', 'sign in to confirm', 'not a bot')):
        hint = ' YouTube returned HTTP 403 (check yt-dlp in ~/.cache/spartans-vision/ytenv or run release_worker.sh --ensure-ytdlp).'
    return (
        f'could not download the video ({joined}).{hint} '
        'YouTube sometimes refuses automated downloads: download it yourself and pass --file clip.mp4 '
        '(with --file-offset-s if it is a clip).'
    )


def ffmpeg_binary():
    """ffmpeg to run: $FFMPEG when set, else the one on PATH."""
    return os.environ.get('FFMPEG') or shutil.which('ffmpeg')


def video_format(max_height):
    """yt-dlp format choice, H.264 first: OpenCV on the Spark cannot decode AV1, so a run on it fails at calibration."""
    return (f'bv*[height<={max_height}][vcodec^=avc1][ext=mp4]+ba[ext=m4a]/'
            f'bv*[height<={max_height}][ext=mp4][vcodec!^=av01]+ba[ext=m4a]/'
            f'bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/'
            f'b[height<={max_height}][ext=mp4]/b[height<={max_height}]')


def _ytdlp(youtube_id, out, max_height, extra):
    """One yt-dlp attempt. Returns (ok, error text).

    YouTube now answers 403 to every media URL unless its JavaScript challenge is solved, which needs a JS runtime
    and the bundled yt-dlp-ejs solver package (installed with yt-dlp[default]). An old yt-dlp fails the same way: keep it current."""
    youtube_id = valid_youtube_id(youtube_id)
    max_filesize = os.environ.get('VISION_MAX_VIDEO_FILESIZE', '500M')
    cmd = [ytdlp_binary(), '--no-playlist', '--max-filesize', max_filesize,
           '-f', video_format(max_height),
           '--merge-output-format', 'mp4', '--http-chunk-size', '4M', '--retries', '10', '--fragment-retries', '10', *extra,
           '-o', str(out), '--', f'https://www.youtube.com/watch?v={youtube_id}']
    if shutil.which('node'):
        cmd[1:1] = ['--js-runtimes', 'node']   # solve YouTube's player challenge using local yt-dlp-ejs
    try:
        run = _run_media_cmd(cmd, YTDLP_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        Path(out).unlink(missing_ok=True)
        return False, f'yt-dlp timed out after {exc.timeout}s'
    ok = run.returncode == 0 and Path(out).exists() and Path(out).stat().st_size > 0
    if not ok:
        Path(out).unlink(missing_ok=True)
    return ok, extract_ytdlp_error(run.stderr or run.stdout)


def _split_long_range(start_s, end_s):
    """Split a long request into an auto-first part ([start, start + 26 s]) plus the rest in two halves."""
    auto_end = min(end_s, start_s + AUTO_PART_SECONDS)
    mid = float(round((auto_end + end_s) / 2.0))
    return [r for r in ((start_s, auto_end), (auto_end, mid), (mid, end_s)) if r[1] > r[0]]


def _download_exact_part(youtube_id, begin, finish, folder, max_height):
    """Download one exact [begin, finish] slice and store it in the cache. Returns Path or raises RuntimeError."""
    youtube_id = valid_youtube_id(youtube_id)
    begin, finish = float(math.floor(begin)), float(math.ceil(finish))
    cached = cache.find_video(youtube_id, begin, finish, max_height, exact=True)
    if cached and cached[1] == begin:
        return Path(cached[0])
    section = ['--download-sections', f'*{begin:.0f}-{finish:.0f}']
    attempts = [
        (f'{youtube_id}_{int(begin)}_{int(finish)}.mp4', section + ['--force-keyframes-at-cuts'], 'cut at keyframes'),
        (f'{youtube_id}_{int(begin)}_{int(finish)}_b.mp4', section, 'uncut'),
    ]
    errors = []
    for name, extra, label in attempts:
        out = Path(folder) / name
        ok, error = _ytdlp(youtube_id, out, max_height, extra)
        if ok:
            return cache.store_video(youtube_id, out, begin, begin, finish, max_height)
        errors.append(f'{label}: {extract_ytdlp_error(error, limit=180)}')
        if is_permanent_ytdlp_error(error):
            break
    raise RuntimeError('; '.join(errors))


def download_sections_parallel(youtube_id, ranges, folder, max_height=720):
    """Download several [start, end] sections with yt-dlp in parallel, cache each one, and return the paths in order."""
    youtube_id = valid_youtube_id(youtube_id)
    if not ranges:
        return []
    if not ytdlp_binary():
        raise SystemExit('yt-dlp is not installed (pip install -U "yt-dlp[default]"), or pass --file with a local copy')

    def one(item):
        idx, (start_s, end_s) = item
        part_dir = Path(folder) / f'part_{idx}'
        part_dir.mkdir(parents=True, exist_ok=True)
        return _download_exact_part(youtube_id, start_s, end_s, part_dir, max_height)

    with cf.ThreadPoolExecutor(PARALLEL_DOWNLOAD_WORKERS) as pool:
        return list(pool.map(one, enumerate(ranges)))


def playable(path, times_s):
    """True when a frame decodes at each of `times_s` (seconds into the file). A stream-copied join of parts that YouTube
    encoded with different parameters stops decoding at the part boundary ("non-existing PPS"), which silently ended
    qm2's run at 94 s of a 163 s match, so joins and long cached files are checked before use."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        for t in times_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t)) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                return False
        return True
    finally:
        cap.release()


def video_seconds(path):
    """Decoded length of a video file in seconds (frames / fps), or None when it cannot be read."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        return frames / fps if cap.isOpened() and fps > 0 and frames > 0 else None
    finally:
        cap.release()


def _trim_past_video_end(parts, ranges):
    """Drop the parts that lie past the end of the video. A request without a known match start asks for the first
    480 s, but most TBA match videos are shorter: the part where the video ends comes back short and later parts come
    back as stubs of about a second (2026hop_qm20: 214.5 s video, part [254, 482] was 0.9 s of AV1), so the join could
    never decode near 480 s and the worker downloaded the whole video again. Returns (parts, ranges, video end in
    seconds of the original or None, whether a later stub confirms the end)."""
    for i, (path, (start, end)) in enumerate(zip(parts, ranges)):
        seconds = video_seconds(path)
        if seconds is None or seconds >= (end - start) - VIDEO_END_SLACK_S:
            continue
        later = [video_seconds(p) for p in parts[i + 1:]]
        confirmed = bool(later) and all(s is None or s <= VIDEO_END_SLACK_S for s in later)
        return parts[:i + 1], ranges[:i + 1], float(start) + seconds, confirmed
    return parts, ranges, None, False


def _join_checkpoints(ranges, begin, finish):
    """File seconds to test in a join: both sides of each part boundary and near the end."""
    points = []
    for start, _ in ranges[1:]:
        boundary = float(start) - float(begin)
        points += [boundary - 1.0, boundary + 0.5, boundary + 3.0]
    points.append(max(0.0, float(finish) - float(begin) - 2.0))
    return [p for p in points if p >= 0]


def _concat_sections(parts, out, reencode=False):
    """Join section files. Stream copy through the concat demuxer (lossless, fast) needs every part encoded the same way;
    when they are not (YouTube re-served a part), the concat FILTER decodes each part with its own decoder and re-encodes
    the video (audio is not needed for vision)."""
    ffmpeg = ffmpeg_binary() or 'ffmpeg'
    if reencode:
        inputs = [arg for p in parts for arg in ('-i', str(Path(p).resolve()))]
        graph = ''.join(f'[{i}:v:0]' for i in range(len(parts))) + f'concat=n={len(parts)}:v=1:a=0[v]'
        cmd = [ffmpeg, '-y', *inputs, '-filter_complex', graph, '-map', '[v]',
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', str(out)]
    else:
        list_file = Path(out).with_suffix('.concat.txt')
        escaped = []
        for p in parts:
            resolved = str(Path(p).resolve())
            if '\n' in resolved or '\r' in resolved:
                raise ValueError(f'unsafe newline in concat part path: {p!r}')
            escaped.append(resolved.replace("'", r"'\''"))
        list_file.write_text(''.join(f"file '{p}'\n" for p in escaped))
        cmd = [ffmpeg, '-y', '-f', 'concat', '-safe', '0', '-i', str(list_file),
               '-fflags', '+genpts', '-c', 'copy', '-avoid_negative_ts', 'make_zero', str(out)]
    try:
        run = _run_media_cmd(cmd, FFMPEG_TIMEOUT_S * (4 if reencode else 1))
    except subprocess.TimeoutExpired as exc:
        Path(out).unlink(missing_ok=True)
        return False, f'ffmpeg timed out after {exc.timeout}s'
    ok = run.returncode == 0 and Path(out).exists() and Path(out).stat().st_size > 0
    if not ok:
        Path(out).unlink(missing_ok=True)
    return ok, (run.stderr or run.stdout)[-300:]


def _download_single_section(youtube_id, start_s, end_s, begin, finish, folder, max_height):
    youtube_id = valid_youtube_id(youtube_id)
    section = ['--download-sections', f'*{begin:.0f}-{finish:.0f}']
    attempts = [
        (f'{youtube_id}_{int(begin)}_{int(finish)}.mp4', section + ['--force-keyframes-at-cuts'], begin, 'the section, cut at keyframes'),
        (f'{youtube_id}_{int(begin)}_{int(finish)}_b.mp4', section, begin, 'the section, uncut'),
    ]
    if finish <= MAX_FULL_VIDEO_FALLBACK_S:
        attempts.append((f'{youtube_id}_full.mp4', [], 0, 'the whole video'))
    errors = []
    for name, extra, offset, label in attempts:
        out = Path(folder) / name
        log(f'downloading {youtube_id} {start_s:.0f}s-{end_s:.0f}s at <= {max_height}p ({label})')
        ok, error = _ytdlp(youtube_id, out, max_height, extra)
        if ok:
            covered_end = finish if extra else 1e9            # a whole-video download covers every later request too
            return cache.store_video(youtube_id, out, offset, offset, covered_end, max_height), offset
        errors.append(f'{label}: {extract_ytdlp_error(error, limit=180)}')
        if is_permanent_ytdlp_error(error):
            log(f'  failed with permanent YouTube error ({error.strip()[-100:]}); not retrying other section modes')
            break
        log(f'  failed ({error.strip()[-100:]}); trying the next way')
    raise SystemExit(format_download_error(youtube_id, errors))


def download_section(youtube_id, start_s, end_s, folder, max_height=720):
    """Get [start_s, end_s] of a YouTube video at <= max_height. Returns (path, seconds of the original before the file starts).

    Fetching only the match (or only the 23 s of auto) instead of a whole stream VOD is the biggest saving for
    the local path. Requests longer than 60 s are split into an auto-first part ([start, start + 26 s]) plus the
    rest in two parallel parts and joined losslessly with ffmpeg concat. Sectioned downloads hand the cut to
    ffmpeg, which fails on some streams (one earlier run died with ffmpeg exit 8), so this falls back instead of
    giving up: a single section cut at keyframes, the same section uncut, then the whole video.
    """
    youtube_id = valid_youtube_id(youtube_id)
    begin, finish = float(max(0, math.floor(start_s - 2))), float(math.ceil(end_s + 2))
    cached = cache.find_video(youtube_id, begin, finish, max_height)
    if cached:
        path, offset = cached
        span = (begin - offset, (begin + finish) / 2 - offset, finish - offset - 2)
        if finish - begin <= PARALLEL_SPLIT_THRESHOLD_S or playable(path, span):
            log(f'video {youtube_id} {start_s:.0f}s-{end_s:.0f}s from the local cache')
            return cached
        log(f'  cached {Path(path).name} does not decode through {start_s:.0f}s-{end_s:.0f}s; dropping it and downloading again')
        cache.forget_video(youtube_id, path)
    if not ytdlp_binary():
        raise SystemExit('yt-dlp is not installed (pip install -U "yt-dlp[default]"), or pass --file with a local copy')
    if end_s - start_s > PARALLEL_SPLIT_THRESHOLD_S:
        ranges = _split_long_range(begin, finish)
        log(f'downloading {youtube_id} {start_s:.0f}s-{end_s:.0f}s at <= {max_height}p in {len(ranges)} parallel parts')
        try:
            parts = download_sections_parallel(youtube_id, ranges, folder, max_height)
            parts, ranges, video_end, confirmed = _trim_past_video_end(parts, ranges)
            covered_end = finish
            if video_end is not None:
                log(f'  the video ends at {video_end:.1f}s, before the {finish:.0f}s requested; joining the '
                    f'{len(parts)} part(s) up to its end')
                # A stub after the short part confirms the video really ends there, so nothing later exists to fetch.
                covered_end = finish if confirmed else video_end
            joined = Path(folder) / f'{youtube_id}_{int(begin)}_{int(finish)}_joined.mp4'
            checkpoints = _join_checkpoints(ranges, begin, video_end if video_end is not None else finish)
            ok, error = _concat_sections(parts, joined)
            if ok and not playable(joined, checkpoints):
                log('  the stream-copied join does not decode past a part boundary (parts encoded differently); re-encoding the join')
                ok, error = _concat_sections(parts, joined, reencode=True)
                ok = ok and playable(joined, checkpoints)
                error = error if ok else 'the re-encoded join does not decode either'
            if ok:
                return cache.store_video(youtube_id, joined, begin, begin, covered_end, max_height), begin
            log(f'  parallel join failed ({error.strip()[-100:]}); falling back to a single download')
        except Exception as error:  # noqa: BLE001
            if is_permanent_ytdlp_error(str(error)):
                raise SystemExit(format_download_error(youtube_id, [str(error)])) from error
            log(f'  parallel download failed ({str(error)[-100:]}); falling back to a single download')
    return _download_single_section(youtube_id, start_s, end_s, begin, finish, folder, max_height)


def find_match_window(client, youtube_id, search_end_s=900):
    """Ask a cheap VLM when the match starts and ends (video seconds). Low resolution, 1 fps is plenty
    to read the field-timer overlay, so this costs a few cents on the cheap tier. Returns (start, end)."""
    youtube_id = valid_youtube_id(youtube_id)
    data = client.generate_json([
        {'youtube': f'https://www.youtube.com/watch?v={youtube_id}', 'start': 0, 'end': search_end_s, 'fps': 1},
        {'text': ('FRC match broadcast. Watch the robots on the field carpet (do not read any broadcast overlay). '
                  'In video seconds from the start of this clip: when does the autonomous period begin (robots on the '
                  'field start moving from their starting lines and launching fuel) and when does the match end (robots '
                  'stop moving on the field at the end of teleop/endgame)? Be precise to the second and describe the '
                  'field action you observed in evidence.')}], TIMING_SCHEMA, max_tokens=800, media_resolution='low')
    return float(data['match_start_s']), float(data['match_end_s']), data.get('evidence', '')


def plausible_match(start, end):
    """A match is about 2:40 including the 3 s transition; anything far off means the match window is wrong."""
    return 130 <= end - start <= 200


def auto_window(start_s):
    return start_s, start_s + (AUTO_END_MS + AUTO_GRACE_MS) / 1000.0


def tmpdir(prefix='vision_'):
    return tempfile.mkdtemp(prefix=prefix)
