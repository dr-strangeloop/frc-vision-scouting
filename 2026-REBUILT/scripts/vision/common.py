"""Shared helpers: environment, field constants, retries with backoff, TBA."""
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FIELD = json.loads((ROOT / 'src/lib/vision/data/rebuiltField.json').read_text())
YOUTUBE_ID_RE = re.compile(r'^[A-Za-z0-9_-]{11}$')
MATCH_KEY_RE = re.compile(r'^[A-Za-z0-9_-]{3,40}$')
TEAM_NUMBER_RE = re.compile(r'^\d{1,5}[a-z]?$', re.I)
_SECRET_QUERY_RE = re.compile(r'([?&](?:token|key|api_key|apikey|sig|signature|access_token)=)[^&\s\'\"]+', re.I)
_BEARER_RE = re.compile(r'(Bearer\s+)[A-Za-z0-9._~+/=-]+', re.I)


def valid_youtube_id(value):
    """Validate that `value` is a canonical 11-character YouTube video ID (`^[A-Za-z0-9_-]{11}$`)."""
    s = str(value or '').strip()
    if not YOUTUBE_ID_RE.fullmatch(s):
        raise ValueError(f'invalid YouTube video ID: {value!r}')
    return s


def valid_match_key(value):
    """Validate that `value` is a safe match key token with no URL/path metacharacters."""
    s = str(value or '').strip()
    if not MATCH_KEY_RE.fullmatch(s):
        raise ValueError(f'invalid match key: {value!r}')
    return s


def clean_team_list(items, lowercase=False):
    r"""Return validated bare FRC team numbers (`^\d{1,5}[a-z]?$`), dropping non-team prompt-injection text."""
    out = []
    for item in (items or [])[:6]:
        raw = str(item or '').strip()
        if raw.lower().startswith('frc'):
            raw = raw[3:].strip()
        if lowercase:
            raw = raw.lower()
        if TEAM_NUMBER_RE.fullmatch(raw):
            out.append(raw)
    return out


def redact_secrets(text):
    """Strip query-string tokens (`?token=...`, `&key=...`) and Bearer tokens from error/log strings."""
    s = str(text or '')
    s = _SECRET_QUERY_RE.sub(r'\1<redacted>', s)
    s = _BEARER_RE.sub(r'\1<redacted>', s)
    return s

# Match time is milliseconds from the moment the autonomous period starts. Every timestamp this
# package writes to the database is match time, with the video's offset stored in
# vision_views.sync_offset_ms, so the page can seek the video and the analytics can compare
# against the 20 second auto window without knowing where the match sits in a broadcast.
AUTO_END_MS = int(FIELD['autoSeconds'] * 1000)
AUTO_GRACE_MS = int(FIELD['autoScoringGraceSeconds'] * 1000)


def log(message):
    print(message, flush=True)


def env(name, default=None, required=True):
    """Read NAME from the process environment, then from the repo's .env."""
    if os.environ.get(name):
        return os.environ[name]
    env_file = ROOT / '.env'
    if env_file.exists():
        match = re.search(rf'^{name}=(.+)$', env_file.read_text(), re.M)
        if match and match.group(1).strip():
            return match.group(1).strip()
    if default is not None:
        return default
    if required:
        sys.exit(f'{name} is missing from the environment and {env_file}')
    return None


class RetryableError(RuntimeError):
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def with_retries(call, attempts=5, base=2.0, cap=60.0, sleep=time.sleep):
    """Run call(); on RetryableError wait base*2^n (capped, with jitter, honouring Retry-After).

    The old fixed 3s/6s/9s retry hammered a rate-limited API three times and gave up; 429s and
    5xx are transient, and honouring Retry-After clears them in one wait instead of three.
    """
    last = None
    for attempt in range(attempts):
        try:
            return call()
        except RetryableError as error:
            last = error
            if attempt == attempts - 1:
                break
            wait = error.retry_after if error.retry_after is not None else min(cap, base * 2 ** attempt)
            sleep(wait * (0.75 + random.random() * 0.5))
    raise last


def http_json(url, body=None, headers=None, method=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method or ('POST' if data else 'GET'),
                                     headers={'Content-Type': 'application/json', **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        text = error.read()[:400]
        if error.code in (408, 425, 429, 500, 502, 503, 504):
            retry_after = error.headers.get('Retry-After')
            raise RetryableError(f'HTTP {error.code}: {text!r}', float(retry_after) if retry_after and retry_after.isdigit() else None)
        raise RuntimeError(f'HTTP {error.code} from {redact_secrets(url)}: {text!r}')
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        raise RetryableError(f'network error: {redact_secrets(error)}')


def tba_match(match_key, tba_key):
    key = urllib.parse.quote(valid_match_key(match_key), safe='')
    return http_json(f'https://www.thebluealliance.com/api/v3/match/{key}',
                     headers={'X-TBA-Auth-Key': tba_key, 'User-Agent': 'spartan-comp/1.0'}, timeout=60)


def roster_from_tba(match):
    """({'red': ['971', ...], 'blue': [...]}) with bare team numbers."""
    return {color: clean_team_list(match['alliances'][color]['team_keys']) for color in ('red', 'blue')}
