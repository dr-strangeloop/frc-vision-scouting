"""One interface over Gemini and any OpenAI-compatible server (vLLM, SGLang, Ollama).

Everything in the pipeline that needs a vision-language model calls `client.generate_json(parts,
schema)`. Swapping Gemini for a local Qwen3-VL is then a flag, not a rewrite, and the two can be
compared on the same prompts (see compare.py).

Parts are plain dicts: {'text': str}, {'image': bytes, 'mime': 'image/jpeg'} or, Gemini only,
{'youtube': url, 'start': s, 'end': s, 'fps': n}.
"""
import base64
import json
import os
import threading
import time
from pathlib import Path

import cache

from common import RetryableError, http_json, with_retries

# USD per 1M tokens, from https://ai.google.dev/gemini-api/docs/pricing (checked 2026-10-05).
# Flash prices shown are the promotional rate that runs through 2026-12-31. Used only for the
# cost estimate printed at the end of a run; Google's invoice is the source of truth.
GEMINI_PRICES = {
    # Standard tier, promotional through 2026-11-30 (then 4.00 / 20.00). Batch is not used here, so batch = standard.
    'gemini-4-argon': {'in': 2.00, 'out': 10.00, 'batch_in': 2.00, 'batch_out': 10.00},
    'gemini-3.1-pro-preview': {'in': 2.00, 'out': 12.00, 'batch_in': 1.00, 'batch_out': 6.00},
    'gemini-3.8-flash': {'in': 0.75, 'out': 3.75, 'batch_in': 0.375, 'batch_out': 1.875},
    'gemini-3.7-flash': {'in': 0.75, 'out': 3.75, 'batch_in': 0.375, 'batch_out': 1.875},
    'gemini-3.5-flash': {'in': 1.50, 'out': 9.00, 'batch_in': 0.75, 'batch_out': 4.50},
    'gemini-3.1-flash-lite': {'in': 0.25, 'out': 1.50, 'batch_in': 0.125, 'batch_out': 0.75},
    'gemini-2.5-pro': {'in': 1.25, 'out': 10.00, 'batch_in': 0.625, 'batch_out': 5.00},
    'gemini-2.5-flash': {'in': 0.30, 'out': 2.50, 'batch_in': 0.15, 'batch_out': 1.25},
    'gemini-2.5-flash-lite': {'in': 0.10, 'out': 0.40, 'batch_in': 0.05, 'batch_out': 0.20},
}

DEFAULT_GEMINI_PRICE = GEMINI_PRICES['gemini-4-argon']
NON_RETRYABLE_FINISH_REASONS = ('SAFETY', 'PROHIBITED_CONTENT', 'BLOCKLIST', 'SPII', 'IMAGE_SAFETY')
GEMINI_HTTP_TIMEOUT_S = 180


def price_for_model(model: str | None) -> dict | None:
    """Resolve pricing for any Gemini model name (including models/ prefixes and versioned suffixes)."""
    name = str(model or '').strip().lower().removeprefix('models/')
    if not name:
        return None
    if name in GEMINI_PRICES:
        return GEMINI_PRICES[name]
    for key in sorted(GEMINI_PRICES, key=len, reverse=True):
        if name.startswith(key):
            return GEMINI_PRICES[key]
    if name.startswith('gemini'):
        if 'flash-lite' in name:
            return GEMINI_PRICES['gemini-3.1-flash-lite']
        if 'flash' in name:
            return GEMINI_PRICES['gemini-3.8-flash']
        if 'pro' in name:
            return GEMINI_PRICES['gemini-3.1-pro-preview']
        if 'argon' in name:
            return GEMINI_PRICES['gemini-4-argon']
        return DEFAULT_GEMINI_PRICE
    return None

# Model roles. Pro only where a mistake is expensive; the cheap tier does the bulk.
GEMINI_TIERS = {'bulk': 'gemini-3.8-flash', 'cheap': 'gemini-3.1-flash-lite', 'verify': 'gemini-3.1-pro-preview'}

# How hard each tier thinks. Gemini 3 models reason by default, which is slow and billed as output tokens; reading
# events or bumper numbers off video is perception, not reasoning, so the bulk tiers think as little as the model allows.
# Only the verifier, where a wrong answer is expensive, is allowed to think a moderate amount.
GEMINI_THINKING = {'bulk': 'low', 'cheap': 'minimal', 'verify': 'medium'}


def to_gemini_schema(schema):
    """JSON Schema (lowercase types) -> Gemini's OpenAPI subset (uppercase types)."""
    if isinstance(schema, dict):
        out = {}
        for key, value in schema.items():
            if key == 'type' and isinstance(value, str):
                out[key] = value.upper()
            elif key in ('additionalProperties', '$schema'):
                continue
            else:
                out[key] = to_gemini_schema(value)
        return out
    if isinstance(schema, list):
        return [to_gemini_schema(item) for item in schema]
    return schema


def _ledger_file():
    """Agent Tracker's usage ledger, when the tracker is installed: every Gemini call is added to its Cost today."""
    if os.environ.get('AGENTTRACKER_LEDGER', '1') in ('0', 'false', 'no'):
        return None
    base = Path(os.environ.get('AGENTTRACKER_DIR') or Path.home() / 'Desktop' / 'agenttracker' / 'data')
    return base / 'usage.jsonl' if base.is_dir() else None


def record_usage(model, tokens_in, tokens_out, cached=0):
    path = _ledger_file()
    if not path:
        return
    line = json.dumps({'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'source': 'vision', 'label': os.environ.get('VISION_RUN_LABEL', 'vision run'),
                       'model': model, 'in': tokens_in, 'out': tokens_out, 'cached': cached})
    try:
        with open(path, 'a') as handle:            # one short append per call: safe across threads and processes
            handle.write(line + '\n')
    except OSError:
        pass


class BudgetExceeded(RuntimeError):
    """The run has spent its Gemini budget (VISION_MAX_USD); stopping beats an open-ended bill."""


class UsageMeter:
    """Thread-safe token and cost tally across a run, with a spending cap.

    VISION_MAX_USD (default 5) is the most one run may spend at list prices; the call that crosses it is the last.
    A normal Gemini auto run costs about $0.50, so the cap only stops runaways (a wrong or very long video)."""

    def __init__(self, max_usd=None):
        try:
            self.max_usd = float(max_usd if max_usd is not None else os.environ.get('VISION_MAX_USD', 5))
        except ValueError:
            self.max_usd = 5.0
        self._lock = threading.Lock()
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.thoughts_out = 0
        self.cost = 0.0
        self.reserved_cost = 0.0
        self.in_flight = 0
        self.by_model = {}
        self.cache_hits = 0

    def hit(self):
        """A request answered from the local cache: no tokens, no cost, no wait."""
        with self._lock:
            self.cache_hits += 1

    def estimate_cost(self, model, parts=(), max_tokens=16000):
        """Conservative in-flight cost estimate so concurrent worker calls cannot overshoot VISION_MAX_USD."""
        price = price_for_model(model)
        if not price:
            return 0.0
        tokens_in = 0
        media_units = 0
        for part in (parts or ()):
            if not isinstance(part, dict):
                continue
            if 'text' in part:
                tokens_in += max(1, len(str(part.get('text') or '')) // 4)
            elif 'image' in part:
                tokens_in += 258
                media_units += 1
            elif 'youtube' in part:
                span = max(1.0, float(part.get('end') or 0) - float(part.get('start') or 0))
                fps = max(0.5, float(part.get('fps') or 1))
                frames = int(span * fps)
                tokens_in += max(258, frames * 258)
                media_units += max(1, frames // 4)
        est_out = max(128, min(int(max_tokens or 1000), 2000) // 4, media_units * 120)
        est = (tokens_in * price['in'] + est_out * price['out']) / 1e6
        with self._lock:
            if self.calls > 0 and self.cost > 0:
                est = max(est, self.cost / self.calls)
        return est

    def reserve(self, model, parts=(), max_tokens=16000):
        """Context manager reserving estimated in-flight cost before an HTTP call."""
        meter = self
        est = self.estimate_cost(model, parts, max_tokens=max_tokens)

        class _Reservation:
            def __enter__(self_inner):
                with meter._lock:
                    if meter.max_usd > 0:
                        if meter.cost > meter.max_usd:
                            raise BudgetExceeded(
                                f'this run is over its ${meter.max_usd:.2f} Gemini limit (VISION_MAX_USD); no more calls are made'
                            )
                        if meter.in_flight > 0 and (meter.cost + meter.reserved_cost + est) > meter.max_usd:
                            raise BudgetExceeded(
                                f'concurrent calls would exceed this run\'s ${meter.max_usd:.2f} Gemini limit '
                                f'(spent ${meter.cost:.2f} + ${meter.reserved_cost + est:.2f} in flight, VISION_MAX_USD)'
                            )
                    meter.in_flight += 1
                    meter.reserved_cost += est
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                with meter._lock:
                    meter.in_flight = max(0, meter.in_flight - 1)
                    meter.reserved_cost = max(0.0, meter.reserved_cost - est)
                    if meter.in_flight == 0:
                        meter.reserved_cost = 0.0
                return False

        return _Reservation()

    def add(self, model, tokens_in, tokens_out, batch=False, thoughts_out=0):
        price = price_for_model(model)
        cost = 0.0
        if price:
            cost = (tokens_in * price['batch_in' if batch else 'in'] + tokens_out * price['batch_out' if batch else 'out']) / 1e6
        with self._lock:
            self.calls += 1
            self.tokens_in += tokens_in
            self.tokens_out += tokens_out
            self.thoughts_out += int(thoughts_out or 0)
            self.cost += cost
            entry = self.by_model.setdefault(model, {'calls': 0, 'in': 0, 'out': 0, 'usd': 0.0})
            entry['calls'] += 1
            entry['in'] += tokens_in
            entry['out'] += tokens_out
            entry['usd'] = round(entry.get('usd', 0.0) + cost, 4)
            total_cost = self.cost
        record_usage(model, tokens_in, tokens_out)
        if self.max_usd > 0 and total_cost > self.max_usd:
            raise BudgetExceeded(f'stopped after spending ${total_cost:.2f} on Gemini, over this run\'s ${self.max_usd:.2f} limit '
                                 f'(VISION_MAX_USD). Check the video is one match, or raise the limit.')

    def over_budget(self):
        with self._lock:
            if self.max_usd <= 0:
                return False
            if self.cost > self.max_usd:
                return True
            return self.in_flight > 0 and (self.cost + self.reserved_cost) >= self.max_usd

    def summary(self):
        with self._lock:
            return {'calls': self.calls, 'cache_hits': self.cache_hits, 'tokens_in': self.tokens_in, 'tokens_out': self.tokens_out,
                    'thoughts_out': self.thoughts_out, 'estimated_usd': round(self.cost, 4), 'by_model': {k: dict(v) for k, v in self.by_model.items()}}


class GeminiClient:
    kind = 'gemini'

    def __init__(self, model, api_key, meter=None, thinking_level=None, media_resolution=None):
        self.model = model
        self.key = api_key
        self.meter = meter or UsageMeter()
        self.thinking_level = thinking_level
        self.media_resolution = media_resolution

    def _part(self, part):
        if 'text' in part:
            return {'text': part['text']}
        if 'image' in part:
            return {'inline_data': {'mime_type': part.get('mime', 'image/jpeg'), 'data': base64.b64encode(part['image']).decode()}}
        if 'youtube' in part:
            # Gemini durations allow at most 9 fractional digits; a raw float like 153.33333333333334 is rejected (HTTP 400),
            # which silently failed every verification call. Millisecond precision is plenty for video offsets.
            meta = {'start_offset': f"{max(0.0, float(part['start'])):.3f}s", 'end_offset': f"{max(0.0, float(part['end'])):.3f}s"}
            if part.get('fps'):
                meta['fps'] = part['fps']
            return {'file_data': {'file_uri': part['youtube']}, 'video_metadata': meta}
        raise ValueError(f'unsupported part {list(part)}')

    def _body(self, parts, schema, max_tokens, temperature, media_resolution):
        config = {'maxOutputTokens': max_tokens, 'responseMimeType': 'application/json', 'responseSchema': to_gemini_schema(schema)}
        # Google recommends leaving temperature at its default for Gemini 3 (lower values can loop);
        # callers on older models may pass one.
        if temperature is not None:
            config['temperature'] = temperature
        if self.thinking_level:
            config['thinkingConfig'] = {'thinkingLevel': self.thinking_level}
        resolution = media_resolution or self.media_resolution
        if resolution:
            config['mediaResolution'] = f'MEDIA_RESOLUTION_{resolution.upper()}'
        return {'contents': [{'role': 'user', 'parts': [self._part(p) for p in parts]}], 'generationConfig': config}

    def generate_json(self, parts, schema, max_tokens=16000, temperature=None, media_resolution=None):
        parts = list(parts)
        initial_thinking = self.thinking_level
        key = cache.request_key(self.model, parts, schema, thinking=initial_thinking, media=media_resolution or self.media_resolution,
                                max_tokens=max_tokens, temperature=temperature)
        saved = cache.get_answer(key)
        if saved is not None:
            self.meter.hit()
            return saved
        if self.meter.over_budget():                    # refuse before paying for another call
            raise BudgetExceeded(f'this run is over its ${self.meter.max_usd:.2f} Gemini limit (VISION_MAX_USD); no more calls are made')
        result = self._generate(parts, schema, max_tokens, temperature, media_resolution)
        cache.put_answer(key, result)
        if self.thinking_level != initial_thinking:
            alt_key = cache.request_key(self.model, parts, schema, thinking=self.thinking_level, media=media_resolution or self.media_resolution,
                                        max_tokens=max_tokens, temperature=temperature)
            cache.put_answer(alt_key, result)
        return result

    def _generate(self, parts, schema, max_tokens, temperature, media_resolution):
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent'
        body = self._body(parts, schema, max_tokens, temperature, media_resolution)

        def call():
            if self.meter.over_budget():
                raise BudgetExceeded(f'this run is over its ${self.meter.max_usd:.2f} Gemini limit (VISION_MAX_USD); no more calls are made')
            timeout_s = int(os.environ.get('VISION_GEMINI_TIMEOUT_S', GEMINI_HTTP_TIMEOUT_S))
            with self.meter.reserve(self.model, parts, max_tokens=max_tokens):
                payload = http_json(url, body, {'x-goog-api-key': self.key}, timeout=timeout_s)
            candidate = (payload.get('candidates') or [{}])[0]
            parts_out = (candidate.get('content') or {}).get('parts') or []
            text = ''.join(p.get('text', '') for p in parts_out if not p.get('thought')).strip()
            if text.startswith('```'):
                text = text.strip('`').split('\n', 1)[-1].rsplit('```', 1)[0].strip()
            usage = payload.get('usageMetadata') or {}
            tokens_in = usage.get('promptTokenCount') or 0
            thoughts_tokens = usage.get('thoughtsTokenCount') or 0
            tokens_out = (usage.get('candidatesTokenCount') or 0) + thoughts_tokens
            if not text:
                block_reason = (payload.get('promptFeedback') or {}).get('blockReason')
                finish_reason = candidate.get('finishReason')
                self.meter.add(self.model, tokens_in, tokens_out, thoughts_out=thoughts_tokens)
                if block_reason or finish_reason in NON_RETRYABLE_FINISH_REASONS:
                    raise RuntimeError(f'Gemini blocked response (reason={block_reason or finish_reason})')
                raise RetryableError(f'empty response (finishReason={finish_reason})')
            try:
                parsed = json.loads(text)
            except ValueError as err:
                self.meter.add(self.model, tokens_in, tokens_out, thoughts_out=thoughts_tokens)
                raise RetryableError(f'malformed JSON response: {err}') from err
            try:
                self.meter.add(self.model, tokens_in, tokens_out, thoughts_out=thoughts_tokens)
            except BudgetExceeded:
                key = cache.request_key(self.model, parts, schema, thinking=self.thinking_level,
                                        media=media_resolution or self.media_resolution,
                                        max_tokens=max_tokens, temperature=temperature)
                cache.put_answer(key, parsed)
                raise
            return parsed

        try:
            return with_retries(call)
        except RuntimeError as error:
            # The thinking setting is the newest, least-documented field we send. A model that does not accept
            # it must not fail the whole run: drop it for this client and carry on at the model's default.
            if self.thinking_level and 'HTTP 400' in str(error) and 'think' in str(error).lower():
                self.thinking_level = None
                body = self._body(parts, schema, max_tokens, temperature, media_resolution)
                return with_retries(call)
            raise


class OpenAICompatClient:
    """Local VLM behind vLLM / SGLang / Ollama. Images only (send sampled frames, not a video)."""
    kind = 'local'

    def __init__(self, base_url, model, api_key='EMPTY', meter=None):
        self.base = base_url.rstrip('/')
        self.model = model
        self.key = api_key
        self.meter = meter or UsageMeter()

    def generate_json(self, parts, schema, max_tokens=8000, temperature=0.1):
        parts = list(parts)
        content = []
        for part in parts:
            if 'text' in part:
                content.append({'type': 'text', 'text': part['text']})
            elif 'image' in part:
                content.append({'type': 'image_url', 'image_url': {'url': f"data:{part.get('mime', 'image/jpeg')};base64,{base64.b64encode(part['image']).decode()}"}})
            else:
                raise ValueError('a local VLM takes frames, not a YouTube URL; cut frames first')
        body = {'model': self.model, 'messages': [{'role': 'user', 'content': content}], 'max_tokens': max_tokens, 'temperature': temperature,
                # vLLM constrains decoding to the schema; servers that do not know the field ignore it.
                'response_format': {'type': 'json_schema', 'json_schema': {'name': 'result', 'schema': schema}}}

        def call():
            payload = http_json(f'{self.base}/chat/completions', body, {'Authorization': f'Bearer {self.key}'}, timeout=900)
            usage = payload.get('usage') or {}
            self.meter.add(self.model, usage.get('prompt_tokens') or 0, usage.get('completion_tokens') or 0)
            choices = payload.get('choices') or [{}]
            text = ((choices[0].get('message') or {}).get('content') or '').strip()
            if text.startswith('```'):
                text = text.strip('`').split('\n', 1)[-1].rsplit('```', 1)[0].strip()
            if not text:
                raise RetryableError('empty response from local VLM')
            try:
                return json.loads(text)
            except ValueError as err:
                raise RetryableError(f'malformed JSON response: {err}') from err

        return with_retries(call)


def make_client(kind, model=None, meter=None, **options):   # options: thinking_level, media_resolution (Gemini)
    from common import env
    if kind == 'gemini':
        return GeminiClient(model or GEMINI_TIERS['bulk'], env('GEMINI_API_KEY'), meter, **options)
    if kind == 'local':
        return OpenAICompatClient(env('LOCAL_VLM_URL', 'http://localhost:8000/v1'), model or env('LOCAL_VLM_MODEL', 'Qwen/Qwen3-VL-32B-Instruct-FP8'), env('LOCAL_VLM_KEY', 'EMPTY'), meter)
    raise ValueError(f'unknown VLM kind {kind}')
