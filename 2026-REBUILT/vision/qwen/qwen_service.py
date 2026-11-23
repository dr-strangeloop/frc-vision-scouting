#!/usr/bin/env python3
"""Authenticated Qwen3.8-27B BF16 service for DGX Spark."""
from __future__ import annotations

import asyncio
import base64
import hmac
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
from contextlib import asynccontextmanager

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, Qwen3VLForConditionalGeneration

from robot_appearance_contract import match_content, parse_match
from reference_features import ReferenceFeatureCache
from qwen_contract import SYSTEM_PROMPT, TASK_PROMPT, normalize_result, parse_json_response
from fuel_count_contract import FUEL_COUNT_PROMPT, normalize_fuel_count
from team_identity_contract import (
    COMPACT_TASK_PROMPT as IDENTITY_COMPACT_PROMPT,
    GRID_LETTERS,
    GRID_TASK_PROMPT as IDENTITY_GRID_PROMPT,
    parse_grid_line,
    SYSTEM_PROMPT as IDENTITY_SYSTEM_PROMPT,
    TASK_PROMPT as IDENTITY_TASK_PROMPT,
    normalize_compact_reads,
    normalize_reads,
    parse_json_response as parse_identity_json,
)
from fewshot_robot_contract import PROMPT as FEWSHOT_ROBOT_PROMPT, parse as parse_fewshot_robot, normalize as normalize_fewshot_robot
from landmark_contract import (
    SYSTEM_PROMPT as LANDMARK_SYSTEM_PROMPT,
    normalize_landmarks,
    parse_json_response as parse_landmark_json,
    valid_landmark_ids,
)

MODEL_NAME = os.environ.get("VISION_QWEN_MODEL", "Qwen/Qwen3.8-27B")
MODEL_REVISION = os.environ.get("VISION_QWEN_REVISION", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
TOKEN = os.environ.get("VISION_QWEN_TOKEN", "")
ATTENTION = os.environ.get("VISION_QWEN_ATTENTION", "sdpa")
DEVICE_MAP = os.environ.get("VISION_QWEN_DEVICE_MAP", "cuda")
MAX_IMAGES = int(os.environ.get("VISION_QWEN_MAX_IMAGES", "8"))
MAX_IMAGE_PIXELS = int(os.environ.get("VISION_QWEN_MAX_IMAGE_PIXELS", str(1280 * 720)))
MAX_JPEG_BYTES = int(os.environ.get("VISION_QWEN_MAX_JPEG_BYTES", str(2 * 1024 * 1024)))
MAX_NEW_TOKENS = int(os.environ.get("VISION_QWEN_MAX_NEW_TOKENS", "900"))

state = {
    "model": None, "processor": None, "loaded_at": None,
    "requests": 0, "failures": 0, "last_latency_ms": None,
}
BODY_MODEL = "nvidia/Cosmos-Reason2-8B"
BODY_REVISION = "a9fae2cf89dc64db96b12860417f0eb403013bb9"
BODY_ATTENTION = "eager"
BODY_MODEL_PATH = os.environ.get("VISION_BODY_ID_MODEL_PATH", "")
body_state = {"model": None, "processor": None, "cache": None}
inference_lock = asyncio.Lock()


class FrameInput(BaseModel):
    timestamp_ms: int = Field(ge=0)
    jpeg_base64: str


class AnalyzeRequest(BaseModel):
    match_key: str | None = None
    view_id: str
    view_label: str | None = None
    clip_start_ms: int = Field(ge=0)
    clip_end_ms: int = Field(gt=0)
    frames: list[FrameInput]


class TeamNumberCrop(BaseModel):
    crop_id: str = Field(min_length=1, max_length=200)
    track_key: str = Field(min_length=1, max_length=200)
    alliance: str = Field(pattern="^(red|blue)$")
    candidate_team_keys: list[str] = Field(min_length=1, max_length=3)
    jpeg_base64: str


class TeamNumberRequest(BaseModel):
    crops: list[TeamNumberCrop] = Field(min_length=1, max_length=MAX_IMAGES)


MAX_DIGIT_CROPS = int(os.environ.get("VISION_QWEN_MAX_DIGIT_CROPS", "32"))


class BumperDigitsRequest(BaseModel):
    crops: list[TeamNumberCrop] = Field(min_length=1, max_length=MAX_DIGIT_CROPS)

class BodyReference(BaseModel):
    reference_id: str = Field(pattern="^[A-F]$")
    jpeg_base64: str


class BodyTarget(BaseModel):
    crop_id: str = Field(pattern="^[A-Za-z0-9_-]{1,32}$")
    jpeg_base64: str


class BodyMatchRequest(BaseModel):
    references: list[BodyReference] = Field(min_length=2, max_length=6)
    targets: list[BodyTarget] = Field(min_length=1, max_length=12)


class RobotExample(FrameInput):
    boxes: list[dict] = Field(default_factory=list, max_length=12)

class LandmarkRequest(BaseModel):
    frame: FrameInput
    prompt: str = Field(min_length=1, max_length=8000)
    landmark_ids: list[str] = Field(min_length=4, max_length=64)

class FewShotRobotRequest(BaseModel):
    examples: list[RobotExample] = Field(min_length=1, max_length=6)
    target: FrameInput


def decode_frame(frame: FrameInput):
    try:
        if len(frame.jpeg_base64) > (MAX_JPEG_BYTES * 4 // 3) + 16:
            raise ValueError("encoded frame exceeds byte limit")
        decoded = base64.b64decode(frame.jpeg_base64, validate=True)
        if len(decoded) > MAX_JPEG_BYTES:
            raise ValueError("decoded frame exceeds byte limit")
        image = Image.open(io.BytesIO(decoded)).convert("RGB")
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Invalid JPEG frame: {error}") from error
    if image.width * image.height > MAX_IMAGE_PIXELS:
        scale = math.sqrt(MAX_IMAGE_PIXELS / (image.width * image.height))
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    return image


def model_health():
    # A health probe must not wait on a long GPU generation. The runner uses
    # this endpoint for liveness while individual clips are being processed.
    cuda = {"available": torch.cuda.is_available(), "memory_reporting": "deferred"}
    return {
        "ready": state["model"] is not None,
        "model": MODEL_NAME, "revision": MODEL_REVISION, "dtype": "bfloat16",
        "attention": ATTENTION, "loaded_at": state["loaded_at"],
        "requests": state["requests"], "failures": state["failures"],
        "last_latency_ms": state["last_latency_ms"], "cuda": cuda,
        "body_identity": {"configured": bool(BODY_MODEL_PATH), "loaded": body_state["model"] is not None,
                          "model": BODY_MODEL, "revision": BODY_REVISION, "attention": BODY_ATTENTION},
    }


def verify_token(authorization: str | None = Header(default=None)):
    supplied = authorization.removeprefix("Bearer ") if authorization else ""
    if not TOKEN or not hmac.compare_digest(supplied, TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")


@asynccontextmanager
async def lifespan(_app):
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen service requires CUDA on the DGX Spark")
    options = {
        "dtype": torch.bfloat16,
        "device_map": DEVICE_MAP,
        "attn_implementation": ATTENTION,
        "low_cpu_mem_usage": True,
    }
    if MODEL_REVISION:
        options["revision"] = MODEL_REVISION
    # The pinned Qwen3.8 checkpoint declares model_type=qwen3_5.
    state["model"] = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_NAME, **options).eval()
    state["processor"] = AutoProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    state["loaded_at"] = time.time()
    yield
    state["model"] = None
    state["processor"] = None


app = FastAPI(title="Spartans Vision Qwen", lifespan=lifespan)


@app.get("/health")
def health():
    return model_health()


@app.post("/analyze", dependencies=[Depends(verify_token)])
async def analyze(payload: AnalyzeRequest):
    if payload.clip_end_ms <= payload.clip_start_ms:
        raise HTTPException(status_code=400, detail="clip_end_ms must exceed clip_start_ms")
    if not 2 <= len(payload.frames) <= MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"frames must contain 2..{MAX_IMAGES} images")
    images = [decode_frame(frame) for frame in payload.frames]
    content = []
    for frame, image in zip(payload.frames, images):
        content.extend([
            {"type": "text", "text": f"Frame timestamp_ms={frame.timestamp_ms}"},
            {"type": "image", "image": image},
        ])
    content.append({"type": "text", "text": TASK_PROMPT})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    started = time.perf_counter()
    async with inference_lock:
        try:
            inputs = state["processor"].apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False,
            ).to(state["model"].device)
            with torch.inference_mode():
                generated = state["model"].generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                )
            trimmed = generated[:, inputs["input_ids"].shape[-1]:]
            raw = state["processor"].batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )[0]
            parsed, error = parse_json_response(raw)
            result = None
            if not error:
                result, error = normalize_result(parsed, payload.clip_start_ms, payload.clip_end_ms)
            if error:
                raise HTTPException(status_code=422, detail={"error": error, "raw_response": raw})
            state["requests"] += 1
            state["last_latency_ms"] = round((time.perf_counter() - started) * 1000)
            return {
                "model": MODEL_NAME, "revision": MODEL_REVISION, "dtype": "bfloat16",
                "latency_ms": state["last_latency_ms"], "result": result,
                "raw_response": raw,
            }
        except HTTPException:
            state["failures"] += 1
            raise
        except Exception as error:
            state["failures"] += 1
            raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/count-fuel", dependencies=[Depends(verify_token)])
async def count_fuel(payload: AnalyzeRequest):
    """Count a short, masked wide-view clip without an event-list token ceiling."""
    if payload.clip_end_ms <= payload.clip_start_ms:
        raise HTTPException(status_code=400, detail="clip_end_ms must exceed clip_start_ms")
    if not 2 <= len(payload.frames) <= MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"frames must contain 2..{MAX_IMAGES} images")
    images = [decode_frame(frame) for frame in payload.frames]
    content = []
    for frame, image in zip(payload.frames, images):
        content.extend([
            {"type": "text", "text": f"Frame timestamp_ms={frame.timestamp_ms}"},
            {"type": "image", "image": image},
        ])
    content.append({"type": "text", "text": FUEL_COUNT_PROMPT})
    started = time.perf_counter()
    async with inference_lock:
        try:
            inputs = state["processor"].apply_chat_template(
                [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                 {"role": "user", "content": content}],
                tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False,
            ).to(state["model"].device)
            with torch.inference_mode():
                generated = state["model"].generate(
                    **inputs, max_new_tokens=min(MAX_NEW_TOKENS, 220), do_sample=False,
                )
            raw = state["processor"].batch_decode(
                generated[:, inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )[0]
            result, error = normalize_fuel_count(raw)
            if error:
                state["failures"] += 1
                raise HTTPException(status_code=422, detail={"error": error, "raw_response": raw})
            state["requests"] += 1
            state["last_latency_ms"] = round((time.perf_counter() - started) * 1000)
            return {"model": MODEL_NAME, "revision": MODEL_REVISION,
                    "dtype": "bfloat16", "latency_ms": state["last_latency_ms"],
                    "result": result, "raw_response": raw}
        except HTTPException:
            raise
        except Exception as error:
            state["failures"] += 1
            raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/read-team-numbers", dependencies=[Depends(verify_token)])
async def read_team_numbers(payload: TeamNumberRequest):
    """Read only roster-constrained bumper numbers from in-memory robot crops.

    This is loopback-authenticated like /analyze; no crop is persisted by this
    service. A rejected/malformed response returns 422 so the runner leaves
    the associated robot unidentified instead of guessing.
    """
    images = [decode_frame(crop) for crop in payload.crops]
    if len({crop.crop_id for crop in payload.crops}) != len(payload.crops):
        raise HTTPException(status_code=400, detail="crop IDs must be unique")
    original_ids = {f"c{i}": crop.crop_id for i, crop in enumerate(payload.crops)}
    permitted = {f"c{i}": set(crop.candidate_team_keys) for i, crop in enumerate(payload.crops)}
    content = []
    for i, (crop, image) in enumerate(zip(payload.crops, images)):
        content.extend([
            # Keep roster hints outside inference. Validate literal OCR against
            # the roster afterwards, so a blurry crop cannot invite a guess.
            {"type": "text", "text": f"crop_id=c{i}; read the printed bumper digits only"},
            {"type": "image", "image": image},
        ])
    content.append({"type": "text", "text": IDENTITY_TASK_PROMPT})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": IDENTITY_SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    started = time.perf_counter()
    async with inference_lock:
        try:
            inputs = state["processor"].apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False,
            ).to(state["model"].device)
            with torch.inference_mode():
                generated = state["model"].generate(**inputs, max_new_tokens=500, do_sample=False)
            trimmed = generated[:, inputs["input_ids"].shape[-1]:]
            raw = state["processor"].batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            parsed, error = parse_identity_json(raw)
            reads = None
            if not error:
                reads, error = normalize_reads(parsed, permitted)
            if error:
                raise HTTPException(status_code=422, detail={"error": error, "raw_response": raw})
            for read in reads:
                read["crop_id"] = original_ids[read["crop_id"]]
            state["requests"] += 1
            state["last_latency_ms"] = round((time.perf_counter() - started) * 1000)
            return {"model": MODEL_NAME, "revision": MODEL_REVISION, "latency_ms": state["last_latency_ms"],
                    "reads": reads, "raw_response": raw}
        except HTTPException:
            state["failures"] += 1
            raise
        except Exception as error:
            state["failures"] += 1
            raise HTTPException(status_code=500, detail=str(error)) from error

@app.post("/read-bumper-digits", dependencies=[Depends(verify_token)])
async def read_bumper_digits(payload: BumperDigitsRequest):
    """Compact /read-team-numbers for many small crops: the same literal, roster-blind reading, answered as one JSON map
    of crop id to digits (no evidence text), so a batch costs a few tokens per crop. Roster validation is afterwards,
    as in /read-team-numbers; a malformed answer is a 422 and leaves the robots unidentified."""
    images = [decode_frame(crop) for crop in payload.crops]
    if len({crop.crop_id for crop in payload.crops}) != len(payload.crops):
        raise HTTPException(status_code=400, detail="crop IDs must be unique")
    original_ids = {f"c{i}": crop.crop_id for i, crop in enumerate(payload.crops)}
    permitted = {f"c{i}": set(crop.candidate_team_keys) for i, crop in enumerate(payload.crops)}
    content = []
    for i, image in enumerate(images):
        content.extend([{"type": "text", "text": f"c{i}:"}, {"type": "image", "image": image}])
    content.append({"type": "text", "text": IDENTITY_COMPACT_PROMPT})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": IDENTITY_SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    started = time.perf_counter()
    async with inference_lock:
        try:
            inputs = state["processor"].apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False,
            ).to(state["model"].device)
            with torch.inference_mode():
                generated = state["model"].generate(**inputs, max_new_tokens=24 + 14 * len(images), do_sample=False)
            trimmed = generated[:, inputs["input_ids"].shape[-1]:]
            raw = state["processor"].batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            parsed, error = parse_identity_json(raw)
            reads = None
            if not error:
                reads, error = normalize_compact_reads(parsed, permitted)
            if error:
                raise HTTPException(status_code=422, detail={"error": error, "raw_response": raw})
            for read in reads:
                read["crop_id"] = original_ids[read["crop_id"]]
            state["requests"] += 1
            state["last_latency_ms"] = round((time.perf_counter() - started) * 1000)
            return {"model": MODEL_NAME, "revision": MODEL_REVISION, "latency_ms": state["last_latency_ms"],
                    "reads": reads, "raw_response": raw}
        except HTTPException:
            state["failures"] += 1
            raise
        except Exception as error:
            state["failures"] += 1
            raise HTTPException(status_code=500, detail=str(error)) from error


def bumper_grid(images, cell=(288, 216), columns=4):
    """One image holding up to 24 crops, each letterboxed into a cell with its letter in a black box at the top-left."""
    from PIL import ImageFont
    rows = (len(images) + columns - 1) // columns
    grid = Image.new("RGB", (columns * cell[0], rows * cell[1]), (40, 40, 40))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:
        font = ImageFont.load_default()
    for i, image in enumerate(images):
        x, y = (i % columns) * cell[0], (i // columns) * cell[1]
        scale = min((cell[0] - 4) / image.width, (cell[1] - 4) / image.height)
        fitted = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
        grid.paste(fitted, (x + (cell[0] - fitted.width) // 2, y + (cell[1] - fitted.height) // 2))
        draw.rectangle((x, y, x + 34, y + 34), fill=(0, 0, 0))
        draw.text((x + 7, y + 2), GRID_LETTERS[i], fill=(255, 255, 255), font=font)
        draw.rectangle((x, y, x + cell[0] - 1, y + cell[1] - 1), outline=(255, 255, 255), width=2)
    return grid


@app.post("/read-bumper-grid", dependencies=[Depends(verify_token)])
async def read_bumper_grid(payload: BumperDigitsRequest):
    """/read-bumper-digits with the crops drawn into one lettered grid image: the model pays its per-image cost once
    per grid instead of once per crop. Same literal, roster-blind reading and afterwards roster check."""
    if len(payload.crops) > len(GRID_LETTERS):
        raise HTTPException(status_code=400, detail=f"at most {len(GRID_LETTERS)} crops per grid")
    images = [decode_frame(crop) for crop in payload.crops]
    if len({crop.crop_id for crop in payload.crops}) != len(payload.crops):
        raise HTTPException(status_code=400, detail="crop IDs must be unique")
    letters = GRID_LETTERS[:len(images)]
    original_ids = {letter: crop.crop_id for letter, crop in zip(letters, payload.crops)}
    permitted = {letter: set(crop.candidate_team_keys) for letter, crop in zip(letters, payload.crops)}
    content = [{"type": "text", "text": "Cells: " + ", ".join(letters)},
               {"type": "image", "image": bumper_grid(images)},
               {"type": "text", "text": IDENTITY_GRID_PROMPT}]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": IDENTITY_SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    started = time.perf_counter()
    async with inference_lock:
        try:
            inputs = state["processor"].apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False,
            ).to(state["model"].device)
            with torch.inference_mode():
                generated = state["model"].generate(**inputs, max_new_tokens=8 + 8 * len(images), do_sample=False)
            trimmed = generated[:, inputs["input_ids"].shape[-1]:]
            raw = state["processor"].batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            parsed, error = parse_grid_line(raw, letters)
            reads = None
            if not error:
                reads, error = normalize_compact_reads(parsed, permitted)
            if error:
                raise HTTPException(status_code=422, detail={"error": error, "raw_response": raw})
            for read in reads:
                read["crop_id"] = original_ids[read["crop_id"]]
            state["requests"] += 1
            state["last_latency_ms"] = round((time.perf_counter() - started) * 1000)
            return {"model": MODEL_NAME, "revision": MODEL_REVISION, "latency_ms": state["last_latency_ms"],
                    "reads": reads, "raw_response": raw}
        except HTTPException:
            state["failures"] += 1
            raise
        except Exception as error:
            state["failures"] += 1
            raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/propose-landmarks", dependencies=[Depends(verify_token)])
async def propose_landmarks(payload: LandmarkRequest):
    """Propose field landmark positions on one broadcast frame for automatic calibration.

    The caller (auto_calibration.py on this Spark) blacks out the overlay band first and fits, gates and confirms
    the calibration itself; this only returns proposals for the requested landmark ids. Malformed output is a 422.
    """
    if not valid_landmark_ids(payload.landmark_ids):
        raise HTTPException(status_code=400, detail="landmark_ids must be unique lowercase ids")
    image = decode_frame(payload.frame)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": LANDMARK_SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": payload.prompt + '\nReturn exactly {"landmarks":[{"id":"","u":0.0,"v":0.0,'
                                                      '"visible":true,"confidence":0.0}]}.'},
        ]},
    ]
    started = time.perf_counter()
    async with inference_lock:
        try:
            inputs = state["processor"].apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False,
            ).to(state["model"].device)
            with torch.inference_mode():
                generated = state["model"].generate(**inputs, max_new_tokens=2000, do_sample=False)
            trimmed = generated[:, inputs["input_ids"].shape[-1]:]
            raw = state["processor"].batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            parsed, error = parse_landmark_json(raw)
            landmarks = None
            if not error:
                landmarks, error = normalize_landmarks(parsed, set(payload.landmark_ids))
            if error:
                raise HTTPException(status_code=422, detail={"error": error, "raw_response": raw})
            state["requests"] += 1
            state["last_latency_ms"] = round((time.perf_counter() - started) * 1000)
            return {"model": MODEL_NAME, "revision": MODEL_REVISION, "latency_ms": state["last_latency_ms"],
                    "landmarks": landmarks, "image_size": [image.width, image.height]}
        except HTTPException:
            state["failures"] += 1
            raise
        except Exception as error:
            state["failures"] += 1
            raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/annotate-robots", dependencies=[Depends(verify_token)])
async def annotate_robots(payload: FewShotRobotRequest):
    """Few-shot private robot annotation; responses remain proposals only."""
    examples = [(example, decode_frame(example)) for example in payload.examples]
    target = decode_frame(payload.target)
    content = []
    for example, image in examples:
        # Render the exact human boxes into the example image as well as text.
        # This gives the model an unambiguous visual convention for both color
        # and geometry without changing the unannotated target image.
        image = image.copy(); draw = ImageDraw.Draw(image)
        for box in example.boxes:
            try:
                x1,y1,x2,y2 = [float(value) / 1000 for value in box["box"]]
                color = "#ff4050" if box.get("alliance") == "red" else "#36a8ff"
                draw.rectangle((x1*image.width,y1*image.height,x2*image.width,y2*image.height), outline=color, width=max(3, image.width//300))
            except Exception:
                continue
        labels = json.dumps(example.boxes, separators=(",", ":"))
        content.extend([{"type":"text","text":f"LABELED EXAMPLE. Boxes use 0..1000 coordinates: {labels}"},{"type":"image","image":image}])
    content.extend([{"type":"text","text":"TARGET IMAGE: annotate this image only."},{"type":"image","image":target},{"type":"text","text":FEWSHOT_ROBOT_PROMPT}])
    started=time.perf_counter()
    async with inference_lock:
        try:
            inputs=state["processor"].apply_chat_template([{"role":"user","content":content}],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt",enable_thinking=False).to(state["model"].device)
            with torch.inference_mode(): generated=state["model"].generate(**inputs,max_new_tokens=700,do_sample=False)
            raw=state["processor"].batch_decode(generated[:,inputs["input_ids"].shape[-1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
            parsed,error=parse_fewshot_robot(raw); boxes=None
            if not error: boxes,error=normalize_fewshot_robot(parsed)
            if error: raise HTTPException(status_code=422,detail={"error":error,"raw_response":raw})
            state["requests"]+=1; state["last_latency_ms"]=round((time.perf_counter()-started)*1000)
            return {"boxes":boxes,"latency_ms":state["last_latency_ms"],"raw_response":raw}
        except HTTPException:
            state["failures"]+=1; raise
        except Exception as error:
            state["failures"]+=1; raise HTTPException(status_code=500,detail=str(error)) from error


def load_body_model():
    """Only an explicitly mounted, revision-pinned local cache is eligible; never download at request time."""
    if body_state["model"] is not None:
        return
    path = Path(BODY_MODEL_PATH)
    if not BODY_MODEL_PATH or path.name != BODY_REVISION or not (path / "config.json").is_file():
        raise HTTPException(status_code=503, detail="Pinned local robot body model is unavailable")
    torch.cuda.empty_cache()
    body_state["processor"] = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    body_state["processor"].tokenizer.padding_side = "left"
    body_state["model"] = Qwen3VLForConditionalGeneration.from_pretrained(
        str(path), dtype=torch.bfloat16, device_map=DEVICE_MAP,
        attn_implementation=BODY_ATTENTION, local_files_only=True,
    ).eval()
    body_state["cache"] = ReferenceFeatureCache(body_state["model"].model.get_image_features, torch.cat)
    body_state["model"].model.get_image_features = body_state["cache"]


@app.post("/match-robot-bodies", dependencies=[Depends(verify_token)])
async def match_robot_bodies(payload: BodyMatchRequest):
    ids = [target.crop_id for target in payload.targets]
    refs = [reference.reference_id for reference in payload.references]
    if len(set(ids)) != len(ids) or any(refs.count(ref) < 2 for ref in set(refs)):
        raise HTTPException(status_code=400, detail="Unique target IDs and two views per reference are required")
    reference_images = [decode_frame(FrameInput(timestamp_ms=0, jpeg_base64=ref.jpeg_base64))
                        for ref in payload.references]
    target_images = [decode_frame(FrameInput(timestamp_ms=0, jpeg_base64=target.jpeg_base64))
                     for target in payload.targets]
    started = time.perf_counter()
    async with inference_lock:
        await asyncio.to_thread(load_body_model)
        reference_keys = [hashlib.sha256(base64.b64decode(ref.jpeg_base64)).hexdigest()
                          for ref in payload.references]
        def generate(images):
            model, processor = body_state["model"], body_state["processor"]
            messages = [[{"role": "user", "content": match_content(reference_images, refs, image)}]
                        for image in images]
            body_state["cache"].image_keys = (reference_keys + [None]) * len(images)
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, padding=True,
                return_dict=True, return_tensors="pt", enable_thinking=False,
            ).to(model.device)
            with torch.inference_mode():
                tokens = model.generate(**inputs, max_new_tokens=48, do_sample=False)
            return processor.batch_decode(tokens[:, inputs["input_ids"].shape[-1]:],
                                          skip_special_tokens=True, clean_up_tokenization_spaces=False)
        matches = {}
        # Keep memory bounded alongside the larger literal-bumper reader. Each conversation has one target.
        for start in range(0, len(ids), 4):
            raw = await asyncio.to_thread(generate, target_images[start:start + 4])
            for crop_id, text in zip(ids[start:start + 4], raw):
                matches[crop_id] = parse_match(text, refs)
    return {"matches": matches, "model": BODY_MODEL, "revision": BODY_REVISION,
            "latency_ms": round((time.perf_counter() - started) * 1000)}
