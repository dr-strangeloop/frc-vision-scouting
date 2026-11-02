"""VRAM budgeting. The whole vision stack must fit in one budget (default 128 GB).

Weights are estimated from parameter count and numeric format; a KV-cache / activation allowance is
added per model. The numbers are planning estimates (vLLM's real usage depends on context length,
batch size and its own allocator), so `plan()` keeps headroom and the runner also enforces the limit
at runtime (`enforce`) and tells vLLM how much it may take (`vllm_args`).
"""
from dataclasses import dataclass

BYTES_PER_PARAM = {'bf16': 2.0, 'fp16': 2.0, 'fp8': 1.0, 'int8': 1.0, 'int4': 0.55, 'awq': 0.55, 'fp32': 4.0}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    role: str            # 'detector' | 'vlm' | 'segmenter' | 'ocr'
    params_b: float      # billions of parameters (total, not active)
    fmt: str
    overhead_gb: float   # activations / KV cache / CUDA context at the intended batch size
    note: str = ''

    @property
    def weights_gb(self):
        return self.params_b * BYTES_PER_PARAM[self.fmt]

    @property
    def total_gb(self):
        return self.weights_gb + self.overhead_gb


# A catalogue of the candidates, sized from published parameter counts. Overheads assume the
# settings this pipeline uses (<= 24k context for the VLM, batch 8-16 for the detector).
CATALOGUE = {
    'rfdetr-medium': ModelSpec('rfdetr-medium', 'detector', 0.033, 'fp16', 3.0, 'robot / bumper detector, batch 16'),
    'yolo26-m-obb': ModelSpec('yolo26-m-obb', 'detector', 0.021, 'fp16', 2.5, 'oriented boxes give heading'),
    'sam3.1': ModelSpec('sam3.1', 'segmenter', 0.85, 'bf16', 8.0, 'optional: masks for fuel / robots, 16 objects per pass'),
    'qwen3-vl-8b-fp8': ModelSpec('qwen3-vl-8b-fp8', 'vlm', 8.8, 'fp8', 8.0),
    'qwen3-vl-30b-a3b-fp8': ModelSpec('qwen3-vl-30b-a3b-fp8', 'vlm', 31.0, 'fp8', 18.0, '3B active parameters: fastest large option'),
    'qwen3-vl-30b-a3b-bf16': ModelSpec('qwen3-vl-30b-a3b-bf16', 'vlm', 31.0, 'bf16', 24.0, 'the app default today'),
    'qwen3-vl-32b-fp8': ModelSpec('qwen3-vl-32b-fp8', 'vlm', 33.0, 'fp8', 20.0, 'dense, most accurate that fits comfortably'),
    'qwen3-vl-32b-bf16': ModelSpec('qwen3-vl-32b-bf16', 'vlm', 33.0, 'bf16', 28.0),
    'qwen3-vl-235b-a22b-int4': ModelSpec('qwen3-vl-235b-a22b-int4', 'vlm', 235.0, 'int4', 22.0, '~150 GB: does not fit a 128 GB budget'),
    'qwen3-vl-235b-a22b-fp8': ModelSpec('qwen3-vl-235b-a22b-fp8', 'vlm', 235.0, 'fp8', 30.0, '~265 GB: needs several GPUs'),
}

# Preference order per profile: first stack whose sum fits the budget (with headroom) wins.
PROFILES = {
    'accuracy': [('rfdetr-medium', 'qwen3-vl-32b-bf16'), ('rfdetr-medium', 'qwen3-vl-32b-fp8'), ('rfdetr-medium', 'qwen3-vl-30b-a3b-fp8'), ('rfdetr-medium', 'qwen3-vl-8b-fp8')],
    'speed': [('rfdetr-medium', 'qwen3-vl-30b-a3b-fp8'), ('rfdetr-medium', 'qwen3-vl-8b-fp8')],
    'detector-only': [('rfdetr-medium',), ('yolo26-m-obb',)],
}


def stack_gb(names):
    return sum(CATALOGUE[n].total_gb for n in names)


def plan(budget_gb=128.0, profile='accuracy', headroom=0.10, extra=()):
    """Pick the best stack that fits budget_gb * (1 - headroom). `extra` models (e.g. 'sam3.1') are
    added to every candidate. Returns dict(models, gb, budget_gb, usable_gb, profile) or raises."""
    usable = budget_gb * (1 - headroom)
    for names in PROFILES[profile]:
        full = tuple(names) + tuple(extra)
        if stack_gb(full) <= usable:
            return {'models': list(full), 'gb': round(stack_gb(full), 1), 'budget_gb': budget_gb, 'usable_gb': round(usable, 1), 'profile': profile,
                    'breakdown': {n: round(CATALOGUE[n].total_gb, 1) for n in full}}
    raise ValueError(f'no {profile} stack fits {usable:.0f} GB; smallest candidate needs {min(stack_gb(tuple(s) + tuple(extra)) for s in PROFILES[profile]):.0f} GB')


def vllm_args(model_gb, total_gpu_gb, tensor_parallel=1, max_len=24576):
    """`vllm serve` flags that cap the VLM at model_gb of a GPU (per GPU when tensor parallel)."""
    util = min(0.95, max(0.1, model_gb / (total_gpu_gb * tensor_parallel)))
    return ['--gpu-memory-utilization', f'{util:.2f}', '--max-model-len', str(max_len), '--tensor-parallel-size', str(tensor_parallel),
            '--limit-mm-per-prompt', '{"image": 24}', '--enable-prefix-caching']


def enforce(budget_gb, device=0):
    """Hard-cap THIS process's CUDA allocations (detector side). The VLM server is its own process
    and is capped with vllm_args. Returns the fraction applied, or None without CUDA."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    fraction = min(1.0, budget_gb / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    return fraction


def current_usage_gb(device=0):
    try:
        import torch
        return torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else None
    except ImportError:
        return None
