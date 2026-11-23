# syntax=docker/dockerfile:1
# Override this tag with a DGX Spark-supported ARM64 NGC PyTorch release when
# the host image is upgraded. CUDA/PyTorch come from NGC, never pip-installed.
ARG NVIDIA_PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3
FROM ${NVIDIA_PYTORCH_IMAGE}

WORKDIR /app
COPY requirements.txt .
RUN PIP_CONSTRAINT=/dev/null python -m pip install --no-cache-dir -r requirements.txt
COPY *_contract.py reference_features.py qwen_service.py ./

ENV HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "qwen_service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
