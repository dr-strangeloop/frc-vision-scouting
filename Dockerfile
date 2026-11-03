# CPU image: calibration, Gemini mode and FUEL counting. For a local detector on a GPU, start from an NVIDIA
# PyTorch image instead and install requirements-local.txt.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-local.txt ./
ARG LOCAL=0
RUN pip install --no-cache-dir -r $([ "$LOCAL" = 1 ] && echo requirements-local.txt || echo requirements.txt)
COPY . .
ENTRYPOINT ["python3", "vision.py"]
CMD ["check"]
