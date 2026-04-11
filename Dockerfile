FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf-home \
    HUGGINGFACE_HUB_CACHE=/tmp/hf-home/hub \
    TRANSFORMERS_CACHE=/tmp/hf-home/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    BASE_MODEL_ID=Qwen/Qwen-Image-Edit-2511 \
    CHECKPOINT_REPO_ID=Phr00t/Qwen-Image-Edit-Rapid-AIO \
    CHECKPOINT_FILENAME=v23/Qwen-Rapid-AIO-NSFW-v23.safetensors \
    DEFAULT_NUM_INFERENCE_STEPS=4 \
    DEFAULT_TRUE_GUIDANCE_SCALE=1.0 \
    DEFAULT_REWRITE_PROMPT=false \
    RUNPOD_USE_CACHED_BASE_MODEL=true \
    RUNPOD_ENABLE_BUCKET_UPLOADS=false \
    RUNPOD_INIT_TIMEOUT=1800

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.runpod.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.runpod.txt

COPY handler.py runpod_inference.py ./
COPY qwenimage ./qwenimage

CMD ["python", "-u", "handler.py"]
