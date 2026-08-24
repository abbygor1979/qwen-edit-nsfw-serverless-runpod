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
    DEFAULT_NUM_INFERENCE_STEPS=6 \
    DEFAULT_TRUE_GUIDANCE_SCALE=1.3 \
    MIN_IDENTITY_TRUE_GUIDANCE_SCALE=1.3 \
    DEFAULT_REWRITE_PROMPT=false \
    FACE_MASK_STRATEGY=smart \
    FACE_MASK_MODE=surface_fx \
    FACE_MASK_STRENGTH=0.86 \
    FACE_MASK_DEBUG=false \
    QUALITY_MODE=balanced \
    ADAPTIVE_GENERATION=true \
    MIN_NATIVE_LONG_EDGE=1536 \
    MIN_NATIVE_SHORT_EDGE=1216 \
    MIN_NATIVE_PIXELS=2179072 \
    MAX_NATIVE_LONG_EDGE=2048 \
    GENERATION_SIZE_MULTIPLE=32 \
    MIN_OUTPUT_LONG_EDGE=1920 \
    MIN_OUTPUT_SHORT_EDGE=1080 \
    MIN_OUTPUT_PIXELS=2073600 \
    POSTPROCESS_UPSCALE_MODE=detail \
    RUNPOD_USE_CACHED_BASE_MODEL=true \
    OOM_RETRY_ATTEMPTS=2 \
    OOM_RETRY_SCALE=0.86 \
    OOM_RETRY_MIN_STEPS=4 \
    RUNPOD_INIT_TIMEOUT=1800

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.runpod.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --ignore-installed -r requirements.runpod.txt

# torchaudio ships in the base image for an audio use case this worker never
# touches. Its compiled libtorchaudio.so fails to load in this environment,
# and diffusers' qwenimage pipeline import chain does not tolerate "installed
# but broken" the way it tolerates "not installed" (no try/except around the
# load, only around the ImportError). Removing the package entirely turns an
# unhandled native-library crash into a plain, already-handled ImportError.
RUN pip uninstall -y torchaudio || true

COPY handler.py runpod_inference.py face_masking.py ./
COPY qwenimage ./qwenimage

CMD ["python", "-u", "handler.py"]
