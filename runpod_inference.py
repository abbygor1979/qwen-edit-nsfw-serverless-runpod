from __future__ import annotations

import base64
import gc
import json
import os
import random
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Sequence

import requests
import torch
from huggingface_hub import InferenceClient, hf_hub_download
from PIL import Image
from safetensors.torch import load_file

RUNPOD_HF_CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")
DEFAULT_OUTPUT_DIR = Path("/tmp/runpod-output")
MAX_SEED = 2**31 - 1

SYSTEM_PROMPT = """
# Edit Instruction Rewriter
You are a professional edit instruction rewriter. Your task is to generate a precise, concise, and visually achievable professional-level edit instruction based on the user-provided instruction and the image to be edited.

Please strictly follow the rewriting rules below:

## 1. General Principles
- Keep the rewritten prompt concise and comprehensive. Avoid overly long sentences and unnecessary descriptive language.
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.
- Keep the main part of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.
- All added objects or modifications must align with the logic and style of the scene in the input images.
- If multiple sub-images are to be generated, describe the content of each sub-image individually.

## 2. Task-Type Handling Rules

### 1. Add, Delete, Replace Tasks
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.
- If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.).
- Remove meaningless instructions.
- For replacement tasks, specify "Replace Y with X" and briefly describe the key visual features of X.

### 2. Text Editing Tasks
- All text content must be enclosed in English double quotes.
- Keep the original language of the text, and keep the capitalization.
- Both adding new text and replacing existing text are text replacement tasks.

### 3. Human Editing Tasks
- Make the smallest changes to the given user's prompt.
- If changes to background, action, expression, camera shot, or ambient lighting are required, please list each modification individually.
- Edits to makeup or facial features or expression must be subtle, not exaggerated, and must preserve the subject's identity consistency.

### 4. Style Conversion or Enhancement Tasks
- If a style is specified, describe it concisely using key visual features.
- Colorization tasks and old photo restoration must use the fixed template:
  "Restore and colorize the old photo."
- Clearly specify the object to be modified.

### 5. Material Replacement
- Clearly specify the object and the material.

### 6. Logo and Pattern Editing
- Material replacement should preserve the original shape and structure as much as possible.
- When migrating logos or patterns to new scenes, ensure shape and structure consistency.

### 7. Multi-Image Tasks
- Rewritten prompts must clearly point out which image's element is being modified.
- For stylization tasks, describe the reference image's style in the rewritten prompt, while preserving the visual content of the source image.

## 3. Rationale and Logic Check
- Resolve contradictory instructions.
- Supplement missing critical information.

# Output Format Example
```json
{
  "Rewritten": "..."
}
```
"""

IDENTITY_LOCK_INSTRUCTION = (
    "Preserve the subject's face and identity exactly as in the source image. "
    "Do not change facial structure, skin texture, eye shape, eye color, eyebrows, nose, lips, teeth, ears, age, "
    "expression, head shape, hairstyle, hairline, or any distinguishing facial detail. "
    "Keep the face aligned, natural, and unchanged even if the user prompt requests otherwise."
)

IDENTITY_LOCK_NEGATIVE_PROMPT = (
    "changed face, different face, altered identity, new identity, face swap, different person, "
    "modified facial features, altered eyes, altered eyebrows, altered nose, altered lips, altered jawline, "
    "altered cheekbones, altered skin texture, altered hairline, altered hairstyle, de-aged face, aged face, "
    "beautified face, retouched face, distorted face, asymmetrical face, malformed face, duplicated face"
)


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _to_optional_int(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    return int(value)


def _to_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _listify(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _merge_prompt(user_prompt: str, enforce_identity_lock: bool) -> str:
    if not enforce_identity_lock:
        return user_prompt.strip()

    prompt = user_prompt.strip()
    if not prompt:
        return IDENTITY_LOCK_INSTRUCTION

    return f"{prompt}\n\nAdditional hard requirement: {IDENTITY_LOCK_INSTRUCTION}"


def _merge_negative_prompt(user_negative_prompt: str, enforce_identity_lock: bool) -> str:
    base_negative = user_negative_prompt.strip()
    if not enforce_identity_lock:
        return base_negative or " "

    if not base_negative:
        return IDENTITY_LOCK_NEGATIVE_PROMPT

    return f"{base_negative}, {IDENTITY_LOCK_NEGATIVE_PROMPT}"


def _image_bytes_to_data_uri(image_bytes: bytes, image_format: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    mime = f"image/{image_format.lower()}"
    return f"data:{mime};base64,{encoded}"


def _pil_to_bytes(image: Image.Image, image_format: str) -> bytes:
    buffer = BytesIO()
    save_image = image.convert("RGB") if image_format.upper() in {"JPEG", "JPG"} else image
    save_image.save(buffer, format=image_format.upper())
    return buffer.getvalue()


def _decode_base64_image(value: str) -> bytes:
    payload = value.split(",", 1)[1] if "," in value and value.startswith("data:") else value
    return base64.b64decode(payload)


def _open_pil_image(source: Any) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    if isinstance(source, dict):
        if source.get("url"):
            source = source["url"]
        elif source.get("base64"):
            source = source["base64"]
        elif source.get("path"):
            source = source["path"]
        else:
            raise ValueError("Image dictionary must contain one of: url, base64, path")

    if not isinstance(source, str):
        raise TypeError(f"Unsupported image input type: {type(source)!r}")

    if source.startswith(("http://", "https://")):
        response = requests.get(source, timeout=120)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    if source.startswith("data:image/"):
        return Image.open(BytesIO(_decode_base64_image(source))).convert("RGB")

    local_path = Path(source)
    if local_path.exists():
        return Image.open(local_path).convert("RGB")

    return Image.open(BytesIO(_decode_base64_image(source))).convert("RGB")


def _collect_input_images(job_input: Dict[str, Any]) -> List[Image.Image]:
    candidates: List[Any] = []
    candidates.extend(_listify(job_input.get("images")))
    candidates.extend(_listify(job_input.get("image")))
    candidates.extend(_listify(job_input.get("image_url")))
    candidates.extend(_listify(job_input.get("image_urls")))
    candidates.extend(_listify(job_input.get("image_base64")))
    candidates.extend(_listify(job_input.get("image_base64s")))
    candidates = [item for item in candidates if item not in (None, "")]

    if not candidates:
        raise ValueError(
            "At least one input image is required. Provide input.images or one of image_url/image_base64."
        )

    return [_open_pil_image(item) for item in candidates]


def resolve_snapshot_path(model_id: str, cache_root: Path = RUNPOD_HF_CACHE_ROOT) -> str:
    if "/" not in model_id:
        raise ValueError(f"model_id '{model_id}' must be in 'org/name' format")

    org, name = model_id.split("/", 1)
    model_root = cache_root / f"models--{org}--{name}"
    refs_main = model_root / "refs" / "main"
    snapshots_dir = model_root / "snapshots"

    if refs_main.is_file():
        snapshot_hash = refs_main.read_text(encoding="utf-8").strip()
        candidate = snapshots_dir / snapshot_hash
        if candidate.is_dir():
            return str(candidate)

    if snapshots_dir.is_dir():
        versions = sorted(entry for entry in snapshots_dir.iterdir() if entry.is_dir())
        if versions:
            return str(versions[0])

    raise FileNotFoundError(f"Cached model not found for '{model_id}' in '{cache_root}'")


def _try_resolve_cached_model(model_id: str) -> str | None:
    try:
        return resolve_snapshot_path(model_id)
    except Exception:
        return None


def _import_pipeline_class():
    try:
        from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline

        return QwenImageEditPlusPipeline
    except Exception:
        from diffusers import QwenImageEditPlusPipeline

        return QwenImageEditPlusPipeline


def _import_transformer_class():
    from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel

    return QwenImageTransformer2DModel


def _try_import_bucket_upload():
    try:
        from runpod.serverless.utils import rp_upload

        return rp_upload
    except Exception:
        return None


@dataclass(frozen=True)
class WorkerConfig:
    base_model_id: str = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
    base_model_path: str | None = os.environ.get("BASE_MODEL_PATH") or None
    checkpoint_repo_id: str = os.environ.get("CHECKPOINT_REPO_ID", "Phr00t/Qwen-Image-Edit-Rapid-AIO")
    checkpoint_filename: str = os.environ.get(
        "CHECKPOINT_FILENAME",
        "v23/Qwen-Rapid-AIO-NSFW-v23.safetensors",
    )
    checkpoint_local_path: str | None = os.environ.get("CHECKPOINT_LOCAL_PATH") or None
    checkpoint_revision: str | None = os.environ.get("CHECKPOINT_REVISION") or None
    hf_token: str | None = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None
    rewrite_api_key: str | None = (
        os.environ.get("HF_INFERENCE_API_KEY")
        or os.environ.get("INFERENCE_PROVIDERS_API_KEY")
        or os.environ.get("inference_providers")
        or None
    )
    rewrite_provider: str = os.environ.get("REWRITE_PROVIDER", "nebius")
    rewrite_model: str = os.environ.get("REWRITE_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")
    default_num_inference_steps: int = _to_int(os.environ.get("DEFAULT_NUM_INFERENCE_STEPS"), 4)
    default_true_guidance_scale: float = _to_float(os.environ.get("DEFAULT_TRUE_GUIDANCE_SCALE"), 1.0)
    default_rewrite_prompt: bool = _to_bool(os.environ.get("DEFAULT_REWRITE_PROMPT"), False)
    lock_face_identity: bool = _to_bool(os.environ.get("LOCK_FACE_IDENTITY"), True)
    use_cached_base_model: bool = _to_bool(os.environ.get("RUNPOD_USE_CACHED_BASE_MODEL"), True)
    enable_bucket_uploads: bool = _to_bool(os.environ.get("RUNPOD_ENABLE_BUCKET_UPLOADS"), False)
    output_dir: Path = Path(os.environ.get("RUNPOD_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    torch_dtype_name: str = os.environ.get("TORCH_DTYPE", "bfloat16")
    skip_model_load: bool = _to_bool(os.environ.get("RUNPOD_SKIP_MODEL_LOAD"), False)

    @property
    def torch_dtype(self) -> torch.dtype:
        if not hasattr(torch, self.torch_dtype_name):
            raise ValueError(f"Unsupported TORCH_DTYPE '{self.torch_dtype_name}'")
        return getattr(torch, self.torch_dtype_name)

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def generator_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"


class QwenRunpodService:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.pipe = None
        self._load_lock = Lock()
        self._load_seconds = 0.0

    def _resolve_base_model_source(self) -> tuple[str, bool]:
        if self.config.base_model_path:
            base_model_path = Path(self.config.base_model_path)
            if base_model_path.exists():
                return str(base_model_path), True

        if self.config.use_cached_base_model:
            cached_path = _try_resolve_cached_model(self.config.base_model_id)
            if cached_path:
                return cached_path, True

        return self.config.base_model_id, False

    def _resolve_checkpoint_path(self) -> str:
        if self.config.checkpoint_local_path:
            checkpoint_path = Path(self.config.checkpoint_local_path)
            if checkpoint_path.exists():
                return str(checkpoint_path)

        return hf_hub_download(
            repo_id=self.config.checkpoint_repo_id,
            filename=self.config.checkpoint_filename,
            repo_type="model",
            revision=self.config.checkpoint_revision,
            token=self.config.hf_token,
        )

    def _inject_checkpoint(self, checkpoint_path: str) -> None:
        print(f"[model] loading checkpoint weights from: {checkpoint_path}")
        state_dict = load_file(checkpoint_path)

        transformer_weights: Dict[str, torch.Tensor] = {}
        vae_weights: Dict[str, torch.Tensor] = {}
        text_encoder_weights: Dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            if key.startswith("model.diffusion_model."):
                transformer_weights[key.replace("model.diffusion_model.", "")] = value
            elif key.startswith("transformer."):
                transformer_weights[key.replace("transformer.", "")] = value
            elif key.startswith("first_stage_model."):
                vae_weights[key.replace("first_stage_model.", "")] = value
            elif key.startswith("vae."):
                vae_weights[key.replace("vae.", "")] = value
            elif "conditioner.embedders.0." in key:
                text_encoder_weights[key.replace("conditioner.embedders.0.", "")] = value
            elif key.startswith("text_encoder."):
                text_encoder_weights[key.replace("text_encoder.", "")] = value

        print(
            "[model] checkpoint split stats:",
            {
                "transformer": len(transformer_weights),
                "vae": len(vae_weights),
                "text_encoder": len(text_encoder_weights),
            },
        )

        if transformer_weights:
            msg = self.pipe.transformer.load_state_dict(transformer_weights, strict=False)
            print(
                "[model] transformer injected",
                {"missing_keys": len(msg.missing_keys), "unexpected_keys": len(msg.unexpected_keys)},
            )
        else:
            raise RuntimeError("No transformer weights were found in the checkpoint file.")

        if vae_weights:
            self.pipe.vae.load_state_dict(vae_weights, strict=False)

        if text_encoder_weights:
            self.pipe.text_encoder.load_state_dict(text_encoder_weights, strict=False)

        del state_dict
        del transformer_weights
        del vae_weights
        del text_encoder_weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_loaded(self) -> None:
        if self.pipe is not None or self.config.skip_model_load:
            return

        with self._load_lock:
            if self.pipe is not None or self.config.skip_model_load:
                return

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for this worker. No GPU is visible inside the container.")

            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

            pipeline_class = _import_pipeline_class()
            base_model_source, local_files_only = self._resolve_base_model_source()

            print(f"[model] loading base pipeline from: {base_model_source}")
            started_at = time.time()
            self.pipe = pipeline_class.from_pretrained(
                base_model_source,
                torch_dtype=self.config.torch_dtype,
                local_files_only=local_files_only,
                token=self.config.hf_token,
            ).to(self.config.device)
            transformer_class = _import_transformer_class()
            self.pipe.transformer.__class__ = transformer_class

            checkpoint_path = self._resolve_checkpoint_path()
            self._inject_checkpoint(checkpoint_path)
            self._load_seconds = round(time.time() - started_at, 3)
            print(f"[model] worker load complete in {self._load_seconds}s")

    def _rewrite_prompt(self, prompt: str, images: Sequence[Image.Image]) -> str:
        if not self.config.rewrite_api_key:
            return prompt

        try:
            client = InferenceClient(provider=self.config.rewrite_provider, api_key=self.config.rewrite_api_key)
            content: List[Dict[str, Any]] = [
                {
                    "type": "text",
                    "text": (
                        f"{SYSTEM_PROMPT}\n\nUser Input: {prompt}\n\nRewritten Prompt:"
                    ),
                }
            ]

            for image in images:
                image_bytes = _pil_to_bytes(image, "PNG")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_bytes_to_data_uri(image_bytes, "png")},
                    }
                )

            completion = client.chat.completions.create(
                model=self.config.rewrite_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that rewrites image editing instructions.",
                    },
                    {"role": "user", "content": content},
                ],
            )
            result = completion.choices[0].message.content or ""
            cleaned = result.replace("```json", "").replace("```", "").strip()

            if '"Rewritten"' in cleaned:
                parsed = json.loads(cleaned)
                return str(parsed.get("Rewritten", prompt)).strip()

            return cleaned or prompt
        except Exception as exc:
            print(f"[rewrite] prompt rewrite failed, falling back to original prompt: {exc}")
            return prompt

    def _serialize_output(
        self,
        job_id: str,
        images: Sequence[Image.Image],
        image_format: str,
        upload_to_bucket: bool,
    ) -> List[Dict[str, Any]]:
        image_format = image_format.lower()
        if image_format == "jpg":
            image_format = "jpeg"

        output_dir = self.config.output_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        rp_upload = _try_import_bucket_upload()
        use_bucket = upload_to_bucket and rp_upload is not None
        payloads: List[Dict[str, Any]] = []

        for index, image in enumerate(images):
            image_bytes = _pil_to_bytes(image, image_format)
            file_name = f"output_{index}.{image_format}"
            file_path = output_dir / file_name
            file_path.write_bytes(image_bytes)

            item: Dict[str, Any] = {
                "index": index,
                "width": image.width,
                "height": image.height,
                "format": image_format,
                "file_name": file_name,
            }

            if use_bucket:
                try:
                    item["image_url"] = rp_upload.upload_image(job_id, str(file_path))
                except Exception as exc:
                    print(f"[output] bucket upload failed, returning base64 instead: {exc}")
                    use_bucket = False

            if not use_bucket:
                item["image_url"] = _image_bytes_to_data_uri(image_bytes, image_format)

            payloads.append(item)

        return payloads

    def _cleanup(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def predict(self, job: Dict[str, Any]) -> Dict[str, Any]:
        job_input = job.get("input", {}) or {}
        prompt = str(job_input.get("prompt") or "").strip()
        if not prompt:
            return {"status": "error", "error": "`input.prompt` is required."}

        if self.config.skip_model_load:
            return {
                "status": "skipped",
                "message": "RUNPOD_SKIP_MODEL_LOAD=1 is set. Worker wiring is valid but model loading was skipped.",
                "input_echo": job_input,
            }

        images = _collect_input_images(job_input)
        self.ensure_loaded()

        seed = _to_int(job_input.get("seed"), 42)
        if _to_bool(job_input.get("randomize_seed"), False):
            seed = random.randint(0, MAX_SEED)

        true_guidance_scale = _to_float(
            job_input.get("true_guidance_scale"),
            self.config.default_true_guidance_scale,
        )
        num_inference_steps = _to_int(
            job_input.get("num_inference_steps"),
            self.config.default_num_inference_steps,
        )
        num_images_per_prompt = _to_int(job_input.get("num_images_per_prompt"), 1)
        rewrite_prompt = _to_bool(job_input.get("rewrite_prompt"), self.config.default_rewrite_prompt)
        enforce_identity_lock = _to_bool(job_input.get("lock_face_identity"), self.config.lock_face_identity)
        negative_prompt = _merge_negative_prompt(str(job_input.get("negative_prompt", " ")), enforce_identity_lock)
        height = _to_optional_int(job_input.get("height"))
        width = _to_optional_int(job_input.get("width"))
        output_format = str(job_input.get("output_format", "png")).strip().lower()
        upload_to_bucket = _to_bool(job_input.get("upload_to_bucket"), self.config.enable_bucket_uploads)
        resolved_prompt = self._rewrite_prompt(prompt, images) if rewrite_prompt else prompt
        resolved_prompt = _merge_prompt(resolved_prompt, enforce_identity_lock)

        generator = torch.Generator(device=self.config.generator_device).manual_seed(seed)
        started_at = time.time()

        with torch.inference_mode():
            output = self.pipe(
                image=images,
                prompt=resolved_prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                generator=generator,
                true_cfg_scale=true_guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
            )

        image_payloads = self._serialize_output(
            job_id=str(job.get("id", f"job-{int(time.time())}")),
            images=output.images,
            image_format=output_format,
            upload_to_bucket=upload_to_bucket,
        )
        inference_seconds = round(time.time() - started_at, 3)
        self._cleanup()

        return {
            "status": "success",
            "seed": seed,
            "prompt": prompt,
            "resolved_prompt": resolved_prompt,
            "negative_prompt": negative_prompt,
            "num_images": len(image_payloads),
            "images": image_payloads,
            "timings": {
                "worker_load_seconds": self._load_seconds,
                "inference_seconds": inference_seconds,
            },
            "model": {
                "base_model_id": self.config.base_model_id,
                "checkpoint_repo_id": self.config.checkpoint_repo_id,
                "checkpoint_filename": self.config.checkpoint_filename,
            },
        }


_SERVICE = QwenRunpodService(WorkerConfig())


def handle_job(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return _SERVICE.predict(job)
    except Exception as exc:
        print(f"[handler] job failed: {exc}")
        return {
            "status": "error",
            "error": str(exc),
        }
