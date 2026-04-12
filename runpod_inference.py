from __future__ import annotations

import base64
import gc
import json
import math
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
from PIL import Image, ImageEnhance, ImageFilter
from safetensors.torch import load_file

RUNPOD_HF_CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")
DEFAULT_OUTPUT_DIR = Path("/tmp/runpod-output")
MAX_SEED = 2**31 - 1
DEFAULT_MIN_OUTPUT_LONG_EDGE = 1920
DEFAULT_MIN_OUTPUT_SHORT_EDGE = 1080
DEFAULT_MIN_OUTPUT_PIXELS = 1920 * 1080
DEFAULT_NATIVE_MIN_LONG_EDGE = 1536
DEFAULT_NATIVE_MIN_SHORT_EDGE = 1216
DEFAULT_NATIVE_MIN_PIXELS = 1216 * 1792
DEFAULT_NATIVE_MAX_LONG_EDGE = 2048
DEFAULT_GENERATION_SIZE_MULTIPLE = 32
DEFAULT_QUALITY_MODE = "balanced"
DEFAULT_POSTPROCESS_UPSCALE_MODE = "detail"
DEFAULT_FACE_MASK_STRATEGY = "smart"
DEFAULT_FACE_MASK_STRENGTH = 0.86

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


def _cache_root_candidates() -> List[Path]:
    candidates: List[Path] = [RUNPOD_HF_CACHE_ROOT]
    env_hf_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    env_transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
    env_hf_home = os.environ.get("HF_HOME")

    for value in (env_hf_cache, env_transformers_cache):
        if value:
            candidates.append(Path(value))

    if env_hf_home:
        candidates.append(Path(env_hf_home) / "hub")

    candidates.extend(
        [
            Path("/runpod-volume/hf-home/hub"),
            Path("/tmp/hf-home/hub"),
        ]
    )

    unique: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _ensure_effective_true_cfg_scale(
    requested_scale: float,
    enforce_identity_lock: bool,
    face_mask_mode: str,
    minimum_identity_scale: float,
) -> float:
    if not enforce_identity_lock and face_mask_mode == "off":
        return requested_scale

    if requested_scale > 1.0:
        return requested_scale

    return minimum_identity_scale


def _has_explicit_value(job_input: Dict[str, Any], key: str) -> bool:
    return key in job_input and job_input.get(key) not in (None, "", "null")


def _normalize_choice(value: Any, allowed: set[str], default: str) -> str:
    normalized = str(value or default).strip().lower()
    return normalized if normalized in allowed else default


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _infer_prompt_intent(prompt: str) -> str:
    lowered = prompt.lower()
    surface_keywords = (
        "water drop",
        "water droplet",
        "droplets",
        "droplet",
        "wet",
        "sweat",
        "tears",
        "tear",
        "moist",
        "dewy",
        "glitter",
        "shimmer",
        "sparkle",
        "glistening",
        "lashes",
        "eyelash",
        "eyeliner",
        "mascara",
        "makeup",
        "lipstick",
        "blush",
        "freckles",
    )
    text_keywords = ("text", "sign", "caption", "logo", "lettering", "label", "word", "words")
    background_keywords = ("background", "wall", "sky", "room", "scene", "behind", "backdrop")
    if any(keyword in lowered for keyword in surface_keywords):
        return "surface_fx"
    if any(keyword in lowered for keyword in text_keywords):
        return "text"
    if any(keyword in lowered for keyword in background_keywords):
        return "background"
    return "general"


def _normalize_quality_mode(value: Any) -> str:
    return _normalize_choice(value, {"speed", "balanced", "quality"}, DEFAULT_QUALITY_MODE)


def _normalize_mask_strategy(value: Any) -> str:
    return _normalize_choice(value, {"auto", "smart", "legacy"}, DEFAULT_FACE_MASK_STRATEGY)


def _normalize_mask_mode(value: Any) -> str:
    return _normalize_choice(value, {"balanced", "strict", "surface_fx", "off"}, "surface_fx")


def _normalize_upscale_mode(value: Any) -> str:
    return _normalize_choice(value, {"off", "classic", "detail", "auto"}, DEFAULT_POSTPROCESS_UPSCALE_MODE)


def _resolve_native_constraints(
    quality_mode: str,
    face_coverage: float | None,
    minimum_long_edge: int,
    minimum_short_edge: int,
    minimum_pixels: int,
    maximum_long_edge: int,
) -> tuple[int, int, int, int]:
    long_edge = minimum_long_edge
    short_edge = minimum_short_edge
    min_pixels = minimum_pixels
    max_long_edge = maximum_long_edge

    if quality_mode == "speed":
        min_pixels = int(min_pixels * 0.8)
        long_edge = int(long_edge * 0.92)
        short_edge = int(short_edge * 0.92)
    elif quality_mode == "quality":
        min_pixels = int(min_pixels * 1.18)
        long_edge = int(long_edge * 1.08)
        short_edge = int(short_edge * 1.08)

    if face_coverage is not None:
        if face_coverage < 0.14:
            min_pixels = int(min_pixels * 1.2)
            long_edge = max(long_edge, 1664)
            short_edge = max(short_edge, 1280)
        elif face_coverage > 0.34:
            min_pixels = int(min_pixels * 0.92)

    return long_edge, short_edge, min_pixels, max_long_edge


def _resolve_auto_steps(
    quality_mode: str,
    prompt_intent: str,
    width: int,
    height: int,
    face_coverage: float | None,
) -> int:
    base_steps = {"speed": 4, "balanced": 6, "quality": 8}[quality_mode]
    megapixels = (width * height) / 1_000_000.0
    if megapixels > 2.2:
        base_steps += 1
    if megapixels > 2.8 and quality_mode != "speed":
        base_steps += 1
    if face_coverage is not None and face_coverage < 0.14:
        base_steps += 1
    if prompt_intent == "text":
        base_steps += 1
    return max(4, min(10, base_steps))


def _resolve_auto_true_cfg_scale(
    quality_mode: str,
    prompt_intent: str,
    enforce_identity_lock: bool,
    face_mask_mode: str,
    minimum_identity_scale: float,
) -> float:
    base = {"speed": 1.2, "balanced": 1.35, "quality": 1.45}[quality_mode]
    if prompt_intent == "text":
        base += 0.35
    elif prompt_intent == "background":
        base += 0.15
    elif prompt_intent == "surface_fx":
        base -= 0.1

    resolved = _clamp_float(base, 1.0, 2.25)
    return _ensure_effective_true_cfg_scale(
        requested_scale=resolved,
        enforce_identity_lock=enforce_identity_lock,
        face_mask_mode=face_mask_mode,
        minimum_identity_scale=minimum_identity_scale,
    )


def _has_bucket_configured() -> bool:
    required_keys = ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY")
    return all(bool(os.environ.get(key)) for key in required_keys)


def _is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _image_bytes_to_data_uri(image_bytes: bytes, image_format: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    mime = f"image/{image_format.lower()}"
    return f"data:{mime};base64,{encoded}"


def _pil_to_bytes(image: Image.Image, image_format: str) -> bytes:
    buffer = BytesIO()
    save_image = image.convert("RGB") if image_format.upper() in {"JPEG", "JPG"} else image
    save_image.save(buffer, format=image_format.upper())
    return buffer.getvalue()


def _align_dimension(value: float, multiple: int, round_up: bool = True) -> int:
    if multiple <= 1:
        rounded = math.ceil(value) if round_up else math.floor(value)
        return max(1, int(rounded))

    scaled = value / multiple
    rounded = math.ceil(scaled) if round_up else math.floor(scaled)
    return max(multiple, int(rounded) * multiple)


def _resolve_generation_size(
    requested_width: int | None,
    requested_height: int | None,
    reference_image: Image.Image | None,
    minimum_long_edge: int,
    minimum_short_edge: int,
    minimum_pixels: int,
    maximum_long_edge: int,
    size_multiple: int,
) -> tuple[int, int]:
    aspect_ratio = 1.0
    if reference_image is not None and reference_image.width > 0 and reference_image.height > 0:
        aspect_ratio = reference_image.width / reference_image.height
    aspect_ratio = max(aspect_ratio, 1e-3)

    if requested_width and requested_height:
        return (
            _align_dimension(requested_width, size_multiple, round_up=True),
            _align_dimension(requested_height, size_multiple, round_up=True),
        )

    if requested_width and not requested_height:
        resolved_height = max(1, int(round(requested_width / aspect_ratio)))
        return (
            _align_dimension(requested_width, size_multiple, round_up=True),
            _align_dimension(resolved_height, size_multiple, round_up=True),
        )

    if requested_height and not requested_width:
        resolved_width = max(1, int(round(requested_height * aspect_ratio)))
        return (
            _align_dimension(resolved_width, size_multiple, round_up=True),
            _align_dimension(requested_height, size_multiple, round_up=True),
        )

    if aspect_ratio >= 1.0:
        width_norm = aspect_ratio
        height_norm = 1.0
    else:
        width_norm = 1.0
        height_norm = 1.0 / aspect_ratio

    long_norm = max(width_norm, height_norm)
    short_norm = min(width_norm, height_norm)
    scale = 1.0

    if minimum_short_edge > 0:
        scale = max(scale, minimum_short_edge / short_norm)
    if minimum_long_edge > 0:
        scale = max(scale, minimum_long_edge / long_norm)
    if minimum_pixels > 0:
        scale = max(scale, math.sqrt(minimum_pixels / (width_norm * height_norm)))

    width = width_norm * scale
    height = height_norm * scale
    align_up = True

    if maximum_long_edge > 0 and max(width, height) > maximum_long_edge:
        cap_scale = maximum_long_edge / max(width, height)
        width *= cap_scale
        height *= cap_scale
        align_up = False

    resolved_width = _align_dimension(width, size_multiple, round_up=align_up)
    resolved_height = _align_dimension(height, size_multiple, round_up=align_up)

    if maximum_long_edge > 0:
        max_aligned_long_edge = _align_dimension(maximum_long_edge, size_multiple, round_up=False)
        if resolved_width >= resolved_height and resolved_width > max_aligned_long_edge:
            scale_down = max_aligned_long_edge / resolved_width
            resolved_width = max_aligned_long_edge
            resolved_height = _align_dimension(resolved_height * scale_down, size_multiple, round_up=False)
        elif resolved_height > resolved_width and resolved_height > max_aligned_long_edge:
            scale_down = max_aligned_long_edge / resolved_height
            resolved_height = max_aligned_long_edge
            resolved_width = _align_dimension(resolved_width * scale_down, size_multiple, round_up=False)

    return resolved_width, resolved_height


def _postprocess_upscaled_image(image: Image.Image, scale: float, upscale_mode: str) -> tuple[Image.Image, Dict[str, Any]]:
    normalized_mode = _normalize_upscale_mode(upscale_mode)
    if normalized_mode == "off":
        return image, {"postprocessed": False, "upscale_mode": "off"}

    processed = image
    if normalized_mode in {"auto", "detail"}:
        sharpen_percent = 120 if scale >= 1.4 else 90
        sharpen_radius = max(1, int(round(scale * 1.5)))
        processed = processed.filter(ImageFilter.UnsharpMask(radius=sharpen_radius, percent=sharpen_percent, threshold=3))
        processed = ImageEnhance.Contrast(processed).enhance(1.03)
        return processed, {"postprocessed": True, "upscale_mode": "detail"}

    return processed, {"postprocessed": False, "upscale_mode": "classic"}


def _finalize_output_resolution(
    image: Image.Image,
    minimum_long_edge: int,
    minimum_short_edge: int,
    minimum_pixels: int,
    upscale_mode: str,
) -> tuple[Image.Image, Dict[str, Any]]:
    width, height = image.size
    if width <= 0 or height <= 0:
        return image, {"upscaled": False, "original_width": width, "original_height": height}

    long_edge = max(width, height)
    short_edge = min(width, height)
    area = width * height

    scale = 1.0
    if minimum_long_edge > 0 and long_edge < minimum_long_edge:
        scale = max(scale, minimum_long_edge / long_edge)
    if minimum_short_edge > 0 and short_edge < minimum_short_edge:
        scale = max(scale, minimum_short_edge / short_edge)
    if minimum_pixels > 0 and area < minimum_pixels:
        scale = max(scale, math.sqrt(minimum_pixels / area))

    if scale <= 1.0:
        processed, postprocess_meta = _postprocess_upscaled_image(image, scale=1.0, upscale_mode=upscale_mode)
        return processed, {"upscaled": False, "original_width": width, "original_height": height, **postprocess_meta}

    resized = image.resize(
        (int(math.ceil(width * scale)), int(math.ceil(height * scale))),
        resample=Image.Resampling.LANCZOS,
    )
    processed, postprocess_meta = _postprocess_upscaled_image(resized, scale=scale, upscale_mode=upscale_mode)
    return processed, {
        "upscaled": True,
        "original_width": width,
        "original_height": height,
        "upscale_scale": round(scale, 4),
        **postprocess_meta,
    }


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
    for cache_root in _cache_root_candidates():
        try:
            resolved_path = resolve_snapshot_path(model_id, cache_root=cache_root)
            print(f"[model] found cached snapshot under: {cache_root}")
            return resolved_path
        except Exception:
            continue
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


def _import_face_masker_class():
    from face_masking import FaceIdentityMasker

    return FaceIdentityMasker


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
    default_num_inference_steps: int = _to_int(os.environ.get("DEFAULT_NUM_INFERENCE_STEPS"), 6)
    default_true_guidance_scale: float = _to_float(os.environ.get("DEFAULT_TRUE_GUIDANCE_SCALE"), 1.3)
    minimum_identity_true_guidance_scale: float = _to_float(
        os.environ.get("MIN_IDENTITY_TRUE_GUIDANCE_SCALE"),
        1.3,
    )
    default_rewrite_prompt: bool = _to_bool(os.environ.get("DEFAULT_REWRITE_PROMPT"), False)
    lock_face_identity: bool = _to_bool(os.environ.get("LOCK_FACE_IDENTITY"), True)
    face_mask_strategy: str = _normalize_mask_strategy(os.environ.get("FACE_MASK_STRATEGY", DEFAULT_FACE_MASK_STRATEGY))
    face_mask_mode: str = _normalize_mask_mode(os.environ.get("FACE_MASK_MODE", "surface_fx"))
    face_mask_strength: float = _clamp_float(
        _to_float(os.environ.get("FACE_MASK_STRENGTH"), DEFAULT_FACE_MASK_STRENGTH),
        0.0,
        1.0,
    )
    face_mask_debug: bool = _to_bool(os.environ.get("FACE_MASK_DEBUG"), False)
    quality_mode: str = _normalize_quality_mode(os.environ.get("QUALITY_MODE", DEFAULT_QUALITY_MODE))
    adaptive_generation: bool = _to_bool(os.environ.get("ADAPTIVE_GENERATION"), True)
    minimum_native_long_edge: int = _to_int(
        os.environ.get("MIN_NATIVE_LONG_EDGE"),
        DEFAULT_NATIVE_MIN_LONG_EDGE,
    )
    minimum_native_short_edge: int = _to_int(
        os.environ.get("MIN_NATIVE_SHORT_EDGE"),
        DEFAULT_NATIVE_MIN_SHORT_EDGE,
    )
    minimum_native_pixels: int = _to_int(
        os.environ.get("MIN_NATIVE_PIXELS"),
        DEFAULT_NATIVE_MIN_PIXELS,
    )
    maximum_native_long_edge: int = _to_int(
        os.environ.get("MAX_NATIVE_LONG_EDGE"),
        DEFAULT_NATIVE_MAX_LONG_EDGE,
    )
    generation_size_multiple: int = _to_int(
        os.environ.get("GENERATION_SIZE_MULTIPLE"),
        DEFAULT_GENERATION_SIZE_MULTIPLE,
    )
    minimum_output_long_edge: int = _to_int(
        os.environ.get("MIN_OUTPUT_LONG_EDGE"),
        DEFAULT_MIN_OUTPUT_LONG_EDGE,
    )
    minimum_output_short_edge: int = _to_int(
        os.environ.get("MIN_OUTPUT_SHORT_EDGE"),
        DEFAULT_MIN_OUTPUT_SHORT_EDGE,
    )
    minimum_output_pixels: int = _to_int(
        os.environ.get("MIN_OUTPUT_PIXELS"),
        DEFAULT_MIN_OUTPUT_PIXELS,
    )
    postprocess_upscale_mode: str = _normalize_upscale_mode(
        os.environ.get("POSTPROCESS_UPSCALE_MODE", DEFAULT_POSTPROCESS_UPSCALE_MODE)
    )
    use_cached_base_model: bool = _to_bool(os.environ.get("RUNPOD_USE_CACHED_BASE_MODEL"), True)
    enable_bucket_uploads: bool = _to_bool(os.environ.get("RUNPOD_ENABLE_BUCKET_UPLOADS"), _has_bucket_configured())
    oom_retry_attempts: int = _to_int(os.environ.get("OOM_RETRY_ATTEMPTS"), 2)
    oom_retry_scale: float = _clamp_float(_to_float(os.environ.get("OOM_RETRY_SCALE"), 0.86), 0.5, 0.95)
    oom_retry_min_steps: int = _to_int(os.environ.get("OOM_RETRY_MIN_STEPS"), 4)
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
        self.face_masker = None
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
                print(f"[model] using local cached base model snapshot: {cached_path}")
                return cached_path, True
            print("[model] cached base model snapshot not found locally, falling back to Hugging Face download")

        return self.config.base_model_id, False

    def _resolve_checkpoint_path(self) -> str:
        if self.config.checkpoint_local_path:
            checkpoint_path = Path(self.config.checkpoint_local_path)
            if checkpoint_path.exists():
                print(f"[model] using local checkpoint path: {checkpoint_path}")
                return str(checkpoint_path)

        print(f"[model] downloading checkpoint from Hugging Face: {self.config.checkpoint_repo_id}/{self.config.checkpoint_filename}")
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
            try:
                if hasattr(self.pipe, "enable_vae_tiling"):
                    self.pipe.enable_vae_tiling()
                elif hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_tiling"):
                    self.pipe.vae.enable_tiling()
                print("[model] enabled VAE tiling")
            except Exception as exc:
                print(f"[model] could not enable VAE tiling: {exc}")

            try:
                if hasattr(self.pipe, "enable_vae_slicing"):
                    self.pipe.enable_vae_slicing()
                elif hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_slicing"):
                    self.pipe.vae.enable_slicing()
                print("[model] enabled VAE slicing")
            except Exception as exc:
                print(f"[model] could not enable VAE slicing: {exc}")

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

    def _get_face_masker(self):
        if self.face_masker is not None:
            return self.face_masker

        try:
            masker_class = _import_face_masker_class()
            self.face_masker = masker_class()
        except Exception as exc:
            print(f"[mask] face masking unavailable, continuing without it: {exc}")
            self.face_masker = False

        return self.face_masker

    def _estimate_face_coverage(self, image: Image.Image) -> float | None:
        masker = self._get_face_masker()
        if masker is False:
            return None

        try:
            return masker.estimate_face_coverage(image)
        except Exception as exc:
            print(f"[mask] face coverage estimation failed: {exc}")
            return None

    def _generate_with_retries(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        num_inference_steps: int,
        true_guidance_scale: float,
        num_images_per_prompt: int,
    ) -> tuple[Any, List[Dict[str, Any]], int, int, int]:
        attempt_width = width
        attempt_height = height
        attempt_steps = num_inference_steps
        attempts: List[Dict[str, Any]] = []

        for attempt_index in range(self.config.oom_retry_attempts + 1):
            try:
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "width": attempt_width,
                        "height": attempt_height,
                        "num_inference_steps": attempt_steps,
                        "status": "running",
                    }
                )
                generator = torch.Generator(device=self.config.generator_device).manual_seed(seed)
                output = self.pipe(
                    image=images,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=attempt_height,
                    width=attempt_width,
                    num_inference_steps=attempt_steps,
                    generator=generator,
                    true_cfg_scale=true_guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                )
                attempts[-1]["status"] = "completed"
                return output, attempts, attempt_width, attempt_height, attempt_steps
            except RuntimeError as exc:
                attempts[-1]["status"] = "failed"
                attempts[-1]["error"] = str(exc)
                if not _is_oom_error(exc) or attempt_index >= self.config.oom_retry_attempts:
                    raise

                print(f"[generation] OOM on attempt {attempt_index + 1}; retrying with smaller settings")
                self._cleanup()
                attempt_width = _align_dimension(attempt_width * self.config.oom_retry_scale, self.config.generation_size_multiple, round_up=False)
                attempt_height = _align_dimension(attempt_height * self.config.oom_retry_scale, self.config.generation_size_multiple, round_up=False)
                attempt_steps = max(self.config.oom_retry_min_steps, attempt_steps - 1)

        raise RuntimeError("generation-retries-exhausted")

    def _apply_face_masking(
        self,
        source_image: Image.Image,
        generated_images: Sequence[Image.Image],
        mode: str,
        strategy: str,
        strength: float,
        debug_masks: bool,
    ) -> tuple[List[Image.Image], List[Dict[str, Any]], List[Dict[str, Image.Image]]]:
        normalized_mode = (mode or "off").strip().lower()
        if normalized_mode == "off":
            metadata = [{"applied": False, "mode": "off", "reason": "disabled", "engine": "none"} for _ in generated_images]
            return list(generated_images), metadata, [{} for _ in generated_images]

        masker = self._get_face_masker()
        if masker is False:
            metadata = [
                {"applied": False, "mode": normalized_mode, "reason": "masker-unavailable", "engine": "none"}
                for _ in generated_images
            ]
            return list(generated_images), metadata, [{} for _ in generated_images]

        protected_images: List[Image.Image] = []
        metadata: List[Dict[str, Any]] = []
        debug_payloads: List[Dict[str, Image.Image]] = []
        for image in generated_images:
            result = masker.protect(
                source_image=source_image,
                generated_image=image,
                mode=normalized_mode,
                strategy=strategy,
                strength=strength,
                debug=debug_masks,
            )
            protected_images.append(result.image)
            metadata.append(
                {
                    "applied": result.applied,
                    "mode": result.mode,
                    "reason": result.reason,
                    "engine": result.engine,
                    **result.metadata,
                }
            )
            debug_payloads.append(result.debug_images)

        return protected_images, metadata, debug_payloads

    def _serialize_named_images(
        self,
        job_id: str,
        group_name: str,
        named_images: Dict[str, Image.Image],
        upload_to_bucket: bool,
    ) -> List[Dict[str, Any]]:
        output_dir = self.config.output_dir / job_id / group_name
        output_dir.mkdir(parents=True, exist_ok=True)
        rp_upload = _try_import_bucket_upload()
        use_bucket = upload_to_bucket and rp_upload is not None
        payloads: List[Dict[str, Any]] = []

        for image_name, image in named_images.items():
            image_bytes = _pil_to_bytes(image, "png")
            file_name = f"{image_name}.png"
            file_path = output_dir / file_name
            file_path.write_bytes(image_bytes)
            item = {
                "name": image_name,
                "width": image.width,
                "height": image.height,
                "format": "png",
                "file_name": file_name,
            }
            if use_bucket:
                try:
                    item["image_url"] = rp_upload.upload_image(job_id, str(file_path))
                except Exception as exc:
                    print(f"[output] debug image bucket upload failed, returning base64 instead: {exc}")
                    use_bucket = False
            if not use_bucket:
                item["image_url"] = _image_bytes_to_data_uri(image_bytes, "png")
            payloads.append(item)

        return payloads

    def _serialize_output(
        self,
        job_id: str,
        images: Sequence[Image.Image],
        image_format: str,
        upload_to_bucket: bool,
        upscale_mode: str,
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
            image, output_resolution = _finalize_output_resolution(
                image=image,
                minimum_long_edge=self.config.minimum_output_long_edge,
                minimum_short_edge=self.config.minimum_output_short_edge,
                minimum_pixels=self.config.minimum_output_pixels,
                upscale_mode=upscale_mode,
            )
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
                **output_resolution,
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
        job_id = str(job.get("id", f"job-{int(time.time())}"))
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

        quality_mode = _normalize_quality_mode(job_input.get("quality_mode", self.config.quality_mode))
        num_images_per_prompt = _to_int(job_input.get("num_images_per_prompt"), 1)
        rewrite_prompt = _to_bool(job_input.get("rewrite_prompt"), self.config.default_rewrite_prompt)
        enforce_identity_lock = _to_bool(job_input.get("lock_face_identity"), self.config.lock_face_identity)
        face_mask_mode = _normalize_mask_mode(job_input.get("face_mask_mode", self.config.face_mask_mode))
        face_mask_strategy = _normalize_mask_strategy(job_input.get("face_mask_strategy", self.config.face_mask_strategy))
        face_mask_strength = _clamp_float(
            _to_float(job_input.get("face_mask_strength"), self.config.face_mask_strength),
            0.0,
            1.0,
        )
        debug_masks = _to_bool(job_input.get("debug_masks"), self.config.face_mask_debug)
        postprocess_upscale_mode = _normalize_upscale_mode(
            job_input.get("postprocess_upscale_mode", self.config.postprocess_upscale_mode)
        )
        prompt_intent = _infer_prompt_intent(prompt)
        face_coverage = self._estimate_face_coverage(images[0]) if self.config.adaptive_generation and images else None

        negative_prompt = _merge_negative_prompt(str(job_input.get("negative_prompt", " ")), enforce_identity_lock)
        requested_height = _to_optional_int(job_input.get("height"))
        requested_width = _to_optional_int(job_input.get("width"))
        native_min_long, native_min_short, native_min_pixels, native_max_long = _resolve_native_constraints(
            quality_mode=quality_mode if self.config.adaptive_generation else self.config.quality_mode,
            face_coverage=face_coverage,
            minimum_long_edge=self.config.minimum_native_long_edge,
            minimum_short_edge=self.config.minimum_native_short_edge,
            minimum_pixels=self.config.minimum_native_pixels,
            maximum_long_edge=self.config.maximum_native_long_edge,
        )
        width, height = _resolve_generation_size(
            requested_width=requested_width,
            requested_height=requested_height,
            reference_image=images[0] if images else None,
            minimum_long_edge=native_min_long,
            minimum_short_edge=native_min_short,
            minimum_pixels=native_min_pixels,
            maximum_long_edge=native_max_long,
            size_multiple=self.config.generation_size_multiple,
        )
        explicit_guidance = _has_explicit_value(job_input, "true_guidance_scale")
        explicit_steps = _has_explicit_value(job_input, "num_inference_steps")
        true_guidance_scale = (
            _to_float(job_input.get("true_guidance_scale"), self.config.default_true_guidance_scale)
            if explicit_guidance
            else _resolve_auto_true_cfg_scale(
                quality_mode=quality_mode,
                prompt_intent=prompt_intent,
                enforce_identity_lock=enforce_identity_lock,
                face_mask_mode=face_mask_mode,
                minimum_identity_scale=self.config.minimum_identity_true_guidance_scale,
            )
        )
        true_guidance_scale = _ensure_effective_true_cfg_scale(
            requested_scale=true_guidance_scale,
            enforce_identity_lock=enforce_identity_lock,
            face_mask_mode=face_mask_mode,
            minimum_identity_scale=self.config.minimum_identity_true_guidance_scale,
        )
        num_inference_steps = (
            _to_int(job_input.get("num_inference_steps"), self.config.default_num_inference_steps)
            if explicit_steps
            else _resolve_auto_steps(
                quality_mode=quality_mode,
                prompt_intent=prompt_intent,
                width=width,
                height=height,
                face_coverage=face_coverage,
            )
        )
        output_format = str(job_input.get("output_format", "png")).strip().lower()
        upload_to_bucket = _to_bool(job_input.get("upload_to_bucket"), self.config.enable_bucket_uploads)
        resolved_prompt = self._rewrite_prompt(prompt, images) if rewrite_prompt else prompt
        resolved_prompt = _merge_prompt(resolved_prompt, enforce_identity_lock)
        print(
            f"[generation] native size {width}x{height}, steps={num_inference_steps}, "
            f"true_cfg_scale={true_guidance_scale}, quality_mode={quality_mode}, "
            f"mask={face_mask_strategy}/{face_mask_mode}@{round(face_mask_strength, 3)}"
        )

        started_at = time.time()

        with torch.inference_mode():
            output, generation_attempts, width, height, num_inference_steps = self._generate_with_retries(
                images=images,
                prompt=resolved_prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                true_guidance_scale=true_guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
            )

        output_images = list(output.images)
        face_masking: List[Dict[str, Any]]
        debug_mask_payloads: List[Dict[str, Image.Image]]
        if images:
            output_images, face_masking, debug_mask_payloads = self._apply_face_masking(
                source_image=images[0],
                generated_images=output_images,
                mode=face_mask_mode,
                strategy=face_mask_strategy,
                strength=face_mask_strength,
                debug_masks=debug_masks,
            )
        else:
            face_masking = [
                {
                    "applied": False,
                    "mode": face_mask_mode,
                    "reason": "no-source-image",
                    "engine": "none",
                    "strategy_requested": face_mask_strategy,
                }
                for _ in output_images
            ]
            debug_mask_payloads = [{} for _ in output_images]

        image_payloads = self._serialize_output(
            job_id=job_id,
            images=output_images,
            image_format=output_format,
            upload_to_bucket=upload_to_bucket,
            upscale_mode=postprocess_upscale_mode,
        )
        serialized_debug_masks: List[Dict[str, Any]] = []
        if debug_masks:
            for index, debug_images in enumerate(debug_mask_payloads):
                if not debug_images:
                    continue
                serialized_debug_masks.append(
                    {
                        "index": index,
                        "items": self._serialize_named_images(
                            job_id=job_id,
                            group_name=f"debug_{index}",
                            named_images=debug_images,
                            upload_to_bucket=upload_to_bucket,
                        ),
                    }
                )
        inference_seconds = round(time.time() - started_at, 3)
        self._cleanup()

        return {
            "status": "success",
            "seed": seed,
            "prompt": prompt,
            "resolved_prompt": resolved_prompt,
            "negative_prompt": negative_prompt,
            "face_mask_mode": face_mask_mode,
            "face_mask_strategy": face_mask_strategy,
            "face_masking": face_masking,
            "debug_masks": serialized_debug_masks,
            "num_images": len(image_payloads),
            "images": image_payloads,
            "generation": {
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "true_guidance_scale": true_guidance_scale,
                "quality_mode": quality_mode,
                "prompt_intent": prompt_intent,
                "face_coverage": round(face_coverage, 4) if face_coverage is not None else None,
                "attempts": generation_attempts,
            },
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
