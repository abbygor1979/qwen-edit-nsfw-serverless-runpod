from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


try:
    import mediapipe as mp
except Exception:  # pragma: no cover - dependency is only required in the Runpod worker image
    mp = None

try:
    import torch
    import facer
except Exception:  # pragma: no cover - parser is lazily loaded in the Runpod worker image
    torch = None
    facer = None


NOSE_INDICES = [1, 2, 4, 5, 6, 19, 45, 48, 49, 64, 94, 97, 98, 115, 168, 195, 197, 220, 275, 279, 289, 344, 440]
LEFT_EYE_CENTER_INDICES = [33, 133, 159, 145]
RIGHT_EYE_CENTER_INDICES = [362, 263, 386, 374]
MOUTH_CORNER_LEFT_INDEX = 61
MOUTH_CORNER_RIGHT_INDEX = 291
NOSE_TIP_INDEX = 1

PARSER_SKIN_LABELS = {"face"}
PARSER_BROW_LABELS = {"lb", "rb"}
PARSER_EYE_LABELS = {"le", "re"}
PARSER_NOSE_LABELS = {"nose"}
PARSER_LIP_LABELS = {"ulip", "llip", "imouth"}
PARSER_HAIR_LABELS = {"hair"}
PARSER_FACE_REGION_LABELS = (
    PARSER_SKIN_LABELS
    | PARSER_BROW_LABELS
    | PARSER_EYE_LABELS
    | PARSER_NOSE_LABELS
    | PARSER_LIP_LABELS
)

DEBUG_REGION_COLORS = {
    "face": (76, 201, 240),
    "surface": (67, 170, 139),
    "core": (244, 63, 94),
    "contour": (251, 191, 36),
    "hairline": (168, 85, 247),
}
PARSER_LABEL_COLORS = {
    "background": (0, 0, 0),
    "face": (255, 250, 79),
    "lb": (255, 125, 138),
    "rb": (213, 32, 29),
    "le": (0, 144, 187),
    "re": (0, 196, 253),
    "nose": (255, 129, 54),
    "ulip": (88, 233, 135),
    "imouth": (255, 76, 249),
    "llip": (0, 117, 27),
    "hair": (255, 0, 0),
}


def _connection_ids(connections: Iterable[tuple[int, int]]) -> list[int]:
    ids: set[int] = set()
    for left, right in connections:
        ids.add(int(left))
        ids.add(int(right))
    return sorted(ids)


def _convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set((float(x), float(y)) for x, y in points))
    if len(unique) <= 2:
        return list(unique)

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _mean_point(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    array = np.asarray(points, dtype=np.float32)
    return float(array[:, 0].mean()), float(array[:, 1].mean())


def _fit_affine(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    rows = []
    values = []
    for (sx, sy), (tx, ty) in zip(source_points, target_points):
        rows.append([sx, sy, 1.0, 0.0, 0.0, 0.0])
        rows.append([0.0, 0.0, 0.0, sx, sy, 1.0])
        values.extend([tx, ty])

    matrix, _, _, _ = np.linalg.lstsq(np.asarray(rows, dtype=np.float32), np.asarray(values, dtype=np.float32), rcond=None)
    affine = np.array(
        [
            [matrix[0], matrix[1], matrix[2]],
            [matrix[3], matrix[4], matrix[5]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return affine


def _warp_image(image: Image.Image, affine: np.ndarray, output_size: tuple[int, int]) -> Image.Image:
    inverse = np.linalg.inv(affine)
    coeffs = (
        float(inverse[0, 0]),
        float(inverse[0, 1]),
        float(inverse[0, 2]),
        float(inverse[1, 0]),
        float(inverse[1, 1]),
        float(inverse[1, 2]),
    )
    return image.convert("RGB").transform(output_size, Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BICUBIC)


def _expand_mask(mask: Image.Image, expand_px: int, blur_radius: float) -> Image.Image:
    expanded = mask
    if expand_px > 0:
        filter_size = max(3, (expand_px * 2) + 1)
        expanded = expanded.filter(ImageFilter.MaxFilter(size=filter_size))
    if blur_radius > 0:
        expanded = expanded.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return expanded


def _contract_mask(mask: Image.Image, contract_px: int, blur_radius: float) -> Image.Image:
    contracted = mask
    if contract_px > 0:
        filter_size = max(3, (contract_px * 2) + 1)
        contracted = contracted.filter(ImageFilter.MinFilter(size=filter_size))
    if blur_radius > 0:
        contracted = contracted.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return contracted


def _mask_to_array(mask: Image.Image) -> np.ndarray:
    return (np.asarray(mask, dtype=np.float32) / 255.0)[..., None]


def _union_masks(*masks: Image.Image) -> Image.Image:
    valid_masks = [np.asarray(mask, dtype=np.uint8) for mask in masks if mask is not None]
    if not valid_masks:
        raise ValueError("At least one mask is required for union.")
    union = valid_masks[0]
    for mask_array in valid_masks[1:]:
        union = np.maximum(union, mask_array)
    return Image.fromarray(union, mode="L")


def _subtract_masks(base: Image.Image, *subtract_masks: Image.Image, blur_radius: float = 0.0) -> Image.Image:
    result = np.asarray(base, dtype=np.float32)
    for mask in subtract_masks:
        if mask is not None:
            result -= np.asarray(mask, dtype=np.float32)
    result = np.clip(result, 0.0, 255.0)
    output = Image.fromarray(result.astype(np.uint8), mode="L")
    if blur_radius > 0:
        output = output.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return output


def _intersect_masks(*masks: Image.Image) -> Image.Image:
    valid_masks = [np.asarray(mask, dtype=np.uint8) for mask in masks if mask is not None]
    if not valid_masks:
        raise ValueError("At least one mask is required for intersection.")
    intersection = valid_masks[0]
    for mask_array in valid_masks[1:]:
        intersection = np.minimum(intersection, mask_array)
    return Image.fromarray(intersection, mode="L")


def _mask_from_labels(
    labels: np.ndarray,
    label_names: Sequence[str],
    names: set[str],
    expand_px: int = 0,
    blur_radius: float = 0.0,
) -> Image.Image:
    label_map = {str(name): index for index, name in enumerate(label_names)}
    active_ids = [label_map[name] for name in names if name in label_map]
    if not active_ids:
        return Image.new("L", (labels.shape[1], labels.shape[0]), 0)

    mask = np.isin(labels, active_ids).astype(np.uint8) * 255
    return _expand_mask(Image.fromarray(mask, mode="L"), expand_px=expand_px, blur_radius=blur_radius)


def _colorize_labels(labels: np.ndarray, label_names: Sequence[str]) -> Image.Image:
    height, width = labels.shape
    colored = np.zeros((height, width, 3), dtype=np.uint8)
    for index, label_name in enumerate(label_names):
        colored[labels == index] = PARSER_LABEL_COLORS.get(label_name, (255, 255, 255))
    return Image.fromarray(colored, mode="RGB")


def _overlay_regions(base_image: Image.Image, regions: Dict[str, Image.Image]) -> Image.Image:
    overlay = np.asarray(base_image.convert("RGB"), dtype=np.float32)
    for name, mask in regions.items():
        color = np.asarray(DEBUG_REGION_COLORS.get(name, (255, 255, 255)), dtype=np.float32)
        alpha = _mask_to_array(mask) * 0.35
        overlay = overlay * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


@dataclass
class FaceMaskResult:
    image: Image.Image
    applied: bool
    mode: str
    reason: str
    engine: str = "legacy"
    metadata: Dict[str, Any] = field(default_factory=dict)
    debug_images: Dict[str, Image.Image] = field(default_factory=dict)


@dataclass
class ParserFaceData:
    labels: np.ndarray
    label_names: list[str]
    rect: tuple[float, float, float, float]
    score: float | None = None


class _FacerParserRuntime:
    def __init__(self, parser_model: str = "farl/lapa/448", device: str | None = None) -> None:
        if torch is None or facer is None:
            raise ImportError("pyfacer and torch are required for parser-driven face masking.")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = facer.face_detector("retinaface/mobilenet", device=self.device)
        self.parser = facer.face_parser(parser_model, device=self.device)
        self.parser_model = parser_model

    def parse(self, image: Image.Image) -> ParserFaceData | None:
        image_tensor = facer.hwc2bchw(np.asarray(image.convert("RGB"), dtype=np.uint8)).to(device=self.device)

        with torch.inference_mode():
            faces = self.detector(image_tensor)
            if "rects" not in faces or faces["rects"].shape[0] == 0:
                return None
            faces = self.parser(image_tensor, faces)

        rects = faces["rects"].detach().cpu().numpy()
        areas = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
        face_index = int(np.argmax(areas))
        seg_logits = faces["seg"]["logits"][face_index]
        labels = seg_logits.argmax(dim=0).detach().cpu().numpy().astype(np.uint8)
        label_names = list(faces["seg"].get("label_names") or self.parser.label_names)
        score = float(faces["scores"][face_index].item()) if "scores" in faces else None

        return ParserFaceData(
            labels=labels,
            label_names=label_names,
            rect=tuple(float(value) for value in rects[face_index]),
            score=score,
        )


class FaceIdentityMasker:
    def __init__(self, parser_model: str = "farl/lapa/448") -> None:
        if mp is None:
            raise ImportError("mediapipe is required for face masking.")

        face_mesh = mp.solutions.face_mesh
        self._mesh = face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

        self._face_oval_ids = _connection_ids(face_mesh.FACEMESH_FACE_OVAL)
        self._left_eye_ids = _connection_ids(face_mesh.FACEMESH_LEFT_EYE)
        self._right_eye_ids = _connection_ids(face_mesh.FACEMESH_RIGHT_EYE)
        self._left_brow_ids = _connection_ids(face_mesh.FACEMESH_LEFT_EYEBROW)
        self._right_brow_ids = _connection_ids(face_mesh.FACEMESH_RIGHT_EYEBROW)
        self._lip_ids = _connection_ids(face_mesh.FACEMESH_LIPS)

        self._parser_model = parser_model
        self._parser_runtime: _FacerParserRuntime | None = None
        self._parser_failed = False

    def _extract_landmarks(self, image: Image.Image) -> list[tuple[float, float]] | None:
        rgb = np.asarray(image.convert("RGB"))
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        width, height = image.size
        points: list[tuple[float, float]] = []
        for landmark in result.multi_face_landmarks[0].landmark:
            points.append((landmark.x * width, landmark.y * height))
        return points

    def estimate_face_coverage(self, image: Image.Image) -> float | None:
        landmarks = self._extract_landmarks(image)
        if not landmarks:
            return None

        face_points = np.asarray([landmarks[index] for index in self._face_oval_ids if index < len(landmarks)], dtype=np.float32)
        if face_points.size == 0:
            return None

        x1, y1 = face_points.min(axis=0)
        x2, y2 = face_points.max(axis=0)
        bbox_area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
        image_area = max(1.0, float(image.width * image.height))
        return bbox_area / image_area

    def _anchors(self, landmarks: list[tuple[float, float]]) -> np.ndarray:
        left_eye = _mean_point([landmarks[index] for index in LEFT_EYE_CENTER_INDICES])
        right_eye = _mean_point([landmarks[index] for index in RIGHT_EYE_CENTER_INDICES])
        nose_tip = landmarks[NOSE_TIP_INDEX]
        mouth_left = landmarks[MOUTH_CORNER_LEFT_INDEX]
        mouth_right = landmarks[MOUTH_CORNER_RIGHT_INDEX]
        return np.asarray([left_eye, right_eye, nose_tip, mouth_left, mouth_right], dtype=np.float32)

    def _mask_from_indices(
        self,
        size: tuple[int, int],
        landmarks: list[tuple[float, float]],
        index_groups: Sequence[Sequence[int]],
        blur_radius: float,
    ) -> Image.Image:
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)

        for indices in index_groups:
            polygon = _convex_hull([landmarks[index] for index in indices if index < len(landmarks)])
            if len(polygon) >= 3:
                draw.polygon(polygon, fill=255)

        if blur_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        return mask

    def _build_legacy_masks(
        self,
        size: tuple[int, int],
        landmarks: list[tuple[float, float]],
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        base = max(size)
        face_blur = max(8.0, base / 80.0)
        core_blur = max(5.0, base / 120.0)
        face_expand = max(10, int(base / 70))
        core_expand = max(6, int(base / 120))

        face_mask = self._mask_from_indices(size, landmarks, [self._face_oval_ids], blur_radius=face_blur)
        core_groups = [
            [*self._left_eye_ids, *self._left_brow_ids],
            [*self._right_eye_ids, *self._right_brow_ids],
            self._lip_ids,
            NOSE_INDICES,
        ]
        core_mask = self._mask_from_indices(size, landmarks, core_groups, blur_radius=core_blur)
        face_mask = _expand_mask(face_mask, expand_px=face_expand, blur_radius=face_blur)
        core_mask = _expand_mask(core_mask, expand_px=core_expand, blur_radius=core_blur)

        contour_outer = _expand_mask(face_mask, expand_px=max(4, face_expand // 2), blur_radius=max(2.0, face_blur * 0.35))
        contour_inner = _contract_mask(face_mask, contract_px=max(4, face_expand // 2), blur_radius=max(2.0, face_blur * 0.15))
        contour_mask = _subtract_masks(contour_outer, contour_inner, blur_radius=max(2.0, face_blur * 0.25))
        return face_mask, core_mask, contour_mask

    def _build_legacy_debug_images(
        self,
        generated_image: Image.Image,
        aligned_source: Image.Image,
        face_mask: Image.Image,
        core_mask: Image.Image,
        contour_mask: Image.Image,
    ) -> Dict[str, Image.Image]:
        return {
            "source_aligned": aligned_source,
            "mask_face": face_mask.convert("L"),
            "mask_core": core_mask.convert("L"),
            "mask_contour": contour_mask.convert("L"),
            "overlay_regions": _overlay_regions(
                generated_image,
                {
                    "face": face_mask,
                    "core": core_mask,
                    "contour": contour_mask,
                },
            ),
        }

    def _legacy_protect(
        self,
        source_image: Image.Image,
        generated_image: Image.Image,
        mode: str = "balanced",
        debug: bool = False,
    ) -> FaceMaskResult:
        normalized_mode = (mode or "balanced").strip().lower()
        if normalized_mode == "surface_fx":
            normalized_mode = "balanced"

        source_landmarks = self._extract_landmarks(source_image)
        if not source_landmarks:
            return FaceMaskResult(image=generated_image, applied=False, mode=normalized_mode, reason="no-source-face", engine="legacy")

        generated_rgb = generated_image.convert("RGB")
        generated_landmarks = self._extract_landmarks(generated_rgb)
        if not generated_landmarks:
            return FaceMaskResult(image=generated_rgb, applied=False, mode=normalized_mode, reason="no-output-face", engine="legacy")

        affine = _fit_affine(self._anchors(source_landmarks), self._anchors(generated_landmarks))
        aligned_source = _warp_image(source_image, affine, generated_rgb.size)
        face_mask, core_mask, contour_mask = self._build_legacy_masks(generated_rgb.size, generated_landmarks)

        source_array = np.asarray(aligned_source, dtype=np.float32)
        generated_array = np.asarray(generated_rgb, dtype=np.float32)
        face_alpha = _mask_to_array(face_mask)
        core_alpha = _mask_to_array(core_mask)

        if normalized_mode == "strict":
            strict_low = source_array * 0.98 + generated_array * 0.02
            strict = generated_array * (1.0 - face_alpha) + strict_low * face_alpha
            strict = strict * (1.0 - core_alpha) + source_array * core_alpha
            return FaceMaskResult(
                image=Image.fromarray(np.clip(strict, 0, 255).astype(np.uint8), mode="RGB"),
                applied=True,
                mode="strict",
                reason="legacy-strict-protection",
                engine="legacy",
                metadata={"strategy_used": "legacy"},
                debug_images=self._build_legacy_debug_images(generated_rgb, aligned_source, face_mask, core_mask, contour_mask) if debug else {},
            )

        detail_radius = max(6.0, max(generated_rgb.size) / 100.0)
        source_low = np.asarray(aligned_source.filter(ImageFilter.GaussianBlur(radius=detail_radius)), dtype=np.float32)
        generated_low = np.asarray(generated_rgb.filter(ImageFilter.GaussianBlur(radius=detail_radius)), dtype=np.float32)
        generated_detail = generated_array - generated_low
        source_detail = source_array - source_low

        face_mix_low = source_low * 0.86 + generated_low * 0.14
        core_mix_low = source_low * 0.98 + generated_low * 0.02

        low_mix = generated_low * (1.0 - face_alpha) + face_mix_low * face_alpha
        low_mix = low_mix * (1.0 - core_alpha) + core_mix_low * core_alpha
        detail_scale = np.ones_like(face_alpha, dtype=np.float32)
        detail_scale = detail_scale * (1.0 - face_alpha) + (0.42 * face_alpha)
        detail_scale = detail_scale * (1.0 - core_alpha) + (0.08 * core_alpha)
        source_detail_boost = (core_alpha * 0.72) + (face_alpha * 0.18)
        inside_face = np.clip(low_mix + (generated_detail * detail_scale) + (source_detail * source_detail_boost), 0, 255)
        final = generated_array * (1.0 - face_alpha) + inside_face * face_alpha

        return FaceMaskResult(
            image=Image.fromarray(np.clip(final, 0, 255).astype(np.uint8), mode="RGB"),
            applied=True,
            mode="balanced",
            reason="legacy-balanced-protection",
            engine="legacy",
            metadata={"strategy_used": "legacy"},
            debug_images=self._build_legacy_debug_images(generated_rgb, aligned_source, face_mask, core_mask, contour_mask) if debug else {},
        )

    def _get_parser_runtime(self) -> _FacerParserRuntime:
        if self._parser_runtime is not None:
            return self._parser_runtime
        if self._parser_failed:
            raise RuntimeError("parser-unavailable")

        try:
            self._parser_runtime = _FacerParserRuntime(parser_model=self._parser_model)
            return self._parser_runtime
        except Exception as exc:  # pragma: no cover - exercised in the worker image
            self._parser_failed = True
            raise RuntimeError(f"parser-load-failed: {exc}") from exc

    def _parse_face_regions(self, image: Image.Image) -> ParserFaceData:
        runtime = self._get_parser_runtime()
        parsed = runtime.parse(image)
        if parsed is None:
            raise RuntimeError("no-parser-face")
        return parsed

    def _build_smart_masks(
        self,
        size: tuple[int, int],
        landmarks: list[tuple[float, float]],
        parser_face: ParserFaceData,
    ) -> Dict[str, Image.Image]:
        base = max(size)
        face_mask_land, core_mask_land, contour_mask_land = self._build_legacy_masks(size, landmarks)

        parser_face_mask = _mask_from_labels(
            parser_face.labels,
            parser_face.label_names,
            PARSER_FACE_REGION_LABELS,
            expand_px=max(2, int(base / 220)),
            blur_radius=max(2.0, base / 220.0),
        )
        parser_skin_mask = _mask_from_labels(
            parser_face.labels,
            parser_face.label_names,
            PARSER_SKIN_LABELS,
            expand_px=max(2, int(base / 250)),
            blur_radius=max(2.0, base / 260.0),
        )
        parser_core_mask = _union_masks(
            _mask_from_labels(parser_face.labels, parser_face.label_names, PARSER_BROW_LABELS, blur_radius=max(1.5, base / 260.0)),
            _mask_from_labels(parser_face.labels, parser_face.label_names, PARSER_EYE_LABELS, blur_radius=max(1.5, base / 260.0)),
            _mask_from_labels(parser_face.labels, parser_face.label_names, PARSER_NOSE_LABELS, blur_radius=max(1.5, base / 260.0)),
            _mask_from_labels(parser_face.labels, parser_face.label_names, PARSER_LIP_LABELS, blur_radius=max(1.5, base / 260.0)),
        )
        parser_hair_mask = _mask_from_labels(
            parser_face.labels,
            parser_face.label_names,
            PARSER_HAIR_LABELS,
            expand_px=max(2, int(base / 250)),
            blur_radius=max(2.0, base / 260.0),
        )

        face_mask = _union_masks(face_mask_land, parser_face_mask, parser_core_mask)
        core_mask = _union_masks(core_mask_land, parser_core_mask)
        contour_mask = _union_masks(
            contour_mask_land,
            _subtract_masks(
                _expand_mask(face_mask_land, expand_px=max(4, int(base / 180)), blur_radius=max(2.0, base / 200.0)),
                _contract_mask(face_mask_land, contract_px=max(4, int(base / 220)), blur_radius=max(1.5, base / 280.0)),
                blur_radius=max(2.0, base / 260.0),
            ),
        )
        hairline_mask = _intersect_masks(
            _expand_mask(parser_hair_mask, expand_px=max(2, int(base / 260)), blur_radius=max(1.5, base / 280.0)),
            _expand_mask(face_mask_land, expand_px=max(4, int(base / 180)), blur_radius=max(2.0, base / 220.0)),
        )

        surface_mask = _subtract_masks(
            _union_masks(parser_skin_mask, parser_face_mask),
            core_mask,
            contour_mask,
            hairline_mask,
            blur_radius=max(2.0, base / 260.0),
        )
        surface_mask = _intersect_masks(surface_mask, face_mask)
        remainder_mask = _subtract_masks(
            face_mask,
            core_mask,
            contour_mask,
            hairline_mask,
            surface_mask,
            blur_radius=max(1.0, base / 320.0),
        )

        return {
            "face": face_mask,
            "core": core_mask,
            "contour": contour_mask,
            "hairline": hairline_mask,
            "surface": surface_mask,
            "remainder": remainder_mask,
            "parser_labels": _colorize_labels(parser_face.labels, parser_face.label_names),
        }

    def _profile(self, mode: str, strength: float) -> Dict[str, Dict[str, float]]:
        normalized_mode = (mode or "balanced").strip().lower()
        clamped_strength = float(np.clip(strength, 0.0, 1.0))

        profile = {
            "surface": {
                "source_low": 0.88 + (0.08 * clamped_strength),
                "generated_detail": 0.72 - (0.20 * clamped_strength),
                "source_detail": 0.12 + (0.12 * clamped_strength),
            },
            "remainder": {
                "source_low": 0.84 + (0.08 * clamped_strength),
                "generated_detail": 0.58 - (0.18 * clamped_strength),
                "source_detail": 0.10 + (0.10 * clamped_strength),
            },
            "hairline": {
                "source_low": 0.88 + (0.08 * clamped_strength),
                "generated_detail": 0.30 - (0.18 * clamped_strength),
                "source_detail": 0.14 + (0.14 * clamped_strength),
            },
            "contour": {
                "source_low": 0.92 + (0.07 * clamped_strength),
                "generated_detail": 0.18 - (0.12 * clamped_strength),
                "source_detail": 0.28 + (0.18 * clamped_strength),
            },
            "core": {
                "source_low": 0.95 + (0.045 * clamped_strength),
                "generated_detail": 0.12 - (0.09 * clamped_strength),
                "source_detail": 0.55 + (0.25 * clamped_strength),
            },
        }

        if normalized_mode == "strict":
            profile["surface"] = {
                "source_low": 0.94 + (0.04 * clamped_strength),
                "generated_detail": 0.35 - (0.18 * clamped_strength),
                "source_detail": 0.25 + (0.20 * clamped_strength),
            }
            profile["remainder"] = {
                "source_low": 0.90 + (0.06 * clamped_strength),
                "generated_detail": 0.26 - (0.14 * clamped_strength),
                "source_detail": 0.18 + (0.18 * clamped_strength),
            }
        elif normalized_mode == "surface_fx":
            profile["surface"] = {
                "source_low": 0.84 + (0.08 * clamped_strength),
                "generated_detail": 0.92 - (0.12 * clamped_strength),
                "source_detail": 0.10 + (0.10 * clamped_strength),
            }
            profile["remainder"] = {
                "source_low": 0.82 + (0.08 * clamped_strength),
                "generated_detail": 0.76 - (0.16 * clamped_strength),
                "source_detail": 0.10 + (0.10 * clamped_strength),
            }

        return profile

    def _build_smart_debug_images(
        self,
        generated_image: Image.Image,
        aligned_source: Image.Image,
        region_masks: Dict[str, Image.Image],
    ) -> Dict[str, Image.Image]:
        overlay_regions = {
            "face": region_masks["face"],
            "surface": region_masks["surface"],
            "core": region_masks["core"],
            "contour": region_masks["contour"],
            "hairline": region_masks["hairline"],
        }
        return {
            "source_aligned": aligned_source,
            "mask_face": region_masks["face"].convert("L"),
            "mask_surface": region_masks["surface"].convert("L"),
            "mask_core": region_masks["core"].convert("L"),
            "mask_contour": region_masks["contour"].convert("L"),
            "mask_hairline": region_masks["hairline"].convert("L"),
            "parser_labels": region_masks["parser_labels"],
            "overlay_regions": _overlay_regions(generated_image, overlay_regions),
        }

    def _smart_protect(
        self,
        source_image: Image.Image,
        generated_image: Image.Image,
        mode: str,
        strength: float,
        debug: bool,
    ) -> FaceMaskResult:
        normalized_mode = (mode or "balanced").strip().lower()
        generated_rgb = generated_image.convert("RGB")

        source_landmarks = self._extract_landmarks(source_image)
        if not source_landmarks:
            return FaceMaskResult(image=generated_rgb, applied=False, mode=normalized_mode, reason="no-source-face", engine="smart")

        generated_landmarks = self._extract_landmarks(generated_rgb)
        if not generated_landmarks:
            return FaceMaskResult(image=generated_rgb, applied=False, mode=normalized_mode, reason="no-output-face", engine="smart")

        affine = _fit_affine(self._anchors(source_landmarks), self._anchors(generated_landmarks))
        aligned_source = _warp_image(source_image, affine, generated_rgb.size)
        parser_face = self._parse_face_regions(generated_rgb)
        region_masks = self._build_smart_masks(generated_rgb.size, generated_landmarks, parser_face)

        source_array = np.asarray(aligned_source, dtype=np.float32)
        generated_array = np.asarray(generated_rgb, dtype=np.float32)
        detail_radius = max(5.0, max(generated_rgb.size) / 120.0)
        source_low = np.asarray(aligned_source.filter(ImageFilter.GaussianBlur(radius=detail_radius)), dtype=np.float32)
        generated_low = np.asarray(generated_rgb.filter(ImageFilter.GaussianBlur(radius=detail_radius)), dtype=np.float32)
        generated_detail = generated_array - generated_low
        source_detail = source_array - source_low

        core_mask = region_masks["core"]
        contour_mask = _subtract_masks(region_masks["contour"], core_mask)
        hairline_mask = _subtract_masks(region_masks["hairline"], core_mask, region_masks["contour"])
        surface_mask = _subtract_masks(region_masks["surface"], core_mask, region_masks["contour"], region_masks["hairline"])
        remainder_mask = _subtract_masks(
            region_masks["remainder"],
            core_mask,
            region_masks["contour"],
            region_masks["hairline"],
            region_masks["surface"],
        )

        profile = self._profile(normalized_mode, strength=strength)
        low_mix = generated_low.copy()
        detail_mix = generated_detail.copy()

        def apply_region(mask: Image.Image, settings: Dict[str, float]) -> None:
            nonlocal low_mix, detail_mix
            alpha = _mask_to_array(mask)
            if float(alpha.max()) <= 0.0:
                return
            region_low = (source_low * settings["source_low"]) + (generated_low * (1.0 - settings["source_low"]))
            region_detail = (generated_detail * settings["generated_detail"]) + (source_detail * settings["source_detail"])
            low_mix = (low_mix * (1.0 - alpha)) + (region_low * alpha)
            detail_mix = (detail_mix * (1.0 - alpha)) + (region_detail * alpha)

        apply_region(surface_mask, profile["surface"])
        apply_region(remainder_mask, profile["remainder"])
        apply_region(hairline_mask, profile["hairline"])
        apply_region(contour_mask, profile["contour"])
        apply_region(core_mask, profile["core"])

        final = np.clip(low_mix + detail_mix, 0, 255)
        result_image = Image.fromarray(final.astype(np.uint8), mode="RGB")
        metadata: Dict[str, Any] = {
            "strategy_used": "smart",
            "parser_model": self._parser_model,
            "strength": round(float(np.clip(strength, 0.0, 1.0)), 3),
            "parser_score": parser_face.score,
            "parser_rect": [round(value, 2) for value in parser_face.rect],
        }

        return FaceMaskResult(
            image=result_image,
            applied=True,
            mode=normalized_mode,
            reason=f"smart-{normalized_mode}-protection",
            engine="smart",
            metadata=metadata,
            debug_images=self._build_smart_debug_images(generated_rgb, aligned_source, region_masks) if debug else {},
        )

    def protect(
        self,
        source_image: Image.Image,
        generated_image: Image.Image,
        mode: str = "balanced",
        strategy: str = "auto",
        strength: float = 0.86,
        debug: bool = False,
    ) -> FaceMaskResult:
        normalized_mode = (mode or "balanced").strip().lower()
        normalized_strategy = (strategy or "auto").strip().lower()

        if normalized_mode == "off":
            return FaceMaskResult(image=generated_image, applied=False, mode="off", reason="disabled", engine="none")

        if normalized_strategy == "legacy":
            result = self._legacy_protect(source_image, generated_image, mode=normalized_mode, debug=debug)
            result.metadata.setdefault("strategy_requested", "legacy")
            return result

        try:
            result = self._smart_protect(
                source_image=source_image,
                generated_image=generated_image,
                mode=normalized_mode,
                strength=strength,
                debug=debug,
            )
            result.metadata.setdefault("strategy_requested", normalized_strategy)
            return result
        except Exception as exc:
            if normalized_strategy == "smart":
                return FaceMaskResult(
                    image=generated_image.convert("RGB"),
                    applied=False,
                    mode=normalized_mode,
                    reason=f"smart-failed:{exc}",
                    engine="smart",
                    metadata={"strategy_requested": "smart"},
                )

            fallback = self._legacy_protect(source_image, generated_image, mode=normalized_mode, debug=debug)
            fallback.reason = f"{fallback.reason}; smart-fallback:{exc}"
            fallback.metadata["strategy_requested"] = normalized_strategy
            fallback.metadata["fallback"] = "legacy"
            return fallback
