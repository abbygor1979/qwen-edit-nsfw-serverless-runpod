from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


try:
    import mediapipe as mp
except Exception:  # pragma: no cover - dependency is only required in the Runpod worker image
    mp = None


NOSE_INDICES = [1, 2, 4, 5, 6, 19, 45, 48, 49, 64, 94, 97, 98, 115, 168, 195, 197, 220, 275, 279, 289, 344, 440]
LEFT_EYE_CENTER_INDICES = [33, 133, 159, 145]
RIGHT_EYE_CENTER_INDICES = [362, 263, 386, 374]
MOUTH_CORNER_LEFT_INDEX = 61
MOUTH_CORNER_RIGHT_INDEX = 291
NOSE_TIP_INDEX = 1


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


@dataclass
class FaceMaskResult:
    image: Image.Image
    applied: bool
    mode: str
    reason: str


class FaceIdentityMasker:
    def __init__(self) -> None:
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

    def _build_masks(self, size: tuple[int, int], landmarks: list[tuple[float, float]]) -> tuple[Image.Image, Image.Image]:
        base = max(size)
        face_blur = max(8.0, base / 80.0)
        core_blur = max(5.0, base / 120.0)

        face_mask = self._mask_from_indices(size, landmarks, [self._face_oval_ids], blur_radius=face_blur)
        core_groups = [
            [*self._left_eye_ids, *self._left_brow_ids],
            [*self._right_eye_ids, *self._right_brow_ids],
            self._lip_ids,
            NOSE_INDICES,
        ]
        core_mask = self._mask_from_indices(size, landmarks, core_groups, blur_radius=core_blur)
        return face_mask, core_mask

    def protect(self, source_image: Image.Image, generated_image: Image.Image, mode: str = "balanced") -> FaceMaskResult:
        normalized_mode = (mode or "balanced").strip().lower()
        if normalized_mode == "off":
            return FaceMaskResult(image=generated_image, applied=False, mode="off", reason="disabled")

        source_landmarks = self._extract_landmarks(source_image)
        if not source_landmarks:
            return FaceMaskResult(image=generated_image, applied=False, mode=normalized_mode, reason="no-source-face")

        generated_rgb = generated_image.convert("RGB")
        generated_landmarks = self._extract_landmarks(generated_rgb)
        if not generated_landmarks:
            return FaceMaskResult(image=generated_rgb, applied=False, mode=normalized_mode, reason="no-output-face")

        affine = _fit_affine(self._anchors(source_landmarks), self._anchors(generated_landmarks))
        aligned_source = _warp_image(source_image, affine, generated_rgb.size)
        face_mask, core_mask = self._build_masks(generated_rgb.size, generated_landmarks)

        source_array = np.asarray(aligned_source, dtype=np.float32)
        generated_array = np.asarray(generated_rgb, dtype=np.float32)
        face_alpha = (np.asarray(face_mask, dtype=np.float32) / 255.0)[..., None]
        core_alpha = (np.asarray(core_mask, dtype=np.float32) / 255.0)[..., None]

        if normalized_mode == "strict":
            final = generated_array * (1.0 - face_alpha) + source_array * face_alpha
            return FaceMaskResult(
                image=Image.fromarray(np.clip(final, 0, 255).astype(np.uint8), mode="RGB"),
                applied=True,
                mode="strict",
                reason="strict-protection",
            )

        detail_radius = max(6.0, max(generated_rgb.size) / 100.0)
        source_low = np.asarray(aligned_source.filter(ImageFilter.GaussianBlur(radius=detail_radius)), dtype=np.float32)
        generated_low = np.asarray(generated_rgb.filter(ImageFilter.GaussianBlur(radius=detail_radius)), dtype=np.float32)
        generated_detail = generated_array - generated_low

        face_mix_low = source_low * 0.72 + generated_low * 0.28
        core_mix_low = source_low * 0.90 + generated_low * 0.10

        low_mix = generated_low * (1.0 - face_alpha) + face_mix_low * face_alpha
        low_mix = low_mix * (1.0 - core_alpha) + core_mix_low * core_alpha
        inside_face = np.clip(low_mix + (generated_detail * 0.92), 0, 255)
        final = generated_array * (1.0 - face_alpha) + inside_face * face_alpha

        return FaceMaskResult(
            image=Image.fromarray(np.clip(final, 0, 255).astype(np.uint8), mode="RGB"),
            applied=True,
            mode="balanced",
            reason="balanced-protection",
        )
