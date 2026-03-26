from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None
    np = None

logger = logging.getLogger(__name__)

_NORMALIZED_WIDTH = 1000
_NORMALIZED_HEIGHT = 630
_MIN_DOCUMENT_AREA_RATIO = 0.2
_EXPECTED_ASPECT_RATIO = 1.586
_MAX_ASPECT_RATIO_DELTA = 0.4

# Relative crop calibrated for the lower-right identifier region on INE reverso.
_INE_REVERSO_ID_CROP = (0.52, 0.60, 0.38, 0.22)


@dataclass(slots=True)
class PreprocessedDocument:
    image_bytes: bytes
    used_specialized_crop: bool = False
    debug_image_path: str | None = None
    quality_flags: list[str] | None = None

    def __post_init__(self) -> None:
        if self.quality_flags is None:
            self.quality_flags = []


class DocumentPreprocessor:
    def __init__(self, debug_dir: str = "") -> None:
        self.debug_dir = debug_dir.strip()

    def preprocess_identity_document(
        self,
        image_bytes: bytes,
        document_type: str,
    ) -> PreprocessedDocument:
        if document_type != "INE_REVERSO":
            return PreprocessedDocument(image_bytes=image_bytes)

        if cv2 is None or np is None:
            logger.warning("OpenCV is not installed; skipping INE reverso preprocessing.")
            return PreprocessedDocument(
                image_bytes=image_bytes,
                quality_flags=["preprocessing_unavailable"],
            )

        try:
            crop = self._extract_ine_reverso_identifier_crop(image_bytes)
        except Exception as exc:  # pragma: no cover
            logger.warning("INE reverso preprocessing failed: %s", exc)
            return PreprocessedDocument(
                image_bytes=image_bytes,
                quality_flags=["preprocessing_failed"],
            )

        if crop is None:
            return PreprocessedDocument(
                image_bytes=image_bytes,
                quality_flags=["document_alignment_failed"],
            )

        debug_image_path = self._maybe_write_debug_crop(crop)
        if debug_image_path:
            logger.info("Saved INE reverso crop to %s", debug_image_path)

        return PreprocessedDocument(
            image_bytes=crop,
            used_specialized_crop=True,
            debug_image_path=debug_image_path,
        )

    def _extract_ine_reverso_identifier_crop(self, image_bytes: bytes) -> bytes | None:
        image = _decode_image(image_bytes)
        corners = _find_document_corners(image)
        if corners is None:
            return None

        aligned = _warp_document(image, corners)
        x, y, w, h = _relative_crop_box(aligned.shape[1], aligned.shape[0], _INE_REVERSO_ID_CROP)
        crop = aligned[y : y + h, x : x + w]
        success, encoded = cv2.imencode(".jpg", crop)
        if not success:
            raise ValueError("Could not encode cropped document image.")
        return encoded.tobytes()

    def _maybe_write_debug_crop(self, image_bytes: bytes) -> str | None:
        if not self.debug_dir:
            return None

        directory = Path(self.debug_dir)
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / f"ine-reverso-crop-{uuid4().hex}.jpg"
        output_path.write_bytes(image_bytes)
        return str(output_path)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    if cv2 is None or np is None:  # pragma: no cover
        raise ValueError("OpenCV is not available.")
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image bytes.")
    return image


def _find_document_corners(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 180)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]

    best_quad: np.ndarray | None = None
    best_area = 0.0
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue

        area = cv2.contourArea(approx)
        if area < image_area * _MIN_DOCUMENT_AREA_RATIO:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if h == 0:
            continue
        aspect_ratio = w / h
        if abs(aspect_ratio - _EXPECTED_ASPECT_RATIO) > _MAX_ASPECT_RATIO_DELTA:
            continue

        if area > best_area:
            best_area = area
            best_quad = approx.reshape(4, 2)

    if best_quad is None:
        return None
    return _order_points(best_quad.astype("float32"))


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def _warp_document(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    destination = np.array(
        [
            [0, 0],
            [_NORMALIZED_WIDTH - 1, 0],
            [_NORMALIZED_WIDTH - 1, _NORMALIZED_HEIGHT - 1],
            [0, _NORMALIZED_HEIGHT - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, matrix, (_NORMALIZED_WIDTH, _NORMALIZED_HEIGHT))


def _relative_crop_box(
    width: int,
    height: int,
    relative_box: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    rel_x, rel_y, rel_w, rel_h = relative_box
    x = max(int(width * rel_x), 0)
    y = max(int(height * rel_y), 0)
    w = min(int(width * rel_w), width - x)
    h = min(int(height * rel_h), height - y)
    return x, y, w, h


from app.core.config import settings

document_preprocessor = DocumentPreprocessor(debug_dir=settings.PREPROCESS_DEBUG_DIR)
