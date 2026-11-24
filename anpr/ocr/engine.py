"""PaddleOCR plate-reading wrapper for the ANPR Autogate System.

:class:`PaddleOcrEngine` wraps the pre-trained PaddleOCR engine and
structurally satisfies the ``OcrEngine`` Protocol defined in
``anpr.core.interfaces``. It is the single I/O adapter that turns a cropped
plate region into the pure :class:`~anpr.core.models.OcrResult` value object the
rest of the pipeline consumes, so no downstream component depends on PaddleOCR
directly (Requirement 14.4).

Reading (Requirement 3.2):

* :meth:`read_plate` runs PaddleOCR on the cropped region and extracts a
  candidate license-plate string of **1 to 9 alphanumeric characters**
  consistent with the Indonesian_Plate_Format (a 2-letter area code, 4-digit
  number, and 3-letter suffix sum to at most 9 characters). Non-alphanumeric
  characters are dropped and the survivors are uppercased so the output feeds
  cleanly into the ``Plate_Normalizer`` (Req 3.2).
* The recognition confidence is always clamped/validated into the inclusive
  range ``0.0`` to ``1.0`` (Req 3.2).
* When there is no readable text (PaddleOCR returns nothing, or the recognized
  text contains no alphanumeric characters), :attr:`OcrResult.text` is set to
  ``None`` and the confidence is ``0.0``.

Timeout (Requirement 3.5):

* The Detection_Pipeline — not this wrapper — enforces the configurable OCR
  processing timeout. This wrapper only ever exposes the ``timed_out`` field on
  the result (always ``False`` here); the pipeline constructs a timed-out
  :class:`OcrResult` when it abandons a slow read.

The PaddleOCR factory is injectable so the parsing logic can be exercised
deterministically in tests without the heavy runtime dependency. See
.kiro/specs/anpr-autogate-system/requirements.md (Requirement 3) and design.md
(OCR_Engine section) for the authoritative acceptance criteria.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from anpr.core.models import OcrResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bounds (design OCR_Engine section, Req 3.2)
# ---------------------------------------------------------------------------

# Confidence is always constrained to the inclusive range 0.0..1.0 (Req 3.2).
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0

# A candidate plate string is bounded to 1..9 alphanumeric characters: the
# Indonesian_Plate_Format's 2-letter area code + 4-digit number + 3-letter
# suffix sum to at most 9 characters (Req 3.2).
MAX_PLATE_CHARS = 9

# Characters kept from a raw OCR reading: ASCII letters and digits only.
_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]")

# The OcrResult returned when a crop yields no readable text (Req 3.2/3.3).
_NO_READABLE_TEXT = OcrResult(text=None, confidence=MIN_CONFIDENCE, timed_out=False)


def _default_ocr_factory() -> Any:
    """Construct a PaddleOCR engine configured for Latin-script plates.

    Imported lazily so the package stays importable before the heavy PaddleOCR
    dependency is installed (matching the guarded-import pattern used by the
    detector wrapper).
    """
    from paddleocr import PaddleOCR  # local import: heavy optional dependency

    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def clamp_confidence(confidence: Any) -> float:
    """Clamp/validate a recognition confidence into the inclusive [0,1] range.

    A ``None`` or non-numeric confidence collapses to ``0.0`` so a malformed
    PaddleOCR result can never produce an out-of-range confidence (Req 3.2).
    """
    if confidence is None or isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return MIN_CONFIDENCE
    return float(min(MAX_CONFIDENCE, max(MIN_CONFIDENCE, confidence)))


def bound_plate_text(raw_text: Optional[str]) -> Optional[str]:
    """Reduce a raw OCR reading to a bounded 1-9 alphanumeric candidate.

    Drops every non-alphanumeric character, uppercases the survivors, and caps
    the result at :data:`MAX_PLATE_CHARS` characters (Req 3.2). Returns ``None``
    when nothing alphanumeric remains, signalling "no readable text".
    """
    if not raw_text:
        return None
    cleaned = _NON_ALNUM_RE.sub("", raw_text).upper()
    if not cleaned:
        return None
    return cleaned[:MAX_PLATE_CHARS]


# ---------------------------------------------------------------------------
# Crop preprocessing
# ---------------------------------------------------------------------------

# Crops shorter than this are upscaled to it (aspect ratio kept) before OCR.
MIN_CROP_HEIGHT_PX = 96
# Replicated border added on every side (fraction of height) so characters at
# the crop edge are not missed by the text detector.
BORDER_RATIO = 0.15


# --- Angle correction tuning -------------------------------------------------
# The plate quadrilateral must cover at least this share of the crop.
_MIN_QUAD_AREA_RATIO = 0.25
# Accepted width/height of the corrected plate (Indonesian car ~3.2, motorcycle ~2.4).
_MIN_PLATE_ASPECT = 1.5
_MAX_PLATE_ASPECT = 6.5
# A quad whose corners sit within this share of the crop size of a straight
# rectangle is already frontal; warping it would only add blur.
_FRONTAL_TOLERANCE = 0.03
# Text-line rotation fallback: ignore tiny skew and implausibly large angles.
_MIN_SKEW_DEG = 1.0
_MAX_SKEW_DEG = 25.0


def _order_corners(pts: Any, np: Any) -> Any:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype("float32")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(sums)], pts[np.argmin(diffs)], pts[np.argmax(sums)], pts[np.argmax(diffs)]],
        dtype="float32",
    )


def _find_plate_quad(gray: Any, cv2: Any, np: Any) -> Optional[Any]:
    """Find the plate's four corners (the largest convex quadrilateral) or None."""
    height, width = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(contour) < _MIN_QUAD_AREA_RATIO * width * height:
            break
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return _order_corners(approx, np)
    return None


def _warp_to_frontal(image: Any, quad: Any, cv2: Any, np: Any) -> Optional[Any]:
    """Perspective-warp ``quad`` to a rectangle; None when not needed/plausible."""
    tl, tr, br, bl = quad
    out_w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    out_h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if out_w < 10 or out_h < 10:
        return None
    if not (_MIN_PLATE_ASPECT <= out_w / out_h <= _MAX_PLATE_ASPECT):
        return None

    # Already frontal? Compare corners with the bounding rectangle.
    height, width = image.shape[:2]
    x0, y0 = quad[:, 0].min(), quad[:, 1].min()
    x1, y1 = quad[:, 0].max(), quad[:, 1].max()
    rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype="float32")
    deviation = np.abs(quad - rect).max(axis=0)
    if deviation[0] <= _FRONTAL_TOLERANCE * width and deviation[1] <= _FRONTAL_TOLERANCE * height:
        return None

    target = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32"
    )
    matrix = cv2.getPerspectiveTransform(quad, target)
    return cv2.warpPerspective(
        image, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _deskew_by_text(image: Any, gray: Any, cv2: Any, np: Any) -> Any:
    """Rotate so the text baseline is horizontal (fallback when no quad found)."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Text is the minority class: use whichever side has fewer pixels.
    if cv2.countNonZero(mask) > mask.size / 2:
        mask = cv2.bitwise_not(mask)
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 50:
        return image
    angle = cv2.minAreaRect(points)[-1]
    # Normalise OpenCV's angle convention to a small rotation in (-45, 45].
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    if not (_MIN_SKEW_DEG <= abs(angle) <= _MAX_SKEW_DEG):
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def correct_angle(image: Any, cv2: Any, np: Any) -> Any:
    """Straighten a plate seen at an angle.

    First tries a perspective warp of the plate's quadrilateral outline to a
    frontal rectangle; when no plausible outline exists, falls back to rotating
    the crop so the text line is horizontal. Returns ``image`` unchanged when the
    plate already looks frontal or nothing reliable is found.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    quad = _find_plate_quad(gray, cv2, np)
    if quad is not None:
        warped = _warp_to_frontal(image, quad, cv2, np)
        if warped is not None:
            return warped
    return _deskew_by_text(image, gray, cv2, np)


def preprocess_crop(crop: Any, deskew: bool = True) -> Any:
    """Correct angle, upscale small crops, boost contrast, and add an edge border.

    ``deskew`` enables the perspective / rotation correction. Returns the crop
    unchanged when it is not a non-empty ``(H, W[, C])`` numpy array or OpenCV is
    unavailable, so callers and tests with stand-in crops are unaffected. Never
    raises: on any processing fault the original crop is used.
    """
    try:
        import cv2  # local import: heavy optional dependency
        import numpy as np
    except ModuleNotFoundError:
        return crop
    try:
        if not isinstance(crop, np.ndarray) or crop.ndim < 2 or crop.size == 0:
            return crop
        image = crop
        if deskew:
            image = correct_angle(image, cv2, np)
        height, width = image.shape[:2]
        if height < MIN_CROP_HEIGHT_PX:
            scale = MIN_CROP_HEIGHT_PX / float(height)
            image = cv2.resize(
                image,
                (max(1, int(round(width * scale))), MIN_CROP_HEIGHT_PX),
                interpolation=cv2.INTER_CUBIC,
            )
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        border = max(1, int(round(gray.shape[0] * BORDER_RATIO)))
        gray = cv2.copyMakeBorder(
            gray, border, border, border, border, cv2.BORDER_REPLICATE
        )
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    except Exception:  # noqa: BLE001 - preprocessing must never break OCR
        logger.debug("Crop preprocessing failed; using the raw crop.", exc_info=True)
        return crop


class PaddleOcrEngine:
    """PaddleOCR plate reader (structurally an ``OcrEngine``).

    Parameters
    ----------
    preprocess:
        When True (default) crops are upscaled, contrast-enhanced and bordered
        by :func:`preprocess_crop` before recognition.
    deskew:
        When True (default, and only with ``preprocess``) a plate seen at an
        angle is straightened (perspective warp, else text-line rotation).
    ocr_factory:
        Zero-argument callable that builds the underlying PaddleOCR engine.
        Injectable so the parsing branches can be unit tested without the
        PaddleOCR runtime dependency. The engine is lazily built on first use.
    """

    def __init__(
        self,
        *,
        ocr_factory: Callable[[], Any] = _default_ocr_factory,
        preprocess: bool = True,
        deskew: bool = True,
    ) -> None:
        self._ocr_factory = ocr_factory
        self._preprocess = preprocess
        self._deskew = deskew
        self._engine: Any = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    def _get_engine(self) -> Any:
        """Return the PaddleOCR engine, building it lazily on first use."""
        if self._engine is None:
            self._engine = self._ocr_factory()
        return self._engine

    # ------------------------------------------------------------------
    # Reading (Req 3.2)
    # ------------------------------------------------------------------
    def read_plate(self, crop: Any) -> OcrResult:
        """Read candidate plate text from a cropped region.

        Returns an :class:`OcrResult` whose ``text`` is a bounded 1-9
        alphanumeric candidate (uppercased) or ``None`` when nothing readable
        was found, and whose ``confidence`` is clamped to the inclusive
        ``[0, 1]`` range (Req 3.2). ``timed_out`` is always ``False`` here — the
        pipeline owns the OCR timeout and sets that field when it abandons a
        slow read (Req 3.5).
        """
        if crop is None:
            return _NO_READABLE_TEXT

        engine = self._get_engine()
        if self._preprocess:
            crop = preprocess_crop(crop, deskew=self._deskew)
        raw_results = engine.ocr(crop)

        candidate = _best_candidate(raw_results)
        if candidate is None:
            return _NO_READABLE_TEXT

        raw_text, raw_confidence = candidate
        text = bound_plate_text(raw_text)
        if text is None:
            # Recognized something, but nothing alphanumeric survived => no
            # readable plate text (Req 3.2).
            return _NO_READABLE_TEXT

        return OcrResult(
            text=text,
            confidence=clamp_confidence(raw_confidence),
            timed_out=False,
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _best_candidate(raw_results: Any) -> Optional[tuple[str, Any]]:
    """Merge every recognized fragment into one ``(text, confidence)`` pair.

    PaddleOCR's ``ocr`` call returns a nested structure that varies slightly by
    version, typically ``[[ [box, (text, confidence)], ... ]]`` (one inner list
    per image). A wide plate such as ``B 1234 CD`` is often detected as several
    separate fragments; keeping only the best one drops the others (typically
    the leading letter). Instead the fragments are ordered into reading order
    (rows top to bottom, left to right within a row) and concatenated; the
    confidence is the length-weighted mean. Fragments much shorter than the
    tallest one (e.g. the small expiry date under an Indonesian plate) are
    ignored. Returns ``None`` when no text was recognized.
    """
    fragments = list(_iter_fragments(raw_results))
    if not fragments:
        return None

    fragments = _drop_small_fragments(fragments)
    ordered = _reading_order(fragments)

    texts: list[str] = []
    weighted = 0.0
    total = 0
    for frag in ordered:
        cleaned = _NON_ALNUM_RE.sub("", frag.text)
        if not cleaned:
            continue
        texts.append(cleaned)
        weighted += clamp_confidence(frag.confidence) * len(cleaned)
        total += len(cleaned)
    if total == 0:
        return None
    return "".join(texts), weighted / total


class _Fragment:
    """One recognized text fragment with optional box geometry."""

    __slots__ = ("text", "confidence", "x", "y", "height")

    def __init__(
        self,
        text: str,
        confidence: Any,
        x: Optional[float] = None,
        y: Optional[float] = None,
        height: Optional[float] = None,
    ) -> None:
        self.text = text
        self.confidence = confidence
        self.x = x  # left edge
        self.y = y  # vertical centre
        self.height = height


def _iter_fragments(node: Any):
    """Yield :class:`_Fragment` objects found anywhere within a result tree.

    Recognizes the classic line shape ``[box, (text, confidence)]`` (box is a
    list of ``[x, y]`` points) and the bare ``(text, confidence)`` pair, and
    otherwise recurses into nested lists/tuples so both the classic and newer
    layouts are handled without assuming an exact shape.
    """
    if node is None:
        return
    if _is_text_confidence_pair(node):
        yield _Fragment(node[0], node[1])
        return
    if (
        isinstance(node, (list, tuple))
        and len(node) == 2
        and _is_text_confidence_pair(node[1])
    ):
        geometry = _box_geometry(node[0])
        yield _Fragment(node[1][0], node[1][1], *geometry)
        return
    if isinstance(node, (list, tuple)):
        for child in node:
            yield from _iter_fragments(child)


def _box_geometry(box: Any) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return ``(left, vertical centre, height)`` of a point-list box, or Nones."""
    try:
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
    except (TypeError, ValueError, IndexError):
        return None, None, None
    if not xs or not ys:
        return None, None, None
    return min(xs), (min(ys) + max(ys)) / 2.0, max(ys) - min(ys)


# A fragment shorter than this fraction of the tallest one is treated as small
# print (e.g. the expiry date under the plate number) and ignored.
_MIN_RELATIVE_HEIGHT = 0.6


def _drop_small_fragments(fragments: list[_Fragment]) -> list[_Fragment]:
    """Drop fragments far smaller than the tallest one (when heights are known)."""
    heights = [f.height for f in fragments if f.height]
    if not heights:
        return fragments
    tallest = max(heights)
    kept = [
        f for f in fragments
        if not f.height or f.height >= tallest * _MIN_RELATIVE_HEIGHT
    ]
    return kept or fragments


def _reading_order(fragments: list[_Fragment]) -> list[_Fragment]:
    """Order fragments into rows (top to bottom), left to right within a row.

    Without geometry the original order is kept.
    """
    if any(f.x is None or f.y is None for f in fragments):
        return fragments
    heights = [f.height for f in fragments if f.height]
    row_tolerance = 0.6 * (sum(heights) / len(heights)) if heights else 0.0

    rows: list[list[_Fragment]] = []
    for frag in sorted(fragments, key=lambda f: f.y):
        if rows and abs(frag.y - rows[-1][0].y) <= row_tolerance:
            rows[-1].append(frag)
        else:
            rows.append([frag])
    return [f for row in rows for f in sorted(row, key=lambda f: f.x)]


def _is_text_confidence_pair(node: Any) -> bool:
    """Return True when ``node`` is a ``(text: str, confidence: number)`` pair."""
    return (
        isinstance(node, (list, tuple))
        and len(node) == 2
        and isinstance(node[0], str)
        and not isinstance(node[1], bool)
        and isinstance(node[1], (int, float))
    )
