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


class PaddleOcrEngine:
    """PaddleOCR plate reader (structurally an ``OcrEngine``).

    Parameters
    ----------
    ocr_factory:
        Zero-argument callable that builds the underlying PaddleOCR engine.
        Injectable so the parsing branches can be unit tested without the
        PaddleOCR runtime dependency. The engine is lazily built on first use.
    """

    def __init__(
        self,
        *,
        ocr_factory: Callable[[], Any] = _default_ocr_factory,
    ) -> None:
        self._ocr_factory = ocr_factory
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
    """Pick the highest-confidence ``(text, confidence)`` pair from PaddleOCR.

    PaddleOCR's ``ocr`` call returns a nested structure that varies slightly by
    version, typically ``[[ [box, (text, confidence)], ... ]]`` (one inner list
    per image). This walks that structure tolerantly, collecting every
    ``(text, confidence)`` pair it can find and returning the one with the
    highest confidence. Returns ``None`` when no text was recognized.
    """
    best: Optional[tuple[str, Any]] = None
    best_confidence = float("-inf")
    for text, confidence in _iter_text_confidence(raw_results):
        numeric = clamp_confidence(confidence)
        if numeric > best_confidence:
            best_confidence = numeric
            best = (text, confidence)
    return best


def _iter_text_confidence(node: Any):
    """Yield ``(text, confidence)`` pairs found anywhere within a result tree.

    Recognizes the PaddleOCR line shape ``(text, confidence)`` where ``text`` is
    a string and ``confidence`` is a number, and otherwise recurses into nested
    lists/tuples so both the classic and newer nested result layouts are
    handled without assuming an exact shape.
    """
    if node is None:
        return
    if _is_text_confidence_pair(node):
        yield node[0], node[1]
        return
    if isinstance(node, (list, tuple)):
        for child in node:
            yield from _iter_text_confidence(child)


def _is_text_confidence_pair(node: Any) -> bool:
    """Return True when ``node`` is a ``(text: str, confidence: number)`` pair."""
    return (
        isinstance(node, (list, tuple))
        and len(node) == 2
        and isinstance(node[0], str)
        and not isinstance(node[1], bool)
        and isinstance(node[1], (int, float))
    )
