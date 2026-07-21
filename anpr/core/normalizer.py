"""Plate normalization logic for the ANPR Autogate System.

The :class:`PlateNormalizer` turns raw OCR output into a standard, comparable
Indonesian license plate string. It is pure core logic with no I/O: it takes a
raw string and returns a :class:`~anpr.core.models.NormalizationResult`, so it
can be reused by the detection pipeline and by the admin resident CRUD flow
without any coupling to concrete peers (Requirement 14.4).

See .kiro/specs/anpr-autogate-system/requirements.md (Requirement 4) for the
authoritative acceptance criteria implemented here.
"""

from __future__ import annotations

import re

from anpr.core.models import NormalizationResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum alphanumeric characters allowed before the plate is rejected outright
# as over-length: a 2-letter area code + 4-digit number + 3-letter suffix = 9,
# but the requirement permits up to 12 before marking format-invalid (Req 4.2).
_MAX_ALNUM_CHARS = 12

# Characters retained during normalization: ASCII letters and digits only.
# Everything else (whitespace, punctuation, symbols) is dropped (Req 4.1).
_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]")

# Indonesian_Plate_Format: a 1-2 letter area code, a 1-4 digit number, and a
# 1-3 letter suffix, evaluated against the already-uppercased string (Req 4.4).
_PLATE_FORMAT_RE = re.compile(r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$")


class PlateNormalizer:
    """Cleans and validates raw OCR text into the Indonesian_Plate_Format.

    The transformation is deterministic and idempotent: feeding an already
    normalized string back through :meth:`normalize` yields an identical
    normalized string (Req 4.6), because the cleaning step only ever removes
    characters and uppercases, both of which are no-ops on a clean string.
    """

    def normalize(self, raw: str) -> NormalizationResult:
        """Normalize ``raw`` OCR text into a :class:`NormalizationResult`.

        Steps, in order:
        1. Drop every character outside ``[A-Za-z0-9]`` (Req 4.1).
        2. Convert the survivors to uppercase (Req 4.3).
        3. Reject an empty result, an over-length result (> 12 alphanumeric
           characters, Req 4.2), or a result that does not conform to the
           Indonesian_Plate_Format (Req 4.4).
        4. On rejection, retain the original raw text and attach a non-empty,
           human-readable rejection reason (Req 4.5).
        """
        # Guard against a None slipping in from an OCR result with no text.
        raw_text = raw if raw is not None else ""

        # Steps 1 & 2: keep only alphanumerics, then uppercase (Req 4.1, 4.3).
        normalized = _NON_ALNUM_RE.sub("", raw_text).upper()

        # An empty normalized string can never conform to the format (Req 4.4);
        # reject it with an explicit reason rather than a bare mismatch.
        if not normalized:
            return NormalizationResult(
                normalized=normalized,
                is_valid=False,
                raw=raw_text,
                reason="No alphanumeric characters found in OCR text.",
            )

        # Req 4.2: more than 12 alphanumeric characters is too long to be a
        # valid Indonesian plate and is marked format-invalid.
        if len(normalized) > _MAX_ALNUM_CHARS:
            return NormalizationResult(
                normalized=normalized,
                is_valid=False,
                raw=raw_text,
                reason=(
                    f"Too many alphanumeric characters: {len(normalized)} "
                    f"exceeds the maximum of {_MAX_ALNUM_CHARS}."
                ),
            )

        # Req 4.4: conforms to the Indonesian_Plate_Format => format-valid.
        if _PLATE_FORMAT_RE.match(normalized):
            return NormalizationResult(
                normalized=normalized,
                is_valid=True,
                raw=raw_text,
                reason=None,
            )

        # Req 4.5: does not conform => format-invalid, retain raw text and give
        # a distinct, human-readable rejection reason for the guard dashboard.
        return NormalizationResult(
            normalized=normalized,
            is_valid=False,
            raw=raw_text,
            reason=(
                f"Normalized text '{normalized}' does not match the "
                "Indonesian plate format (1-2 letters, 1-4 digits, "
                "1-3 letters)."
            ),
        )
