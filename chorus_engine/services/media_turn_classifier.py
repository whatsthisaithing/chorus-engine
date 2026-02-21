"""
Turn-level media request classification.

SID-centric classifier with a compact deterministic acknowledgement block.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


ACKNOWLEDGEMENT_PATTERNS = (
    r"^\s*(perfect|lovely|nice|great|awesome|beautiful)\W*\s*$",
    r"^\s*(love it|that's nice|looks good)\W*\s*$",
)

ACK_MAX_WORDS = 6
ACK_MAX_CHARS = 40

IMAGE_LEXICAL_PATTERNS = (
    r"\b(selfie|photo|picture|pic|image)\b",
    r"\b(send|show|share|take|generate|make|create|draw)\b.{0,24}\b(selfie|photo|picture|pic|image)\b",
    r"\b(selfie|photo|picture|pic|image)\b.{0,24}\b(please|pls|now|again|another)\b",
)

VIDEO_LEXICAL_PATTERNS = (
    r"\b(video|clip|animation|movie)\b",
    r"\b(send|show|share|generate|make|create|render)\b.{0,24}\b(video|clip|animation|movie)\b",
    r"\b(video|clip|animation|movie)\b.{0,24}\b(please|pls|now|again|another)\b",
)

NEGATION_PATTERNS = (
    r"\b(don't|do not|no)\s+(send|show|make|create|generate)\b",
    r"\b(not)\s+(a\s+)?(selfie|photo|picture|pic|image|video|clip|animation|movie)\b",
)

ITERATION_PATTERNS = (
    r"\b(another|again|retry|redo|regenerate|variation|different one)\b",
    r"\b(try|do)\s+(another|again)\b",
)

MEDIA_ANCHOR_PATTERNS = (
    r"\b(image|photo|picture|pic|selfie|video|clip|animation|render|frame|shot)\b",
    r"\b(that|this|last|previous)\s+(image|photo|picture|pic|selfie|video|clip|animation|render)\b",
)


@dataclass
class MediaTurnSignals:
    explicit_media_request: bool = False
    requested_media_type: str = "none"  # image | video | either | none
    is_iteration_request: bool = False
    is_acknowledgement: bool = False
    user_requested_image: bool = False
    user_requested_video: bool = False
    media_request_hint: bool = False
    image_confidence: float = 0.0
    video_confidence: float = 0.0


def classify_media_turn(
    message: str,
    semantic_intents: Iterable,
    explicit_image_threshold: float = 0.5,
    explicit_video_threshold: float = 0.45,
    soft_hint_threshold: float = 0.42,
) -> MediaTurnSignals:
    """
    SID-centric explicit/iteration classification plus deterministic ack block.
    """
    _ = soft_hint_threshold  # maintained for compatibility with call sites
    text = (message or "").strip().lower()
    if not text:
        return MediaTurnSignals()

    image_conf = 0.0
    video_conf = 0.0
    iterate_conf = 0.0
    for intent in semantic_intents or []:
        intent_name = getattr(intent, "name", None)
        conf = float(getattr(intent, "confidence", 0.0) or 0.0)
        if intent_name == "send_image":
            image_conf = max(image_conf, conf)
        elif intent_name == "send_video":
            video_conf = max(video_conf, conf)
        elif intent_name == "iterate_media":
            iterate_conf = max(iterate_conf, conf)

    explicit_image = image_conf >= explicit_image_threshold
    explicit_video = video_conf >= explicit_video_threshold
    is_iteration_request = iterate_conf > 0.0

    # Deterministic lexical fallback when SID confidence is absent/low.
    # This keeps direct requests usable even if embedding-backed SID is unavailable.
    has_negation = any(re.search(pattern, text) for pattern in NEGATION_PATTERNS)
    if not has_negation:
        lexical_image = any(re.search(pattern, text) for pattern in IMAGE_LEXICAL_PATTERNS)
        lexical_video = any(re.search(pattern, text) for pattern in VIDEO_LEXICAL_PATTERNS)
        lexical_iteration = any(re.search(pattern, text) for pattern in ITERATION_PATTERNS)
        lexical_media_anchor = any(re.search(pattern, text) for pattern in MEDIA_ANCHOR_PATTERNS)
        if lexical_image:
            explicit_image = True
            image_conf = max(image_conf, explicit_image_threshold)
        if lexical_video:
            explicit_video = True
            video_conf = max(video_conf, explicit_video_threshold)
        if lexical_iteration and (lexical_media_anchor or lexical_image or lexical_video):
            is_iteration_request = True

    # Deterministic short acknowledgement block applies only when SID has no
    # explicit media request and no iteration intent.
    is_short_ack = (len(text.split()) <= ACK_MAX_WORDS) or (len(text) <= ACK_MAX_CHARS)
    is_acknowledgement = bool(
        is_short_ack
        and any(re.search(pattern, text) for pattern in ACKNOWLEDGEMENT_PATTERNS)
        and not explicit_image
        and not explicit_video
        and not is_iteration_request
    )

    requested_media_type = "none"
    if explicit_image and explicit_video:
        requested_media_type = "either"
    elif explicit_image:
        requested_media_type = "image"
    elif explicit_video:
        requested_media_type = "video"
    elif is_iteration_request:
        requested_media_type = "either"

    explicit_media_request = bool((explicit_image or explicit_video or is_iteration_request) and requested_media_type != "none")

    return MediaTurnSignals(
        explicit_media_request=explicit_media_request,
        requested_media_type=requested_media_type,
        is_iteration_request=is_iteration_request,
        is_acknowledgement=is_acknowledgement,
        user_requested_image=explicit_image,
        user_requested_video=explicit_video,
        media_request_hint=False,
        image_confidence=image_conf,
        video_confidence=video_conf,
    )
