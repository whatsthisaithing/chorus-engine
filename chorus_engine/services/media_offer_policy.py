"""
Proactive media offer policy resolution and gating.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


@dataclass
class EffectiveOfferPolicy:
    offers_enabled: bool
    image_enabled: bool
    video_enabled: bool
    image_min_confidence: float
    video_min_confidence: float
    offer_cooldown_minutes: int
    offer_min_turn_gap: int
    max_offers_per_conversation_per_media: int
    disabled_sources: set[str]


@dataclass
class ExplicitMediaRequest:
    kind: str  # image | video | none
    confidence: float
    user_requested_image: bool
    user_requested_video: bool


@dataclass
class TurnMediaPermissions:
    explicit_media_request: bool
    requested_media_type: str  # image | video | either | none
    is_iteration_request: bool
    cooldown_time_active: bool
    cooldown_message_active: bool
    cooldown_active: bool
    explicit_allowed: bool
    offer_allowed: bool
    media_tool_calls_allowed: bool
    allowed_tools_input: list[str]
    allowed_tools_final: list[str]
    explicit_request_meta: ExplicitMediaRequest
    media_offer_allowed_this_turn: dict[str, bool]
    allow_media_payload_this_turn: dict[str, bool]
    allowed_media_tools: set[str]


def resolve_effective_offer_policy(system_config, character) -> EffectiveOfferPolicy:
    media_cfg = getattr(system_config, "media_tooling", None)
    char_cfg = getattr(character, "proactive_offers", None)

    image_override = getattr(char_cfg, "image", None)
    video_override = getattr(char_cfg, "video", None)

    image_enabled = (
        media_cfg.image_offers_enabled
        if getattr(image_override, "enabled", None) is None
        else bool(image_override.enabled)
    )
    video_enabled = (
        media_cfg.video_offers_enabled
        if getattr(video_override, "enabled", None) is None
        else bool(video_override.enabled)
    )
    image_threshold = (
        media_cfg.offer_min_confidence_image
        if getattr(image_override, "min_confidence", None) is None
        else float(image_override.min_confidence)
    )
    video_threshold = (
        media_cfg.offer_min_confidence_video
        if getattr(video_override, "min_confidence", None) is None
        else float(video_override.min_confidence)
    )

    cooldown = (
        media_cfg.offer_cooldown_minutes
        if getattr(char_cfg, "offer_cooldown_minutes", None) is None
        else int(char_cfg.offer_cooldown_minutes)
    )
    turn_gap = (
        media_cfg.offer_min_turn_gap
        if getattr(char_cfg, "offer_min_turn_gap", None) is None
        else int(char_cfg.offer_min_turn_gap)
    )
    max_offers = (
        media_cfg.max_offers_per_conversation_per_media
        if getattr(char_cfg, "max_offers_per_conversation_per_media", None) is None
        else int(char_cfg.max_offers_per_conversation_per_media)
    )

    return EffectiveOfferPolicy(
        offers_enabled=bool(media_cfg.enabled and media_cfg.offers_enabled),
        image_enabled=image_enabled,
        video_enabled=video_enabled,
        image_min_confidence=image_threshold,
        video_min_confidence=video_threshold,
        offer_cooldown_minutes=cooldown,
        offer_min_turn_gap=turn_gap,
        max_offers_per_conversation_per_media=max_offers,
        disabled_sources={str(s).lower() for s in (media_cfg.disable_offers_for_sources or [])},
    )


def compute_turn_media_permissions(
    *,
    turn_signals: Any,
    policy: EffectiveOfferPolicy,
    conversation: Any,
    source: str,
    current_message_count: int,
    image_generation_enabled: bool,
    video_generation_enabled: bool,
    preferred_iteration_media_type: str = "none",
    allow_autopilot_media_offers: bool | None = None,
    cooldown_blocks_explicit_requests: bool = False,
    cooldown_blocks_offers: bool = True,
) -> TurnMediaPermissions:
    """Compute authoritative per-turn media tool eligibility before prompt assembly."""
    source = (source or "").lower()
    requested_media_type = str(getattr(turn_signals, "requested_media_type", "none") or "none")
    is_iteration_request = bool(getattr(turn_signals, "is_iteration_request", False))
    explicit_media_request = bool(getattr(turn_signals, "explicit_media_request", False))
    is_acknowledgement = bool(getattr(turn_signals, "is_acknowledgement", False))
    sid_requested_image = bool(getattr(turn_signals, "user_requested_image", False))
    sid_requested_video = bool(getattr(turn_signals, "user_requested_video", False))
    image_conf = float(getattr(turn_signals, "image_confidence", 0.0) or 0.0)
    video_conf = float(getattr(turn_signals, "video_confidence", 0.0) or 0.0)

    # Build authoritative allowed tool input list from runtime capabilities.
    allowed_tools_input: list[str] = []
    if image_generation_enabled:
        allowed_tools_input.append("image.generate")
    if video_generation_enabled:
        allowed_tools_input.append("video.generate")

    has_prior_media = preferred_iteration_media_type in {"image", "video"}

    explicit_requested_media_type = "none"
    if sid_requested_image and sid_requested_video:
        explicit_requested_media_type = "either"
    elif sid_requested_image:
        explicit_requested_media_type = "image"
    elif sid_requested_video:
        explicit_requested_media_type = "video"

    # Step 1: acknowledgement block
    if is_acknowledgement and not sid_requested_image and not sid_requested_video and not is_iteration_request:
        requested_media_type = "none"
        explicit_media_request = False
        is_iteration_request = False
        explicit_requested_media_type = "none"

    # Step 2: explicit media request from SID send intents.
    explicit_tool_candidates: list[str] = []
    if explicit_requested_media_type == "image":
        explicit_tool_candidates = ["image.generate"]
    elif explicit_requested_media_type == "video":
        explicit_tool_candidates = ["video.generate"]
    elif explicit_requested_media_type == "either":
        explicit_tool_candidates = ["image.generate", "video.generate"]
    explicit_has_available_tool = any(tool in allowed_tools_input for tool in explicit_tool_candidates)
    explicit_allowed = bool(explicit_requested_media_type != "none" and explicit_has_available_tool)

    # Step 3: SID iteration with required prior media.
    iteration_allowed = bool(
        is_iteration_request
        and has_prior_media
        and (
            (preferred_iteration_media_type == "image" and "image.generate" in allowed_tools_input)
            or (preferred_iteration_media_type == "video" and "video.generate" in allowed_tools_input)
        )
    )

    if explicit_allowed:
        requested_media_type = explicit_requested_media_type
    elif iteration_allowed:
        requested_media_type = preferred_iteration_media_type
    else:
        requested_media_type = "none"
        explicit_media_request = False

    # Step 4: assistant offer path (cooldown-limited).
    image_offer_allowed_now = bool(
        image_generation_enabled
        and is_offer_allowed(
            media_kind="image",
            policy=policy,
            conversation=conversation,
            source=source,
            current_message_count=current_message_count,
        )
    )
    video_offer_allowed_now = bool(
        video_generation_enabled
        and is_offer_allowed(
            media_kind="video",
            policy=policy,
            conversation=conversation,
            source=source,
            current_message_count=current_message_count,
        )
    )

    if allow_autopilot_media_offers is None:
        allow_autopilot_media_offers = bool(policy.offers_enabled)

    # Derive cooldown states for diagnostics.
    cooldown_time_active = bool(
        _in_cooldown(getattr(conversation, "last_image_offer_at", None), policy.offer_cooldown_minutes)
        or _in_cooldown(getattr(conversation, "last_video_offer_at", None), policy.offer_cooldown_minutes)
    )
    cooldown_message_active = bool(
        _within_turn_gap(getattr(conversation, "last_image_offer_message_count", None), current_message_count, policy.offer_min_turn_gap)
        or _within_turn_gap(getattr(conversation, "last_video_offer_message_count", None), current_message_count, policy.offer_min_turn_gap)
    )
    cooldown_active = bool(cooldown_time_active or cooldown_message_active)

    _ = cooldown_blocks_explicit_requests  # explicit/iteration always bypass cooldown
    _ = cooldown_blocks_offers  # offer cooldown is already enforced in is_offer_allowed
    offer_allowed = bool(
        allow_autopilot_media_offers
        and (image_offer_allowed_now or video_offer_allowed_now)
    )
    media_tool_calls_allowed = bool(explicit_allowed or iteration_allowed or offer_allowed)

    if not media_tool_calls_allowed:
        allowed_tools_final: list[str] = []
    elif explicit_allowed or iteration_allowed:
        requested_tool_candidates: list[str] = []
        if requested_media_type == "image":
            requested_tool_candidates = ["image.generate"]
        elif requested_media_type == "video":
            requested_tool_candidates = ["video.generate"]
        elif requested_media_type == "either":
            requested_tool_candidates = ["image.generate", "video.generate"]
        requested_available = [tool for tool in requested_tool_candidates if tool in allowed_tools_input]
        if requested_media_type == "either":
            # Deterministic preference when ambiguous.
            if "image.generate" in requested_available:
                allowed_tools_final = ["image.generate"]
            elif "video.generate" in requested_available:
                allowed_tools_final = ["video.generate"]
            else:
                allowed_tools_final = []
        else:
            allowed_tools_final = requested_available
    else:
        # Offer-only pathway uses currently eligible tools.
        allowed_tools_final = []
        if image_offer_allowed_now and "image.generate" in allowed_tools_input:
            allowed_tools_final.append("image.generate")
        if video_offer_allowed_now and "video.generate" in allowed_tools_input:
            allowed_tools_final.append("video.generate")

    explicit_kind = "none"
    explicit_confidence = 0.0
    if requested_media_type == "image":
        explicit_kind = "image"
        explicit_confidence = image_conf
    elif requested_media_type == "video":
        explicit_kind = "video"
        explicit_confidence = video_conf
    elif requested_media_type == "either":
        explicit_kind = "image" if image_conf >= video_conf else "video"
        explicit_confidence = max(image_conf, video_conf)

    return TurnMediaPermissions(
        explicit_media_request=explicit_allowed or iteration_allowed,
        requested_media_type=requested_media_type,
        is_iteration_request=is_iteration_request,
        cooldown_time_active=cooldown_time_active,
        cooldown_message_active=cooldown_message_active,
        cooldown_active=cooldown_active,
        explicit_allowed=explicit_allowed or iteration_allowed,
        offer_allowed=offer_allowed,
        media_tool_calls_allowed=media_tool_calls_allowed,
        allowed_tools_input=allowed_tools_input,
        allowed_tools_final=allowed_tools_final,
        explicit_request_meta=ExplicitMediaRequest(
            kind=explicit_kind,
            confidence=explicit_confidence,
            user_requested_image=(requested_media_type == "image"),
            user_requested_video=(requested_media_type == "video"),
        ),
        media_offer_allowed_this_turn={
            "image": image_offer_allowed_now and offer_allowed,
            "video": video_offer_allowed_now and offer_allowed,
        },
        allow_media_payload_this_turn={
            "image": "image.generate" in allowed_tools_final,
            "video": "video.generate" in allowed_tools_final,
        },
        allowed_media_tools=set(allowed_tools_final),
    )


def is_offer_allowed(
    *,
    media_kind: str,
    policy: EffectiveOfferPolicy,
    conversation,
    source: str,
    current_message_count: int,
) -> bool:
    if not policy.offers_enabled:
        return False

    if source and source.lower() in policy.disabled_sources:
        return False

    if media_kind == "image":
        if not policy.image_enabled:
            return False
        if getattr(conversation, "allow_image_offers", "true") != "true":
            return False
        if conversation.image_offer_count >= policy.max_offers_per_conversation_per_media:
            return False
        if _in_cooldown(conversation.last_image_offer_at, policy.offer_cooldown_minutes):
            return False
        if _within_turn_gap(conversation.last_image_offer_message_count, current_message_count, policy.offer_min_turn_gap):
            return False
        return True

    if media_kind == "video":
        if not policy.video_enabled:
            return False
        if getattr(conversation, "allow_video_offers", "true") != "true":
            return False
        if conversation.video_offer_count >= policy.max_offers_per_conversation_per_media:
            return False
        if _in_cooldown(conversation.last_video_offer_at, policy.offer_cooldown_minutes):
            return False
        if _within_turn_gap(conversation.last_video_offer_message_count, current_message_count, policy.offer_min_turn_gap):
            return False
        return True

    return False


def record_offer(conversation, media_kind: str, current_message_count: int) -> None:
    now = datetime.utcnow()
    if media_kind == "image":
        conversation.last_image_offer_at = now
        conversation.image_offer_count = int(conversation.image_offer_count or 0) + 1
        conversation.last_image_offer_message_count = current_message_count
    elif media_kind == "video":
        conversation.last_video_offer_at = now
        conversation.video_offer_count = int(conversation.video_offer_count or 0) + 1
        conversation.last_video_offer_message_count = current_message_count


def _in_cooldown(last_offer_at, cooldown_minutes: int) -> bool:
    if not last_offer_at or cooldown_minutes <= 0:
        return False
    return datetime.utcnow() < (last_offer_at + timedelta(minutes=cooldown_minutes))


def _within_turn_gap(last_offer_msg_count, current_message_count: int, min_turn_gap: int) -> bool:
    if last_offer_msg_count is None or min_turn_gap <= 0:
        return False
    delta = int(current_message_count) - int(last_offer_msg_count)
    return delta < min_turn_gap
