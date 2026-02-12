"""Smoke-test utility for media tool-call gating.

Usage examples:
  python media_gate_test.py
  python media_gate_test.py --tools image --cooldown true
  python media_gate_test.py --tools both --allow-autopilot-media-offers true
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chorus_engine.services.media_offer_policy import (
    EffectiveOfferPolicy,
    compute_turn_media_permissions,
)
from chorus_engine.services.media_turn_classifier import classify_media_turn


@dataclass
class FakeIntent:
    name: str
    confidence: float


@dataclass
class FakeConversation:
    allow_image_offers: str = "true"
    allow_video_offers: str = "true"
    image_offer_count: int = 0
    video_offer_count: int = 0
    last_image_offer_at: datetime | None = None
    last_video_offer_at: datetime | None = None
    last_image_offer_message_count: int | None = None
    last_video_offer_message_count: int | None = None


def _str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _build_tool_flags(tools: str) -> tuple[bool, bool]:
    if tools == "image":
        return True, False
    if tools == "video":
        return False, True
    if tools == "both":
        return True, True
    return False, False


def _stub_sid(message: str) -> Iterable[FakeIntent]:
    text = (message or "").lower()
    intents: list[FakeIntent] = []

    image_terms = ("image", "photo", "picture", "pic")
    video_terms = ("video", "animate", "animation")
    iterate_terms = (
        "try again",
        "another one",
        "same idea",
        "different angle",
        "make it",
        "more cinematic",
        "adjust",
        "tweak",
        "version",
        "iterate",
    )

    if any(t in text for t in image_terms):
        intents.append(FakeIntent(name="send_image", confidence=0.95))
    if any(t in text for t in video_terms):
        intents.append(FakeIntent(name="send_video", confidence=0.95))
    if any(t in text for t in iterate_terms):
        intents.append(FakeIntent(name="iterate_media", confidence=0.95))
    return intents


def _real_or_stub_sid(message: str, prefer_real_sid: bool) -> Iterable:
    if not prefer_real_sid:
        return _stub_sid(message)
    try:
        from chorus_engine.services.semantic_intent_detection import get_intent_detector

        detector = get_intent_detector()
        return detector.detect(message, enable_multi_intent=True, debug=False)
    except Exception:
        return _stub_sid(message)


def _expected_for_message(
    message: str,
    image_available: bool,
    video_available: bool,
    preferred_iteration_media_type: str,
) -> tuple[bool, list[str]]:
    text = message.lower()
    acknowledgements = {
        "lovely.",
        "perfect!",
        "nice.",
        "wow",
        "i love it.",
        "gorgeous.",
        "this is perfect.",
        "that's nice.",
    }
    if text in acknowledgements:
        return False, []

    if any(t in text for t in ("try again", "same idea", "more cinematic", "different angle", "make it darker", "another one", "iterate")):
        if preferred_iteration_media_type == "video":
            if video_available:
                return True, ["video.generate"]
            return False, []
        if preferred_iteration_media_type == "image":
            if image_available:
                return True, ["image.generate"]
            return False, []
        if image_available:
            return True, ["image.generate"]
        if video_available:
            return True, ["video.generate"]
        return False, []

    if any(t in text for t in ("video", "animate")):
        return (True, ["video.generate"]) if video_available else (False, [])

    if any(t in text for t in ("image", "photo", "picture")):
        return (True, ["image.generate"]) if image_available else (False, [])

    return False, []


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test media gate logic")
    parser.add_argument("--tools", choices=["image", "video", "both", "none"], default="both")
    parser.add_argument("--cooldown", choices=["true", "false"], default="false")
    parser.add_argument("--allow-autopilot-media-offers", choices=["true", "false"], default="false")
    parser.add_argument("--cooldown-blocks-explicit-requests", choices=["true", "false"], default="false")
    parser.add_argument("--cooldown-blocks-offers", choices=["true", "false"], default="true")
    parser.add_argument("--current-message-count", type=int, default=100)
    parser.add_argument("--preferred-iteration-media-type", choices=["image", "video", "none"], default="image")
    parser.add_argument("--sid-mode", choices=["stub", "real"], default="stub")
    args = parser.parse_args()
    prefer_real_sid = args.sid_mode == "real"

    image_enabled, video_enabled = _build_tool_flags(args.tools)
    cooldown_active = _str_to_bool(args.cooldown)
    allow_autopilot_media_offers = _str_to_bool(args.allow_autopilot_media_offers)
    cooldown_blocks_explicit_requests = _str_to_bool(args.cooldown_blocks_explicit_requests)
    cooldown_blocks_offers = _str_to_bool(args.cooldown_blocks_offers)

    conversation = FakeConversation()
    current_message_count = int(args.current_message_count)
    if cooldown_active:
        now = datetime.utcnow()
        conversation.last_image_offer_at = now
        conversation.last_video_offer_at = now
        conversation.last_image_offer_message_count = max(0, current_message_count - 1)
        conversation.last_video_offer_message_count = max(0, current_message_count - 1)

    policy = EffectiveOfferPolicy(
        offers_enabled=True,
        image_enabled=True,
        video_enabled=True,
        image_min_confidence=0.5,
        video_min_confidence=0.45,
        offer_cooldown_minutes=30,
        offer_min_turn_gap=8,
        max_offers_per_conversation_per_media=2,
        disabled_sources=set(),
    )

    tests = [
        "Lovely.",
        "Perfect!",
        "That's nice.",
        "Nice.",
        "Wow",
        "I love it.",
        "That's a lovely photo of you.",
        "Gorgeous.",
        "This is perfect.",
        "Can you send another image?",
        "Send me another photo of you.",
        "Make a new picture, full body shot.",
        "Generate an image of a nebula over the ocean.",
        "Make a short video of that scene.",
        "Generate a video of you walking through a glowing forest.",
        "Can you animate it?",
        "Try again, but make it darker.",
        "Same idea, different angle.",
        "More cinematic lighting, please.",
        "Another one, with more detail.",
    ]

    failures = 0
    for msg in tests:
        semantic_intents = _real_or_stub_sid(msg, prefer_real_sid=prefer_real_sid)
        signals = classify_media_turn(
            message=msg,
            semantic_intents=semantic_intents,
            explicit_image_threshold=0.5,
            explicit_video_threshold=0.45,
        )
        decision = compute_turn_media_permissions(
            turn_signals=signals,
            policy=policy,
            conversation=conversation,
            source="web",
            current_message_count=current_message_count,
            image_generation_enabled=image_enabled,
            video_generation_enabled=video_enabled,
            preferred_iteration_media_type=args.preferred_iteration_media_type,
            allow_autopilot_media_offers=allow_autopilot_media_offers,
            cooldown_blocks_explicit_requests=cooldown_blocks_explicit_requests,
            cooldown_blocks_offers=cooldown_blocks_offers,
        )

        expected_allowed, expected_tools = _expected_for_message(
            msg,
            image_enabled,
            video_enabled,
            args.preferred_iteration_media_type,
        )
        passed = (
            decision.media_tool_calls_allowed == expected_allowed
            and decision.allowed_tools_final == expected_tools
        )
        if not passed:
            failures += 1

        print(
            f"msg={msg!r} | sid=({signals.requested_media_type}, iter={signals.is_iteration_request}) "
            f"| cooldown={decision.cooldown_active} | tools={decision.allowed_tools_input} "
            f"| allowed={decision.media_tool_calls_allowed} | final={decision.allowed_tools_final} "
            f"| {'PASS' if passed else 'FAIL'}"
        )

    if failures:
        print(f"\nFAILED: {failures} test(s)")
        return 1

    print("\nAll tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
