"""Continuity bootstrap generation service."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from chorus_engine.db.conversation_summary_vector_store import ConversationSummaryVectorStore
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.llm.client import LLMClient
from chorus_engine.llm.text_normalization import normalize_mojibake
from chorus_engine.models.conversation import Conversation, ConversationSummary, Memory, MemoryType
from chorus_engine.models.continuity import ContinuityArc
from chorus_engine.repositories.continuity_repository import ContinuityRepository
from chorus_engine.repositories.conversation_repository import ConversationRepository
from chorus_engine.repositories.memory_repository import MemoryRepository
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.services.json_extraction import extract_json_block
from chorus_engine.services.token_counter import TokenCounter
from chorus_engine.config.models import CharacterConfig

logger = logging.getLogger(__name__)


@dataclass
class RelationshipState:
    familiarity_level: str
    tone_baseline: List[str]
    interaction_contract: List[str]
    boundaries: List[str]
    assistant_role_frame: str


@dataclass
class ArcCandidate:
    title: str
    kind: str
    summary: str
    confidence: str


class ContinuityBootstrapService:
    """Generate and cache continuity bootstrap packets per character."""

    PROMPT_VERSION = "v2"
    SUMMARY_SIM_THRESHOLD = 0.40
    MEMORY_SIM_THRESHOLD = 0.45

    def __init__(
        self,
        db: Session,
        llm_client: LLMClient,
        llm_usage_lock: Optional[Any] = None,
        token_counter: Optional[TokenCounter] = None,
        max_tokens: int = 1024
    ):
        self.db = db
        self.llm_client = llm_client
        self.llm_usage_lock = llm_usage_lock
        self.token_counter = token_counter or TokenCounter()
        self.max_tokens = max_tokens
        self.continuity_repo = ContinuityRepository(db)
        self.conv_repo = ConversationRepository(db)
        self.memory_repo = MemoryRepository(db)
        self._embedding_service: Optional[EmbeddingService] = None
        self._memory_vector_store: Optional[VectorStore] = None
        self._summary_vector_store: Optional[ConversationSummaryVectorStore] = None

    @property
    def embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    @property
    def memory_vector_store(self) -> VectorStore:
        if self._memory_vector_store is None:
            self._memory_vector_store = VectorStore(Path("data/vector_store"))
        return self._memory_vector_store

    @property
    def summary_vector_store(self) -> ConversationSummaryVectorStore:
        if self._summary_vector_store is None:
            self._summary_vector_store = ConversationSummaryVectorStore(Path("data/vector_store"))
        return self._summary_vector_store

    def is_bootstrap_stale(self, character_id: str) -> bool:
        """Return True if summaries or memories exist after last bootstrap."""
        cache = self.continuity_repo.get_cache(character_id)
        last_bootstrap = cache.bootstrap_generated_at if cache else None

        latest_summary = (
            self.db.query(ConversationSummary)
            .join(Conversation, ConversationSummary.conversation_id == Conversation.id)
            .filter(Conversation.character_id == character_id)
            .order_by(ConversationSummary.created_at.desc())
            .first()
        )

        latest_memory = (
            self.db.query(Memory)
            .filter(Memory.character_id == character_id)
            .order_by(Memory.created_at.desc())
            .first()
        )

        latest_summary_at = latest_summary.created_at if latest_summary else None
        latest_memory_at = latest_memory.created_at if latest_memory else None

        latest_activity = None
        if latest_summary_at and latest_memory_at:
            latest_activity = max(latest_summary_at, latest_memory_at)
        else:
            latest_activity = latest_summary_at or latest_memory_at

        if not last_bootstrap:
            return latest_activity is not None
        if not latest_activity:
            return False
        return latest_activity > last_bootstrap

    async def generate_and_save(
        self,
        character: CharacterConfig,
        conversation_id: Optional[str] = None,
        force: bool = False
    ) -> Optional[dict]:
        return await self._generate(character, conversation_id, force=force, persist=True)

    async def generate_preview(
        self,
        character: CharacterConfig,
        conversation_id: Optional[str] = None
    ) -> Optional[dict]:
        return await self._generate(character, conversation_id, force=True, persist=False)

    async def _generate(
        self,
        character: CharacterConfig,
        conversation_id: Optional[str],
        force: bool,
        persist: bool
    ) -> Optional[dict]:
        character_id = character.id
        summaries = self._load_recent_summaries(character_id)
        memories = self._load_recent_memories(character_id)
        existing_arcs = self.continuity_repo.list_arcs(character_id)
        existing_state = self.continuity_repo.get_relationship_state(character_id)

        fingerprint = self._compute_fingerprint(
            character=character,
            summaries=summaries,
            memories=memories,
            arcs=existing_arcs
        )
        cache = self.continuity_repo.get_cache(character_id)
        if cache and cache.bootstrap_inputs_fingerprint == fingerprint and not force and persist:
            logger.info(
                f"[CONTINUITY] Cache up-to-date for {character_id}; skipping regen."
            )
            return {
                "cache": cache,
                "skipped": True
            }

        relationship_state = await self._build_relationship_state(
            character=character,
            summaries=summaries,
            existing_state=existing_state
        )

        candidates = await self._extract_arc_candidates(
            character=character,
            summaries=summaries,
            memories=memories,
            existing_arcs=existing_arcs
        )

        merged_arcs = self._merge_arcs(
            character_id=character_id,
            existing_arcs=existing_arcs,
            candidates=candidates,
            conversation_id=conversation_id
        )

        normalized_arcs = await self._normalize_arcs(
            character=character,
            arcs=merged_arcs
        )

        selected_core, selected_active, score_map = self._select_arcs(
            character_id=character_id,
            arcs=normalized_arcs,
            summaries=summaries,
            memories=memories
        )

        internal_packet = await self._assemble_internal_packet(
            relationship_state=relationship_state,
            core_arcs=selected_core,
            active_arcs=selected_active
        )

        user_preview = await self._assemble_user_preview(
            relationship_state=relationship_state,
            selected_arcs=selected_core + selected_active
        )

        result = {
            "skipped": False,
            "relationship_state": relationship_state,
            "candidates": candidates,
            "merged_arcs": merged_arcs,
            "normalized_arcs": normalized_arcs,
            "core_arcs": selected_core,
            "active_arcs": selected_active,
            "internal_packet": internal_packet,
            "user_preview": user_preview,
            "fingerprint": fingerprint,
            "score_map": score_map
        }

        if not persist:
            return result

        self.continuity_repo.upsert_relationship_state(
            character_id=character_id,
            familiarity_level=relationship_state.familiarity_level,
            tone_baseline=relationship_state.tone_baseline,
            interaction_contract=relationship_state.interaction_contract,
            boundaries=relationship_state.boundaries,
            assistant_role_frame=relationship_state.assistant_role_frame
        )

        for arc in normalized_arcs:
            self.continuity_repo.upsert_arc(arc)

        cache = self.continuity_repo.upsert_cache(
            character_id=character_id,
            internal_packet=internal_packet,
            user_preview=user_preview,
            generated_at=datetime.utcnow(),
            fingerprint=fingerprint
        )

        logger.info(f"[CONTINUITY] Generated cache for {character_id}")
        result["cache"] = cache
        return result

    def _load_recent_summaries(self, character_id: str, limit: int = 5) -> List[ConversationSummary]:
        latest_per_conversation = (
            self.db.query(
                ConversationSummary.conversation_id.label("conversation_id"),
                func.max(ConversationSummary.created_at).label("max_created_at")
            )
            .join(Conversation, ConversationSummary.conversation_id == Conversation.id)
            .filter(Conversation.character_id == character_id)
            .group_by(ConversationSummary.conversation_id)
            .subquery()
        )

        return (
            self.db.query(ConversationSummary)
            .join(
                latest_per_conversation,
                and_(
                    ConversationSummary.conversation_id == latest_per_conversation.c.conversation_id,
                    ConversationSummary.created_at == latest_per_conversation.c.max_created_at
                )
            )
            .order_by(ConversationSummary.created_at.desc())
            .limit(limit)
            .all()
        )

    def _load_recent_memories(self, character_id: str, limit: int = 50) -> List[Memory]:
        memories = (
            self.db.query(Memory)
            .filter(
                Memory.character_id == character_id,
                Memory.status.in_(["approved", "auto_approved"]),
                Memory.memory_type.in_([
                    MemoryType.FACT,
                    MemoryType.PROJECT,
                    MemoryType.EXPERIENCE,
                    MemoryType.STORY,
                    MemoryType.RELATIONSHIP
                ])
            )
            .order_by(Memory.created_at.desc())
            .limit(limit)
            .all()
        )
        return memories

    def _compute_fingerprint(
        self,
        character: CharacterConfig,
        summaries: List[ConversationSummary],
        memories: List[Memory],
        arcs: List[ContinuityArc]
    ) -> str:
        parts = [
            self.PROMPT_VERSION,
            character.id,
            character.role_type,
            character.immersion_level,
            "|".join([s.created_at.isoformat() for s in summaries]),
            "|".join([m.created_at.isoformat() for m in memories]),
            "|".join([a.updated_at.isoformat() for a in arcs if a.updated_at])
        ]
        raw = "||".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def _build_relationship_state(
        self,
        character: CharacterConfig,
        summaries: List[ConversationSummary],
        existing_state: Optional[Any]
    ) -> RelationshipState:
        recent_summaries = [s.summary for s in summaries if s.summary]
        user_preferences = []

        system_prompt = (
            "You are the Relationship State Extractor for a conversational AI system.\n\n"
            "Your task is to infer the CURRENT relationship stance between the USER and the CHARACTER.\n\n"
            "This is NOT a narrative summary.\n"
            "This is a compact description of tone, familiarity, and interaction expectations.\n\n"
            "Inputs:\n"
            f"- Character role_type: {character.role_type}\n"
            f"- Character immersion_level: {character.immersion_level}\n"
            f"- Durable user preferences: {user_preferences}\n"
            f"- Recent conversation summaries: {recent_summaries}\n\n"
            "Rules:\n"
            "- Be conservative.\n"
            "- Prefer stable traits over momentary moods.\n"
            "- Do NOT quote emotional lines verbatim.\n"
            "- Do NOT imply emotional dependency or exclusivity.\n"
            "- Phrase everything in neutral, explainable terms.\n\n"
            "Field guidance:\n"
            "- familiarity_level: how familiar the CHARACTER should behave with the USER\n"
            "  - new: treat as first-time\n"
            "  - familiar: recognize prior chats, light continuity\n"
            "  - established: clear ongoing relationship or collaboration\n"
            "  - close: warm, comfortable rapport (still grounded; no dependency framing)\n\n"
            "- tone_baseline: 2-4 adjectives describing default voice\n"
            "  Examples: warm, playful, direct, reflective, analytical, formal\n\n"
            "- interaction_contract: short working agreements that describe how conversations usually go\n"
            "  Examples:\n"
            "  - \"honest feedback\"\n"
            "  - \"practical, implementation-focused\"\n"
            "  - \"asks clarifying questions when needed\"\n"
            "  - \"collaborative exploration\"\n\n"
            "- boundaries: constraints appropriate to role_type and immersion\n"
            "  Examples:\n"
            "  - \"avoid dependency framing\"\n"
            "  - \"no medical or legal certainty\"\n"
            "  - \"keep claims grounded in provided context\"\n"
            "  - \"for roleplay: avoid locking in specifics unless supported\"\n\n"
            "- assistant_role_frame: one sentence describing how the CHARACTER should understand its role with this USER\n"
            "  Examples:\n"
            "  - assistant: \"A collaborative technical partner focused on building reliable systems.\"\n"
            "  - companion: \"A warm, reflective conversational partner for exploring ideas together.\"\n\n"
            "Output valid JSON ONLY in this exact schema:\n"
            "{\n"
            "  \"familiarity_level\": \"new | familiar | established | close\",\n"
            "  \"tone_baseline\": [\"...\"],\n"
            "  \"interaction_contract\": [\"...\"],\n"
            "  \"boundaries\": [\"...\"],\n"
            "  \"assistant_role_frame\": \"...\"\n"
            "}\n\n"
            "If insufficient information exists, choose the safest, most neutral options."
        )

        response = await self._call_llm(system_prompt, "", max_tokens=self.max_tokens, temperature=0.0)
        data = self._parse_json_object(response) or {}

        familiarity = data.get("familiarity_level") or "new"
        tone = data.get("tone_baseline") or []
        interaction = data.get("interaction_contract") or []
        boundaries = data.get("boundaries") or []
        role_frame = data.get("assistant_role_frame") or ""

        if not tone and existing_state:
            tone = existing_state.tone_baseline or []
        if not role_frame and existing_state:
            role_frame = existing_state.assistant_role_frame or ""

        return RelationshipState(
            familiarity_level=familiarity,
            tone_baseline=tone,
            interaction_contract=interaction,
            boundaries=boundaries,
            assistant_role_frame=role_frame
        )

    async def _extract_arc_candidates(
        self,
        character: CharacterConfig,
        summaries: List[ConversationSummary],
        memories: List[Memory],
        existing_arcs: List[ContinuityArc]
    ) -> List[ArcCandidate]:
        summaries_text = [s.summary for s in summaries if s.summary]
        memory_items = [m.content for m in memories if m.content]
        existing = [
            {
                "title": a.title,
                "kind": a.kind,
                "summary": a.summary,
                "confidence": a.confidence
            }
            for a in existing_arcs
        ]

        system_prompt = (
            "You are an Active Arc Extractor.\n\n"
            "Your task is to identify shared, ongoing contexts between the USER and the CHARACTER that have continuity value.\n\n"
            "An Active Arc is NOT just a topic. It is a continuing project, theme, question, story thread, relationship dynamic, or constraint.\n\n"
            f"Inputs:\n"
            f"- Character role_type: {character.role_type}\n"
            f"- Character immersion_level: {character.immersion_level}\n"
            f"- Conversation summaries: {summaries_text}\n"
            f"- Durable memory items: {memory_items}\n"
            f"- Existing arcs: {existing}\n\n"
            "Rules:\n"
            "- Extract only arcs that appear recurring, unresolved, or central.\n"
            "- Prefer fewer, higher-quality arcs.\n"
            "- Phrase arcs in a way appropriate to role_type and immersion_level.\n"
            "- Avoid overly specific details unless strongly supported.\n"
            "- Avoid emotional dependency framing.\n\n"
            "- Treat existing arcs as weak hypotheses, not ground truth.\n"
            "- Keep an existing arc only when recent summaries/memories support it.\n"
            "- If an existing arc lacks recent support, drop it or lower its confidence.\n"
            "- Prefer arcs that are visible in the most recent conversations.\n\n"
            "Output JSON ONLY as an array of objects:\n"
            "[\n"
            "  {\n"
            "    \"title\": \"Short arc name\",\n"
            "    \"kind\": \"project | theme | question | story | relationship | constraint | meta\",\n"
            "    \"summary\": \"1-2 sentence natural-language description\",\n"
            "    \"confidence\": \"high | medium | low\"\n"
            "  }\n"
            "]"
        )

        response = await self._call_llm(system_prompt, "", max_tokens=self.max_tokens, temperature=0.0)
        data = self._parse_json_array(response) or []
        candidates: List[ArcCandidate] = []
        for item in data:
            try:
                candidates.append(
                    ArcCandidate(
                        title=normalize_mojibake(str(item.get("title", "")).strip()),
                        kind=normalize_mojibake(str(item.get("kind", "theme")).strip()),
                        summary=normalize_mojibake(str(item.get("summary", "")).strip()),
                        confidence=normalize_mojibake(str(item.get("confidence", "medium")).strip()),
                    )
                )
            except Exception:
                continue
        return [c for c in candidates if c.title and c.summary]

    async def _normalize_arcs(
        self,
        character: CharacterConfig,
        arcs: List[ContinuityArc]
    ) -> List[ContinuityArc]:
        if not arcs:
            return []

        input_payload = [
            {
                "title": normalize_mojibake(a.title),
                "kind": normalize_mojibake(a.kind),
                "summary": normalize_mojibake(a.summary),
                "confidence": normalize_mojibake(a.confidence)
            }
            for a in arcs
        ]

        companion_guidance = ""
        if character.role_type == "companion":
            companion_guidance = (
                "- For companion characters, prefer second-person framing (\"You and X...\") "
                "instead of third-person reporting (\"X and Y...\").\n"
            )
        system_prompt = (
            "You are an Arc Normalizer.\n\n"
            "Your task is to rewrite arc summaries so they are:\n"
            "- Natural\n"
            "- Non-creepy\n"
            "- Appropriate to the character's role_type and immersion_level\n\n"
            f"Inputs:\n"
            f"- Character role_type: {character.role_type}\n"
            f"- Character immersion_level: {character.immersion_level}\n"
            f"- Arc candidates: {input_payload}\n\n"
            "Rules:\n"
            "- Do NOT change the meaning of arcs.\n"
            "- Reduce specificity if there is uncertainty.\n"
            "- For assistants: emphasize projects, decisions, constraints.\n"
            "- For companions: emphasize shared exploration and themes.\n"
            f"{companion_guidance}"
            "- For roleplay: preserve narrative continuity without locking into specifics.\n\n"
            "Output JSON ONLY in the same schema as input."
        )

        response = await self._call_llm(system_prompt, "", max_tokens=self.max_tokens, temperature=0.0)
        data = self._parse_json_array(response) or []
        normalized: List[ContinuityArc] = []
        arc_by_key = {self._normalize_key(a.title): a for a in arcs}
        for item in data:
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            key = self._normalize_key(title)
            source = arc_by_key.get(key)
            if not source:
                continue
            source.summary = normalize_mojibake(
                str(item.get("summary", source.summary)).strip() or source.summary
            )
            source.kind = normalize_mojibake(
                str(item.get("kind", source.kind)).strip() or source.kind
            )
            source.confidence = normalize_mojibake(
                str(item.get("confidence", source.confidence)).strip() or source.confidence
            )
            source.updated_at = datetime.utcnow()
            normalized.append(source)
        return normalized or arcs

    def _merge_arcs(
        self,
        character_id: str,
        existing_arcs: List[ContinuityArc],
        candidates: List[ArcCandidate],
        conversation_id: Optional[str]
    ) -> List[ContinuityArc]:
        by_key: Dict[str, ContinuityArc] = {
            self._normalize_key(a.title): a for a in existing_arcs
        }
        now = datetime.utcnow()

        for cand in candidates:
            key = self._normalize_key(cand.title)
            if key in by_key:
                arc = by_key[key]
                arc.summary = cand.summary or arc.summary
                arc.kind = cand.kind or arc.kind
                arc.confidence = cand.confidence or arc.confidence
                arc.frequency_count = (arc.frequency_count or 0) + 1
            else:
                arc = ContinuityArc(
                    character_id=character_id,
                    title=cand.title,
                    kind=cand.kind,
                    summary=cand.summary,
                    confidence=cand.confidence,
                    status="active",
                    stickiness="normal",
                    frequency_count=1
                )
                by_key[key] = arc
            arc.last_touched_conversation_id = conversation_id
            arc.last_touched_conversation_at = now
        return list(by_key.values())

    def _select_arcs(
        self,
        character_id: str,
        arcs: List[ContinuityArc],
        summaries: List[ConversationSummary],
        memories: List[Memory]
    ) -> Tuple[List[ContinuityArc], List[ContinuityArc], Dict[str, dict]]:
        if not arcs:
            return [], [], {}

        # Use the latest summary per conversation in the recent window.
        recent_summaries = summaries
        recent_memories = memories[:15]
        summary_embeddings, summary_embed_stats = self._get_recent_summary_embeddings(
            character_id, recent_summaries
        )
        memory_embeddings, memory_embed_stats = self._get_recent_memory_embeddings(
            character_id, recent_memories
        )
        scored = []
        info_by_arc_key: Dict[str, dict] = {}

        for arc in arcs:
            score_info = self._score_arc(
                character_id=character_id,
                arc=arc,
                recent_summaries=recent_summaries,
                recent_memories=recent_memories,
                summary_embeddings=summary_embeddings,
                summary_embed_stats=summary_embed_stats,
                memory_embeddings=memory_embeddings,
                memory_embed_stats=memory_embed_stats
            )
            arc_key = arc.id or self._normalize_key(arc.title)
            info_by_arc_key[arc_key] = score_info
            scored.append((score_info["core_score"], score_info["active_score"], arc, score_info))

        # Core ranking favors stability with low recent-evidence contribution.
        scored_core = sorted(scored, key=lambda item: item[0], reverse=True)
        # Active ranking is more responsive to recent evidence.
        scored_active = sorted(scored, key=lambda item: item[1], reverse=True)

        # Hysteresis: only promote arcs to core when evidence persists across recent context.
        core_persistent = []
        for _, _, arc, info in scored_core:
            if info["summary_hits"] >= 2 or info["distinct_conversation_hits"] >= 2:
                core_persistent.append(arc)

        if core_persistent:
            core = self._select_diverse(core_persistent, max_count=3)
        else:
            core = self._select_diverse([arc for _, _, arc, _ in scored_core], max_count=3)

        remaining = [arc for _, _, arc, _ in scored_active if arc not in core]
        active_evidence = []
        for arc in remaining:
            arc_key = arc.id or self._normalize_key(arc.title)
            info = info_by_arc_key.get(arc_key, {})
            if (
                info.get("summary_hits", 0) >= 1
                or info.get("memory_hits", 0) >= 1
                or info.get("recent_evidence_score", 0.0) >= 0.15
            ):
                active_evidence.append(arc)
        active_pool = active_evidence or remaining
        active = self._select_diverse(active_pool, max_count=7)

        score_map: Dict[str, dict] = {}
        for _, _, arc, info in scored:
            if not arc:
                continue
            key = arc.id or self._normalize_key(arc.title)
            score_map[key] = info
        return core, active, score_map

    def _select_diverse(self, arcs: List[ContinuityArc], max_count: int) -> List[ContinuityArc]:
        selected: List[ContinuityArc] = []
        seen_kinds = set()
        for arc in arcs:
            if len(selected) >= max_count:
                break
            if arc.kind not in seen_kinds or len(selected) < 2:
                selected.append(arc)
                seen_kinds.add(arc.kind)
        return selected

    def _score_arc(
        self,
        character_id: str,
        arc: ContinuityArc,
        recent_summaries: List[ConversationSummary],
        recent_memories: List[Memory],
        summary_embeddings: List[Tuple[ConversationSummary, List[float]]],
        summary_embed_stats: Dict[str, int],
        memory_embeddings: List[Tuple[Memory, List[float]]],
        memory_embed_stats: Dict[str, int]
    ) -> dict:
        confidence_weight = {"high": 1.0, "medium": 0.7, "low": 0.4}
        stickiness_weight = {"high": 1.2, "normal": 1.0, "low": 0.8}
        confidence_factor = confidence_weight.get(arc.confidence, 0.7)
        stickiness_factor = stickiness_weight.get(arc.stickiness, 1.0)
        base = confidence_factor * stickiness_factor

        recency_penalty = 0.0
        conversations_ago = 0
        if arc.last_touched_conversation_at:
            count = (
                self.db.query(Conversation)
                .filter(
                    Conversation.character_id == character_id,
                    Conversation.updated_at > arc.last_touched_conversation_at
                )
                .count()
            )
            conversations_ago = count
            recency_penalty = min(0.3, 0.05 * count)

        arc_vector = None
        try:
            arc_vector = self.embedding_service.embed(f"{arc.title} - {arc.summary}")
        except Exception:
            arc_vector = None

        summary_hits = 0
        memory_hits = 0
        distinct_conversation_ids = set()
        summary_similarity_details: List[Dict[str, Any]] = []
        memory_similarity_details: List[Dict[str, Any]] = []
        if arc_vector:
            for summary, summary_vector in summary_embeddings:
                score = self._cosine_similarity(arc_vector, summary_vector)
                summary_similarity_details.append(
                    {
                        "summary_id": summary.id,
                        "conversation_id": summary.conversation_id,
                        "similarity": score,
                        "excerpt": (summary.summary or "")[:140]
                    }
                )
                if score >= self.SUMMARY_SIM_THRESHOLD:
                    summary_hits += 1
                    if summary.conversation_id:
                        distinct_conversation_ids.add(summary.conversation_id)
            for memory, memory_vector in memory_embeddings:
                score = self._cosine_similarity(arc_vector, memory_vector)
                memory_similarity_details.append(
                    {
                        "memory_id": memory.id,
                        "conversation_id": memory.conversation_id,
                        "similarity": score,
                        "excerpt": (memory.content or "")[:140]
                    }
                )
                if score >= self.MEMORY_SIM_THRESHOLD:
                    memory_hits += 1
                    if memory.conversation_id:
                        distinct_conversation_ids.add(memory.conversation_id)

        top_summary_matches = sorted(
            summary_similarity_details,
            key=lambda item: item["similarity"],
            reverse=True
        )[:3]
        top_memory_matches = sorted(
            memory_similarity_details,
            key=lambda item: item["similarity"],
            reverse=True
        )[:3]

        summary_norm = min(1.0, summary_hits / max(1, len(recent_summaries)))
        memory_norm = min(1.0, memory_hits / max(1, len(recent_memories)))
        distinct_conversation_hits = len(distinct_conversation_ids)
        recent_evidence_score = min(
            1.0,
            max(0.0, (0.70 * summary_norm) + (0.30 * memory_norm))
        )

        base_score = max(0.0, base * (1.0 - recency_penalty))
        core_score = (base_score * 0.85) + (recent_evidence_score * 0.15)
        active_score = (base_score * 0.60) + (recent_evidence_score * 0.40)
        return {
            "score": active_score,
            "base_score": base_score,
            "core_score": core_score,
            "active_score": active_score,
            "recent_evidence_score": recent_evidence_score,
            "base": base,
            "confidence_factor": confidence_factor,
            "stickiness_factor": stickiness_factor,
            "recency_penalty": recency_penalty,
            "conversations_ago": conversations_ago,
            "summary_hits": summary_hits,
            "memory_hits": memory_hits,
            "distinct_conversation_hits": distinct_conversation_hits,
            "summary_context_count": len(recent_summaries),
            "summary_ids_used": [s.id for s in recent_summaries if s.id],
            "summary_vectors_used": len(summary_embeddings),
            "summary_vectors_embedded_on_the_fly": summary_embed_stats.get("embedded_on_the_fly", 0),
            "summary_vector_lookup_errors": summary_embed_stats.get("lookup_errors", 0),
            "summary_vector_embed_errors": summary_embed_stats.get("embed_errors", 0),
            "memory_vectors_used": len(memory_embeddings),
            "memory_vector_lookup_errors": memory_embed_stats.get("lookup_errors", 0),
            "memory_vector_embed_errors": memory_embed_stats.get("embed_errors", 0),
            "missing_summary_vectors": max(0, len(recent_summaries) - len(summary_embeddings)),
            "missing_memory_vectors": memory_embed_stats.get("missing_vectors", 0),
            "memory_vectors_embedded_on_the_fly": memory_embed_stats.get("embedded_on_the_fly", 0),
            "top_summary_matches": top_summary_matches,
            "top_memory_matches": top_memory_matches,
            "summary_similarity_threshold": self.SUMMARY_SIM_THRESHOLD,
            "memory_similarity_threshold": self.MEMORY_SIM_THRESHOLD
        }

    def _get_recent_summary_embeddings(
        self,
        character_id: str,
        summaries: List[ConversationSummary]
    ) -> Tuple[List[Tuple[ConversationSummary, List[float]]], Dict[str, int]]:
        embedded: List[Tuple[ConversationSummary, List[float]]] = []
        stats = {
            "embedded_on_the_fly": 0,
            "lookup_errors": 0,
            "embed_errors": 0
        }
        for summary in summaries:
            if not summary.conversation_id:
                continue
            vector = None
            try:
                record = self.summary_vector_store.get_summary(character_id, summary.conversation_id)
                vector = record.get("embedding") if record else None
                # Chroma may return numpy arrays; avoid truthiness checks.
                if vector is not None:
                    try:
                        if len(vector) > 0:
                            embedded.append((summary, vector))
                            continue
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"[CONTINUITY] Failed summary embedding lookup for {summary.id}: {e}")
                stats["lookup_errors"] += 1
            if summary.summary:
                try:
                    fallback_vector = self.embedding_service.embed(summary.summary)
                    embedded.append((summary, fallback_vector))
                    stats["embedded_on_the_fly"] += 1
                except Exception:
                    stats["embed_errors"] += 1
        return embedded, stats

    def _get_recent_memory_embeddings(
        self,
        character_id: str,
        memories: List[Memory]
    ) -> Tuple[List[Tuple[Memory, List[float]]], Dict[str, int]]:
        embedded: List[Tuple[Memory, List[float]]] = []
        stats = {
            "embedded_on_the_fly": 0,
            "missing_vectors": 0,
            "lookup_errors": 0,
            "embed_errors": 0
        }

        vector_id_to_embedding: Dict[str, List[float]] = {}
        vector_ids = [m.vector_id for m in memories if m.vector_id]
        if vector_ids:
            try:
                collection = self.memory_vector_store.get_collection(character_id)
                if collection is not None:
                    results = collection.get(ids=vector_ids, include=["embeddings"])
                    ids = results.get("ids") or []
                    embeddings = results.get("embeddings") or []
                    for idx, vector_id in enumerate(ids):
                        if idx < len(embeddings) and embeddings[idx] is not None:
                            vector_id_to_embedding[vector_id] = embeddings[idx]
            except Exception as e:
                logger.debug(f"[CONTINUITY] Failed memory embedding bulk lookup: {e}")
                stats["lookup_errors"] += 1

        for memory in memories:
            vector = vector_id_to_embedding.get(memory.vector_id) if memory.vector_id else None
            if vector is None and memory.content:
                try:
                    # Missing extracted-memory vectors are embedded on demand for scoring only.
                    vector = self.embedding_service.embed(memory.content)
                    stats["embedded_on_the_fly"] += 1
                except Exception:
                    vector = None
                    stats["embed_errors"] += 1
            if vector is not None:
                embedded.append((memory, vector))
            else:
                stats["missing_vectors"] += 1
        return embedded, stats

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        if vec_a is None or vec_b is None:
            return 0.0
        try:
            if len(vec_a) == 0 or len(vec_b) == 0 or len(vec_a) != len(vec_b):
                return 0.0
        except Exception:
            return 0.0
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for a, b in zip(vec_a, vec_b):
            dot += a * b
            norm_a += a * a
            norm_b += b * b
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))

    async def _assemble_internal_packet(
        self,
        relationship_state: RelationshipState,
        core_arcs: List[ContinuityArc],
        active_arcs: List[ContinuityArc]
    ) -> str:
        lines = [
            "[Conversation Bootstrap]",
            "",
            "Relationship State:",
            f"- Familiarity: {relationship_state.familiarity_level}",
            f"- Tone: {', '.join(relationship_state.tone_baseline)}",
            f"- How we interact: {', '.join(relationship_state.interaction_contract)}",
            f"- Boundaries: {', '.join(relationship_state.boundaries)}",
            "",
            "Core Context:"
        ]

        for idx, arc in enumerate(core_arcs, start=1):
            lines.append(f"{idx}) {arc.title} - {arc.summary}")

        if active_arcs:
            lines.append("")
            lines.append("Active Arcs:")
            for idx, arc in enumerate(active_arcs, start=1):
                lines.append(f"{idx}) {arc.title} - {arc.summary}")

        lines.extend(
            [
                "",
                "Instruction:",
                "Use this information to restore continuity and stance at the start of the conversation.",
                "Use subtly and naturally—only when relevant.",
                "Do not quote, paraphrase, or summarize this content to the user.",
                "",
                "[/Conversation Bootstrap]"
            ]
        )
        return normalize_mojibake("\n".join(lines)).strip()

    async def _assemble_user_preview(
        self,
        relationship_state: RelationshipState,
        selected_arcs: List[ContinuityArc]
    ) -> str:
        lines = [
            "If you'd like, I can carry forward this context:",
            ""
        ]

        for arc in selected_arcs:
            lines.append(f"- {arc.summary}")

        lines.extend(
            [
                "",
                "Or we can start fresh—either is fine."
            ]
        )
        return normalize_mojibake("\n".join(lines)).strip()

    def _format_arc_list(self, arcs: List[ContinuityArc]) -> List[dict]:
        return [
            {
                "title": normalize_mojibake(a.title),
                "kind": normalize_mojibake(a.kind),
                "summary": normalize_mojibake(a.summary)
            }
            for a in arcs
        ]

    def _normalize_key(self, text: str) -> str:
        return "".join(c.lower() for c in (text or "") if c.isalnum() or c.isspace()).strip()

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float = 0.0
    ) -> str:
        system_prompt = normalize_mojibake(system_prompt)
        user_prompt = normalize_mojibake(user_prompt)
        if self.llm_usage_lock is not None:
            async with self.llm_usage_lock:
                response = await self.llm_client.generate(
                    system_prompt=system_prompt,
                    prompt=user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
        else:
            response = await self.llm_client.generate(
                system_prompt=system_prompt,
                prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
        return normalize_mojibake(response.content or "")

    def _parse_json_object(self, response: str) -> Optional[Dict[str, Any]]:
        data, _ = extract_json_block(response, "object")
        if isinstance(data, dict):
            return data
        return None

    def _parse_json_array(self, response: str) -> Optional[List[Dict[str, Any]]]:
        data, _ = extract_json_block(response, "array")
        if isinstance(data, list):
            return data
        return None
