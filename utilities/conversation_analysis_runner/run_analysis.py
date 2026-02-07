"""
Run conversation analysis (summary + archivist memories) without persisting results.

Usage:
    python utilities/conversation_analysis_runner/run_analysis.py <conversation_id>

Example:
    python utilities/conversation_analysis_runner/run_analysis.py b4a106a3-4ff3-45dd-917c-9b3e6f0501a4
"""

import sys
import json
import argparse
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from chorus_engine.db.database import SessionLocal
from chorus_engine.models.conversation import Conversation
from chorus_engine.config.loader import ConfigLoader
from chorus_engine.llm.client import create_llm_client
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.services.conversation_analysis_service import ConversationAnalysisService
from chorus_engine.services.json_extraction import extract_json_block


async def run_analysis(
    conversation_id: str,
    output_path: str | None = None,
    include_prompts: bool = False,
    include_raw: bool = False,
    archivist_max_tokens: int | None = None
) -> Path | None:
    db = SessionLocal()
    try:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        if not conversation:
            print(f"Conversation {conversation_id} not found")
            return None

        config_loader = ConfigLoader(project_root)
        system_config = config_loader.load_system_config()
        character = config_loader.load_character(conversation.character_id)

        llm_client = create_llm_client(system_config.llm)
        embedding_service = EmbeddingService()
        vector_store = VectorStore(Path("data/vector_store"))

        analysis_service = ConversationAnalysisService(
            db=db,
            llm_client=llm_client,
            vector_store=vector_store,
            embedding_service=embedding_service,
            temperature=0.1,
            summary_vector_store=None,
            llm_usage_lock=asyncio.Lock(),
            archivist_model=system_config.llm.archivist_model,
            analysis_max_tokens_summary=system_config.llm.analysis_max_tokens_summary,
            analysis_max_tokens_memories=system_config.llm.analysis_max_tokens_memories,
            analysis_min_tokens_summary=system_config.llm.analysis_min_tokens_summary,
            analysis_min_tokens_memories=system_config.llm.analysis_min_tokens_memories
        )

        # Build conversation text and token count
        messages = analysis_service._get_all_messages(conversation_id)
        if not messages:
            print("No messages found for this conversation")
            return None
        token_count = analysis_service._count_tokens(messages)
        conversation_text = analysis_service._format_conversation(messages)

        model_primary, model_fallback = analysis_service._select_models(character)
        summary_model = system_config.llm.archivist_model or model_primary or model_fallback

        async def run_with_retries(
            task_label: str,
            system_prompt: str,
            user_prompt: str,
            parser,
            max_tokens: int,
            temperature: float,
        ) -> tuple[dict | list | None, str, list[dict]]:
            attempts: list[dict] = []
            last_response = ""
            parsed = None

            for attempt in range(1, 3):
                response_obj = await analysis_service.llm_client.generate(
                    system_prompt=system_prompt,
                    prompt=user_prompt,
                    model=summary_model,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                response_text = response_obj.content or ""
                last_response = response_text
                _, parse_mode = extract_json_block(
                    response_text,
                    "object" if task_label == "summary" else "array"
                )
                parsed = parser(response_text)
                success = parsed is not None
                attempts.append({
                    "attempt": attempt,
                    "model": summary_model,
                    "used_model": response_obj.model,
                    "finish_reason": response_obj.finish_reason,
                    "response_empty": response_text.strip() == "",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system_prompt_length": len(system_prompt or ""),
                    "user_prompt_length": len(user_prompt or ""),
                    "usage": response_obj.usage,
                    "parse_mode": parse_mode,
                    "success": success,
                    "response_preview": response_text[:400]
                })
                if success:
                    break

            return parsed, last_response, attempts

        # Summary step
        summary_system_prompt, summary_user_prompt = analysis_service._build_summary_prompt(
            conversation_text=conversation_text,
            token_count=token_count
        )
        summary_data, summary_response, summary_attempts = await run_with_retries(
            task_label="summary",
            system_prompt=summary_system_prompt,
            user_prompt=summary_user_prompt,
            parser=analysis_service._parse_summary_response,
            max_tokens=analysis_service.analysis_max_tokens_summary,
            temperature=0.0
        )

        if not summary_data:
            analysis = {
                "summary": "",
                "participants": [],
                "emotional_arc": "",
                "open_questions": [],
            }
            data = {
                "conversation": {
                    "id": conversation.id,
                    "character_id": conversation.character_id,
                    "title": conversation.title,
                    "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
                    "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else None,
                    "last_analyzed_at": conversation.last_analyzed_at.isoformat() if conversation.last_analyzed_at else None,
                },
                "analysis": analysis,
                "memories": [],
                "error": "Summary parsing failed"
            }

            if include_prompts:
                data["prompts"] = {
                    "summary": {
                        "system": summary_system_prompt,
                        "user": summary_user_prompt
                    }
                }

            if include_raw:
                data["raw_responses"] = {
                    "summary": summary_response
                }

            data["parse_diagnostics"] = {
                "summary": summary_attempts
            }

            json_output = json.dumps(data, indent=2, ensure_ascii=False)

            if output_path:
                output_file = Path(output_path)
            else:
                output_dir = Path("data/exports")
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = output_dir / f"conversation_analysis_preview_{conversation_id[:8]}_{timestamp}.json"

            output_file.write_text(json_output, encoding="utf-8")
            print("Summary parsing failed (output written with diagnostics)")
            print(f"✓ Saved to: {output_file}")
            return output_file

        # Archivist step
        archivist_system_prompt, archivist_user_prompt = analysis_service._build_archivist_prompt(
            conversation_text=conversation_text,
            token_count=token_count
        )
        memories, archivist_response, archivist_attempts = await run_with_retries(
            task_label="archivist",
            system_prompt=archivist_system_prompt,
            user_prompt=archivist_user_prompt,
            parser=lambda response: analysis_service._parse_archivist_response(response, character),
            max_tokens=archivist_max_tokens or analysis_service.analysis_max_tokens_memories,
            temperature=0.0
        )

        analysis = {
            "summary": summary_data.get("summary", ""),
            "participants": summary_data.get("participants", []),
            "emotional_arc": summary_data.get("emotional_arc", ""),
            "open_questions": summary_data.get("open_questions", []),
        }

        data = {
            "conversation": {
                "id": conversation.id,
                "character_id": conversation.character_id,
                "title": conversation.title,
                "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
                "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else None,
                "last_analyzed_at": conversation.last_analyzed_at.isoformat() if conversation.last_analyzed_at else None,
            },
            "analysis": analysis,
            "memories": [
                {
                    "type": m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type),
                    "content": m.content,
                    "confidence": m.confidence,
                    "durability": m.durability,
                    "pattern_eligible": m.pattern_eligible,
                    "reasoning": m.reasoning,
                    "emotional_weight": m.emotional_weight,
                    "participants": m.participants,
                    "key_moments": m.key_moments,
                }
                for m in (memories or [])
            ]
        }

        if include_prompts:
            data["prompts"] = {
                "summary": {
                    "system": summary_system_prompt,
                    "user": summary_user_prompt
                },
                "archivist": {
                    "system": archivist_system_prompt,
                    "user": archivist_user_prompt
                }
            }

        if include_raw:
            data["raw_responses"] = {
                "summary": summary_response,
                "archivist": archivist_response
            }

        data["parse_diagnostics"] = {
            "summary": summary_attempts,
            "archivist": archivist_attempts
        }

        json_output = json.dumps(data, indent=2, ensure_ascii=False)

        if output_path:
            output_file = Path(output_path)
        else:
            output_dir = Path("data/exports")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"conversation_analysis_preview_{conversation_id[:8]}_{timestamp}.json"

        output_file.write_text(json_output, encoding="utf-8")
        print(f"✓ Saved to: {output_file}")
        return output_file
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run conversation analysis without persisting results."
    )
    parser.add_argument("conversation_id", type=str, help="Conversation ID to analyze")
    parser.add_argument(
        "--output",
        type=str,
        help="Optional output file path (defaults to data/exports/...)"
    )
    parser.add_argument(
        "--include-prompts",
        action="store_true",
        help="Include full system and user prompts in the output"
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include raw LLM responses in the output"
    )
    parser.add_argument(
        "--archivist-max-tokens",
        type=int,
        help="Override max tokens for the archivist step only"
    )

    args = parser.parse_args()

    asyncio.run(
        run_analysis(
            args.conversation_id,
            args.output,
            include_prompts=args.include_prompts,
            include_raw=args.include_raw,
            archivist_max_tokens=args.archivist_max_tokens
        )
    )


if __name__ == "__main__":
    main()
