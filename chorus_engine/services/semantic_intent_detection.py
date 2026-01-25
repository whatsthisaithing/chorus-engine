"""
Semantic Intent Detection System

Uses embedding-based semantic similarity to detect user intents from natural language.
Replaces fragile keyword-based regex with robust cosine similarity matching.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Intent:
    """Detected intent with confidence score."""
    name: str
    confidence: float
    raw_message: str


# Intent Prototypes: Carefully crafted phrases representing each intent
INTENT_PROTOTYPES = {
    "send_image": [
        "send me a photo",
        "show me an image",
        "generate a picture",
        "create an image of something",
        "take a photo and send it",
        "can you make me a picture",
        # Detailed/contextual prototypes (help match requests with descriptive details)
        "send me a photo of you doing something specific",
        "could you take a selfie of yourself with a particular expression",
        "show me an image of you holding something",
        "generate a picture of you making a gesture"
    ],
    "send_video": [
        "send me a video",
        "show me a video clip",
        "create a video of something",
        "generate a video showing",
        "make me a video"
    ],
    "set_reminder": [
        "remind me to do something",
        "set a reminder for later",
        "remind me in the future",
        "don't let me forget to do something",
        "create a reminder about something",
        "remind me that I need to do something",
        "remind me to call someone",
        "remind me to send something",
        "set a reminder to contact someone",
        "remind me tomorrow to do something",
        "remind me in a few days to call someone",
        "remind me next week that I need to do something",
        "remind me to schedule an appointment",
        "remind me later to contact someone",
        # Content-heavy prototypes with realistic tasks
        "remind me to call the doctor next week",
        "remind me in three weeks to call the dog groomer",
        "remind me that I need to schedule an appointment",
        "remind me tomorrow to call a business",
        "remind me to call someone in a few weeks",
        "set a reminder to contact the dentist later"
    ]
}

# Confidence Thresholds: Higher for critical actions, lower for safe/reversible ones
# Adjusted for hybrid sentence-level detection (slightly more permissive since
# sentence-level analysis provides better specificity but may score lower)
THRESHOLDS = {
    "set_reminder": 0.50,  # Confirmable action with user review
    "send_image": 0.45,    # Lowered to accommodate sentence-level detection
    "send_video": 0.45,    # Lowered to accommodate sentence-level detection
    "default": 0.50
}

# Global minimum threshold - reject anything below this regardless of intent
# Lowered from 0.45 to 0.42 to accommodate sentence-level detection in hybrid mode
# (sentence-level reduces dilution but may score slightly lower than full message for short inputs)
GLOBAL_MIN_THRESHOLD = 0.42

# Ambiguity margin - if two intents are this close, consider it ambiguous
AMBIGUITY_MARGIN = 0.08

# Exclusion Groups: Mutually exclusive intents (only highest in group triggers)
EXCLUSION_GROUPS = {
    "media_generation": ["send_image", "send_video"]
}


class SemanticIntentDetector:
    """
    Detects user intents using semantic similarity with pre-defined prototypes.
    
    Architecture:
    1. Pre-compute embeddings for all intent prototypes at initialization
    2. For each message, compute embedding and compare to all prototypes
    3. Calculate max cosine similarity for each intent
    4. Apply per-intent thresholds
    5. Check ambiguity margin between top candidates
    6. Return ranked list of detected intents
    """
    
    def __init__(self, embedding_model: SentenceTransformer):
        """
        Initialize detector with pre-computed prototype embeddings.
        
        Args:
            embedding_model: Pre-loaded SentenceTransformer model
                           (should be shared with memory system)
        """
        self.model = embedding_model
        self.prototype_embeddings = self._embed_prototypes()
        
        # Statistics for debugging and monitoring
        self.stats = {
            "total_detections": 0,
            "successful_detections": 0,
            "ambiguous_detections": 0,
            "no_intent_detected": 0,
            "multi_intent_detections": 0
        }
    
    def _embed_prototypes(self) -> Dict[str, List[np.ndarray]]:
        """
        Pre-compute embeddings for all intent prototypes.
        
        Returns:
            Dict mapping intent names to lists of prototype embeddings
        """
        embeddings = {}
        
        for intent_name, prototypes in INTENT_PROTOTYPES.items():
            # Encode all prototypes for this intent
            prototype_vecs = self.model.encode(prototypes, convert_to_numpy=True)
            embeddings[intent_name] = [vec for vec in prototype_vecs]
        
        return embeddings
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Similarity score between 0 and 1
        """
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    @staticmethod
    def _split_sentences(message: str) -> List[str]:
        """
        Split message into sentences for per-sentence analysis.
        
        Args:
            message: Full message text
            
        Returns:
            List of sentence strings (stripped, non-empty)
        """
        import re
        
        # Split on sentence-ending punctuation (., !, ?)
        # Keep the punctuation with the sentence
        sentences = re.split(r'(?<=[.!?])\s+', message)
        
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Filter out very short fragments (< 4 words)
        sentences = [s for s in sentences if len(s.split()) >= 4]
        
        return sentences
    
    @staticmethod
    def _merge_intents(intents: List[Intent]) -> List[Intent]:
        """
        Merge duplicate intents, keeping the highest confidence for each type.
        
        Args:
            intents: List of Intent objects (may contain duplicates)
            
        Returns:
            Deduplicated list with max confidence per intent type
        """
        if not intents:
            return []
        
        # Group by intent name, keep max confidence
        intent_map = {}
        
        for intent in intents:
            if intent.name not in intent_map:
                intent_map[intent.name] = intent
            else:
                # Keep the one with higher confidence
                if intent.confidence > intent_map[intent.name].confidence:
                    intent_map[intent.name] = intent
        
        # Return as sorted list (by confidence)
        merged = list(intent_map.values())
        merged.sort(key=lambda i: i.confidence, reverse=True)
        
        return merged
    
    @staticmethod
    def _anchor_bonus(message: str, intent_name: str) -> float:
        """
        Apply lexical anchor bonus/penalty for specific intents.
        Hybrid approach: adds small boost if intent-specific keywords present,
        or applies penalty if context indicates false positive.
        
        Args:
            message: User's message text (lowercase)
            intent_name: Name of the intent being scored
            
        Returns:
            Bonus/penalty to add to similarity score (-0.15 to +0.05)
        """
        message_lower = message.lower()
        
        if intent_name == "set_reminder":
            # Boost if explicit reminder language present
            anchor_words = ["remind", "reminder", "don't let me forget", "don't forget"]
            if any(word in message_lower for word in anchor_words):
                return 0.05
        
        if intent_name == "send_image":
            # Penalty if talking about past images or hypothetical/future scenarios
            past_indicators = [
                "you showed", "you sent", "you created", "you made",
                "you needed to send", "you generated", "you took",
                "the image you", "the photo you", "the picture you"
            ]
            future_hypothetical = [
                "when you can", "you'll be able", "you will be able",
                "i'll let you know", "being able to", "talking about",
                "i send to you", "i send you", "when i send"
            ]
            
            # Check for past tense indicators
            if any(phrase in message_lower for phrase in past_indicators):
                return -0.15
            
            # Check for future/hypothetical context
            if any(phrase in message_lower for phrase in future_hypothetical):
                return -0.15
        
        return 0.0
    
    def detect(
        self,
        message: str,
        enable_multi_intent: bool = True,
        debug: bool = False
    ) -> List[Intent]:
        """
        Detect intents in a user message using hybrid approach.
        
        Strategy:
        1. Run detection on full message (captures short messages + overall context)
        2. For longer messages (>20 words), also run on individual sentences
        3. Merge results, taking max confidence for each intent type
        
        This handles both:
        - Short, focused messages: "send me a photo" (full message works)
        - Long messages with embedded intents: "I'm planning... Could you send me a selfie..." (sentence-level catches it)
        
        Args:
            message: User's message text
            enable_multi_intent: If True, can detect multiple intents
                               If False, returns only highest-confidence intent
            debug: If True, print detailed scoring information
            
        Returns:
            List of detected Intent objects, sorted by confidence (highest first)
            Empty list if no intent detected or ambiguous
        """
        self.stats["total_detections"] += 1
        
        # Always detect on full message
        full_message_intents = self._detect_single(
            message,
            enable_multi_intent=enable_multi_intent,
            debug=debug,
            context="full message"
        )
        
        # For longer messages, also analyze individual sentences
        word_count = len(message.split())
        
        if word_count > 20:  # Threshold for sentence-level analysis
            sentences = self._split_sentences(message)
            
            if debug and sentences:
                print(f"\n[HYBRID] Message has {word_count} words, analyzing {len(sentences)} sentences separately")
            
            sentence_intents = []
            
            for i, sentence in enumerate(sentences):
                if debug:
                    print(f"\n[HYBRID] Sentence {i+1}: '{sentence[:60]}...'")
                
                intents = self._detect_single(
                    sentence,
                    enable_multi_intent=enable_multi_intent,
                    debug=debug,
                    context=f"sentence {i+1}"
                )
                sentence_intents.extend(intents)
            
            # Merge full message + sentence-level detections
            all_intents = full_message_intents + sentence_intents
            
            if debug and all_intents:
                print(f"\n[HYBRID] Before merge: {len(all_intents)} intents")
                print(f"[HYBRID] Full message: {[i.name for i in full_message_intents]}")
                print(f"[HYBRID] Sentences: {[i.name for i in sentence_intents]}")
            
            merged_intents = self._merge_intents(all_intents)
            
            if debug and merged_intents:
                print(f"[HYBRID] After merge: {[(i.name, f'{i.confidence:.3f}') for i in merged_intents]}")
            
            # Update statistics
            self._update_stats(merged_intents)
            
            return merged_intents
        else:
            # Short message - use full message detection only
            # Update statistics
            self._update_stats(full_message_intents)
            
            return full_message_intents
    
    def _update_stats(self, intents: List[Intent]) -> None:
        """Update detection statistics based on results."""
        if len(intents) == 0:
            self.stats["no_intent_detected"] += 1
        elif len(intents) == 1:
            self.stats["successful_detections"] += 1
        else:
            self.stats["multi_intent_detections"] += 1
            self.stats["successful_detections"] += 1
    
    def _detect_single(
        self,
        message: str,
        enable_multi_intent: bool = True,
        debug: bool = False,
        context: str = ""
    ) -> List[Intent]:
        """
        Detect intents in a single text fragment (sentence or full message).
        
        This is the core detection logic, called by detect() for both
        full messages and individual sentences.
        
        Args:
            message: Text to analyze
            enable_multi_intent: If True, can detect multiple intents
            debug: If True, print detailed scoring information
            context: Description for debug output (e.g., "sentence 1")
            
        Returns:
            List of detected Intent objects
        """
    def _detect_single(
        self,
        message: str,
        enable_multi_intent: bool = True,
        debug: bool = False,
        context: str = ""
    ) -> List[Intent]:
        """
        Detect intents in a single text fragment (sentence or full message).
        
        This is the core detection logic, called by detect() for both
        full messages and individual sentences.
        
        Args:
            message: Text to analyze
            enable_multi_intent: If True, can detect multiple intents
            debug: If True, print detailed scoring information
            context: Description for debug output (e.g., "sentence 1")
            
        Returns:
            List of detected Intent objects
        """
        if debug and context:
            print(f"\n[DETECT {context.upper()}]")
        
        # Embed the incoming message
        message_embedding = self.model.encode(message, convert_to_numpy=True)
        
        # Calculate max similarity for each intent
        intent_scores = {}
        
        for intent_name, prototype_embeddings in self.prototype_embeddings.items():
            # Compare message to all prototypes for this intent
            similarities = [
                self.cosine_similarity(message_embedding, proto_emb)
                for proto_emb in prototype_embeddings
            ]
            
            # Take the maximum similarity (best matching prototype)
            max_similarity = max(similarities)
            
            # Apply lexical anchor bonus (hybrid approach)
            bonus = self._anchor_bonus(message, intent_name)
            max_similarity += bonus
            
            intent_scores[intent_name] = max_similarity
            
            if debug:
                bonus_str = f" (+{bonus:.2f} bonus)" if bonus > 0 else ""
                print(f"  {intent_name}: {max_similarity:.3f}{bonus_str}")
        
        # Sort by confidence (highest first)
        sorted_intents = sorted(
            intent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if debug:
            print(f"  Sorted: {[(name, f'{score:.3f}') for name, score in sorted_intents[:3]]}")
        
        # Check global minimum threshold
        if sorted_intents[0][1] < GLOBAL_MIN_THRESHOLD:
            if debug:
                print(f"  Below global minimum ({GLOBAL_MIN_THRESHOLD})")
            return []
        
        # Check for ambiguity between top two candidates
        if len(sorted_intents) >= 2:
            best_score = sorted_intents[0][1]
            second_score = sorted_intents[1][1]
            
            if (best_score - second_score) < AMBIGUITY_MARGIN:
                if debug:
                    print(f"  Ambiguous: {best_score:.3f} vs {second_score:.3f} (margin < {AMBIGUITY_MARGIN})")
                return []
        
        # Collect intents above their thresholds
        detected_intents = []
        
        for intent_name, confidence in sorted_intents:
            threshold = THRESHOLDS.get(intent_name, THRESHOLDS["default"])
            
            if confidence >= threshold:
                detected_intents.append(Intent(
                    name=intent_name,
                    confidence=confidence,
                    raw_message=message
                ))
                
                if not enable_multi_intent:
                    # Single intent mode - return first match
                    break
        
        # Apply exclusion group filtering
        if len(detected_intents) > 1 and enable_multi_intent:
            detected_intents = self._apply_exclusion_groups(detected_intents)
        
        if debug and detected_intents:
            print(f"  → Detected: {[(i.name, f'{i.confidence:.3f}') for i in detected_intents]}")
        
        return detected_intents
    
    def _apply_exclusion_groups(self, intents: List[Intent]) -> List[Intent]:
        """
        Filter intents to remove spurious multi-intent detections.
        
        Within each exclusion group, only the highest-confidence intent is kept.
        Intents not in any group are always kept.
        
        Args:
            intents: List of detected intents
            
        Returns:
            Filtered list with exclusion groups applied
        """
        filtered = []
        processed_intents = set()
        
        # Process each exclusion group
        for group_name, group_intents in EXCLUSION_GROUPS.items():
            # Find all detected intents in this group
            group_matches = [
                intent for intent in intents
                if intent.name in group_intents
            ]
            
            if group_matches:
                # Keep only the highest-confidence intent from this group
                best = max(group_matches, key=lambda i: i.confidence)
                filtered.append(best)
                
                # Mark all intents in this group as processed
                for intent in group_matches:
                    processed_intents.add(intent.name)
        
        # Add intents not in any exclusion group
        for intent in intents:
            if intent.name not in processed_intents:
                filtered.append(intent)
        
        # Restore original sort order (by confidence)
        filtered.sort(key=lambda i: i.confidence, reverse=True)
        
        return filtered
    
    def get_stats(self) -> Dict:
        """Return detection statistics for monitoring and debugging."""
        return self.stats.copy()


# Singleton instance
_detector_instance: Optional[SemanticIntentDetector] = None


def get_intent_detector(embedding_model: Optional[SentenceTransformer] = None) -> SemanticIntentDetector:
    """
    Get or create the global intent detector instance.
    
    Args:
        embedding_model: SentenceTransformer model (optional - will use shared EmbeddingService if not provided)
        
    Returns:
        Global SemanticIntentDetector instance
    """
    global _detector_instance
    
    if _detector_instance is None:
        if embedding_model is None:
            # Use shared embedding service
            from chorus_engine.services.embedding_service import EmbeddingService
            service = EmbeddingService()
            embedding_model = service.model
        
        _detector_instance = SemanticIntentDetector(embedding_model)
    
    return _detector_instance
