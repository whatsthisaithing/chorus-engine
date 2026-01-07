"""Fast keyword-based intent detection for synchronous actions.

Phase 7.5: Hybrid system that uses keywords for immediate intents
(image/video generation) and async background worker for memory extraction.
"""

import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class KeywordIntents:
    """Result of keyword-based intent detection."""
    generate_image: bool = False
    generate_video: bool = False
    record_memory: bool = False  # Explicit memory recording request
    query_ambient: bool = False
    
    image_confidence: float = 0.0
    video_confidence: float = 0.0
    memory_confidence: float = 0.0
    ambient_confidence: float = 0.0


class KeywordIntentDetector:
    """Fast keyword-based intent detection for blocking actions.
    
    This replaces the LLM-based intent detection for synchronous actions
    to avoid VRAM overhead and reduce latency. Memory extraction is handled
    by a separate async background worker.
    """
    
    # Image generation keywords - must be combined with request indicators
    IMAGE_KEYWORDS = [
        'photo', 'picture', 'image', 'selfie', 'snapshot', 
        'drawing', 'sketch', 'illustration', 'portrait'
    ]
    
    # Request indicators - verbs/phrases that indicate user wants something
    REQUEST_INDICATORS = [
        'send', 'show', 'give', 'create', 'generate', 'make', 'draw',
        'take', 'share', 'post', 'upload', 'shoot',
        'can you', 'could you', 'would you', 'will you',
        'please', "i'd like", 'i want', 'i need',
        'let me see', 'show me'
    ]
    
    # Negative indicators - these suggest NOT a request
    NEGATIVE_INDICATORS = [
        'beautiful', 'nice', 'lovely', 'great', 'amazing', 'wonderful',
        'saw', 'seen', 'looked at', 'checked out', 'found',
        'thanks for', 'thank you for'
    ]
    
    # Video generation keywords
    VIDEO_KEYWORDS = [
        'video', 'animation', 'movie', 'clip', 'footage', 'recording'
    ]
    
    # Explicit memory recording keywords
    MEMORY_KEYWORDS = [
        'remember', "don't forget", "dont forget", 'keep in mind',
        'make a note', 'write down', 'save that', 'record that'
    ]
    
    # Ambient activity query keywords
    AMBIENT_KEYWORDS = [
        'what are you doing', 'what have you been',
        'what are you up to', 'been up to',
        'recent activities', 'lately been'
    ]
    
    def detect(self, message: str) -> KeywordIntents:
        """Detect intents from user message using keywords.
        
        Args:
            message: User's message text
            
        Returns:
            KeywordIntents with detected intents and confidence scores
        """
        import logging
        logger = logging.getLogger(__name__)
        
        message_lower = message.lower()
        intents = KeywordIntents()
        
        # Detect image generation intent
        intents.generate_image, intents.image_confidence = self._detect_image_intent(message_lower)
        
        # Log detection result for debugging
        if intents.generate_image:
            logger.info(f"[KEYWORD] Image intent detected: '{message[:50]}...' (confidence: {intents.image_confidence})")
        
        # Detect video generation intent
        intents.generate_video, intents.video_confidence = self._detect_video_intent(message_lower)
        
        # Detect explicit memory recording
        intents.record_memory, intents.memory_confidence = self._detect_memory_intent(message_lower)
        
        # Detect ambient activity query
        intents.query_ambient, intents.ambient_confidence = self._detect_ambient_intent(message_lower)
        
        return intents
    
    def _detect_image_intent(self, message_lower: str) -> tuple[bool, float]:
        """Detect if user is requesting an image.
        
        Uses smart detection to avoid false positives:
        - Splits message into sentences to handle multi-sentence messages
        - Each sentence checked independently for image intent
        - Skips sentences with negative indicators (compliments about past images)
        - Must have image keyword + request indicator in same sentence
        
        Examples:
            "Can you send me a photo?" -> True (request + keyword)
            "Show me a picture" -> True (request + keyword)
            "Beautiful photo!" -> False (negative indicator, no request)
            "Beautiful. Now send me a photo" -> True (different sentences)
            "I saw a nice picture" -> False (past tense, not request)
        """
        # Split into sentences (periods, exclamation marks, question marks)
        sentences = re.split(r'[.!?]+', message_lower)
        
        # Check each sentence independently
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if this sentence has negative indicators (compliments, past events)
            has_negative = any(neg in sentence for neg in self.NEGATIVE_INDICATORS)
            if has_negative:
                continue  # Skip this sentence, check next one
            
            # Must have image keyword in this sentence
            has_image_keyword = any(kw in sentence for kw in self.IMAGE_KEYWORDS)
            if not has_image_keyword:
                continue
            
            # Must have request indicator in this sentence
            has_request = any(req in sentence for req in self.REQUEST_INDICATORS)
            if not has_request:
                continue
            
            # Found a sentence with image keyword + request, no negatives
            return True, 0.95
        
        # No valid sentence found
        return False, 0.0
    
    def _detect_video_intent(self, message_lower: str) -> tuple[bool, float]:
        """Detect if user is requesting a video.
        
        Uses sentence-based detection like image intent:
        - Splits message into sentences
        - Each sentence checked independently
        - Skips sentences with negative indicators
        - Must have video keyword + request indicator in same sentence
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', message_lower)
        
        # Check each sentence independently
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check for negative indicators in this sentence
            has_negative = any(neg in sentence for neg in self.NEGATIVE_INDICATORS)
            if has_negative:
                continue
            
            # Must have video keyword in this sentence
            has_video_keyword = any(kw in sentence for kw in self.VIDEO_KEYWORDS)
            if not has_video_keyword:
                continue
            
            # Must have request indicator in this sentence
            has_request = any(req in sentence for req in self.REQUEST_INDICATORS)
            if not has_request:
                continue
            
            # Found valid sentence
            return True, 0.95
        
        return False, 0.0
    
    def _detect_memory_intent(self, message_lower: str) -> tuple[bool, float]:
        """Detect if user explicitly wants to record a memory."""
        has_memory_keyword = any(kw in message_lower for kw in self.MEMORY_KEYWORDS)
        if has_memory_keyword:
            return True, 0.9
        return False, 0.0
    
    def _detect_ambient_intent(self, message_lower: str) -> tuple[bool, float]:
        """Detect if user is asking about character's background activities."""
        has_ambient_keyword = any(kw in message_lower for kw in self.AMBIENT_KEYWORDS)
        if has_ambient_keyword:
            return True, 0.85
        return False, 0.0
