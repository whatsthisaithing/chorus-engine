"""
Intent Detection Test Utility

Interactive CLI for testing semantic intent detection with user phrases.
Reports results with confidence scores and logs all detections to file.

Usage:
    python test_intent.py
    
    Then type test phrases and press Enter. Press CTRL+C to exit.
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chorus_engine.services.semantic_intent_detection import get_intent_detector
from sentence_transformers import SentenceTransformer


def main():
    """Run interactive intent detection testing loop."""
    
    print("=" * 70)
    print("INTENT DETECTION TEST UTILITY")
    print("=" * 70)
    print()
    
    # Load embedding model (same as used in memory system)
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Initialize detector
    print("Initializing semantic intent detector...")
    try:
        detector = get_intent_detector(model)
        print("✓ Detector initialized")
    except Exception as e:
        print(f"✗ Failed to initialize detector: {e}")
        return
    
    # Create log file
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"intent_test_{timestamp}.log"
    
    print(f"✓ Logging to: {log_file}")
    print()
    print("=" * 70)
    print()
    print("Type test phrases below. Press CTRL+C to exit.")
    print()
    
    # Interactive loop
    test_count = 0
    
    try:
        while True:
            # Get user input
            try:
                phrase = input("Enter phrase: ").strip()
            except EOFError:
                break
            
            if not phrase:
                continue
            
            test_count += 1
            print()
            print("-" * 70)
            print(f"Test #{test_count}: \"{phrase}\"")
            print("-" * 70)
            
            # Run detection with debug output
            intents = detector.detect(phrase, enable_multi_intent=True, debug=True)
            
            # Log to file
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 70}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Test #{test_count}\n")
                f.write(f"Phrase: {phrase}\n")
                f.write(f"{'-' * 70}\n")
                
                if intents:
                    f.write(f"Detected {len(intents)} intent(s):\n")
                    for intent in intents:
                        f.write(f"  - {intent.name}: {intent.confidence:.4f}\n")
                else:
                    f.write("No intent detected (ambiguous or below threshold)\n")
            
            # Print results
            print()
            if intents:
                print(f"✓ Detected {len(intents)} intent(s):")
                for intent in intents:
                    print(f"  • {intent.name}: {intent.confidence:.4f}")
            else:
                print("✗ No intent detected (ambiguous or below threshold)")
            
            print()
    
    except KeyboardInterrupt:
        print("\n")
        print("=" * 70)
        print("EXITING")
        print("=" * 70)
    
    # Print statistics
    stats = detector.get_stats()
    print()
    print("Session Statistics:")
    print(f"  Total tests: {test_count}")
    print(f"  Successful detections: {stats['successful_detections']}")
    print(f"  Multi-intent detections: {stats['multi_intent_detections']}")
    print(f"  Ambiguous: {stats['ambiguous_detections']}")
    print(f"  No intent: {stats['no_intent_detected']}")
    print()
    print(f"Log saved to: {log_file}")
    print()


if __name__ == "__main__":
    main()
