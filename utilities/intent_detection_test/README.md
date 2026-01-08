# Intent Detection Test Utility

This utility provides an interactive CLI for testing the semantic intent detection system.

## Usage

```bash
cd utilities/intent_detection_test
python test_intent.py
```

## Features

- Interactive testing: Type phrases and immediately see detection results
- Debug output: Shows confidence scores for all intents
- Logging: All tests are logged to `logs/intent_test_TIMESTAMP.log`
- Statistics: Session summary on exit (CTRL+C)

## Example Session

```
Enter phrase: send me a photo of a sunset
--------------------------------------------------------------------
send_image: 0.872
send_video: 0.523
set_reminder: 0.312

✓ Detected 1 intent(s):
  • send_image: 0.8720

Enter phrase: remind me tomorrow to call mom
--------------------------------------------------------------------
send_image: 0.298
send_video: 0.245
set_reminder: 0.891

✓ Detected 1 intent(s):
  • set_reminder: 0.8910
```

## Testing Guidelines

**Test diverse phrasings:**
- Formal: "Please generate an image of a landscape"
- Casual: "show me a pic"
- Indirect: "could you create a photo?"
- Multi-intent: "send me a video and remind me in an hour"

**Look for:**
- False positives (wrong intent detected)
- False negatives (intent missed)
- Ambiguous cases (two intents too close)
- Threshold issues (just below/above cutoff)

## Log Files

Logs are saved to `logs/intent_test_YYYYMMDD_HHMMSS.log` with:
- Timestamp for each test
- Exact phrase tested
- All detected intents with confidence scores
- Detection failures (ambiguous/no intent)
