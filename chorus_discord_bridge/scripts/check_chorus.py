"""Quick health check for Chorus Engine - doesn't require full config."""

import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Get API URL from environment
api_url = os.getenv('CHORUS_API_URL', 'http://localhost:8080')

try:
    # Try to connect to /characters endpoint
    response = requests.get(f"{api_url}/characters", timeout=5)
    
    if response.status_code == 200:
        print(f"[OK] Connected to Chorus Engine at {api_url}")
        sys.exit(0)
    else:
        print(f"[ERROR] Chorus Engine returned status {response.status_code}")
        sys.exit(1)
        
except requests.exceptions.ConnectionError:
    print(f"[ERROR] Cannot connect to Chorus Engine at {api_url}")
    print("Make sure Chorus Engine is running!")
    sys.exit(1)
    
except Exception as e:
    print(f"[ERROR] Health check failed: {e}")
    sys.exit(1)
