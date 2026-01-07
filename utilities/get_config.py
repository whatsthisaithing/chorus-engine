"""Utility to extract configuration values for use in shell scripts."""

import sys
from pathlib import Path

# Add parent directory to path so we can import chorus_engine
sys.path.insert(0, str(Path(__file__).parent.parent))

from chorus_engine.config import ConfigLoader


def main():
    """Print configuration values in a format shell scripts can parse."""
    if len(sys.argv) < 2:
        print("Usage: get_config.py <key>", file=sys.stderr)
        print("Available keys: api_host, api_port, api_url", file=sys.stderr)
        sys.exit(1)
    
    key = sys.argv[1].lower()
    
    try:
        loader = ConfigLoader()
        config = loader.load_system_config()
        
        if key == "api_host":
            print(config.api_host)
        elif key == "api_port":
            print(config.api_port)
        elif key == "api_url":
            print(f"http://{config.api_host}:{config.api_port}")
        else:
            print(f"Unknown key: {key}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        # Fallback to defaults on error
        if key == "api_host":
            print("localhost")
        elif key == "api_port":
            print("8080")
        elif key == "api_url":
            print("http://localhost:8080")
        else:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
