"""Main entry point for Chorus Engine."""

import logging
import sys
import io
from pathlib import Path

import uvicorn


def setup_logging(debug: bool = False):
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    
    # Force UTF-8 encoding for stdout/stderr on Windows
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    # Setup handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    file_handler = None
    log_file = None
    
    # Add file handler if debug mode is enabled
    if debug:
        from datetime import datetime
        log_dir = Path("data/debug_logs/server")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"server_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        handlers.append(file_handler)
    
    # Set root logger to INFO to avoid verbose library logs
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Force reconfiguration even if already configured
    )
    
    # Only set DEBUG for our app loggers, not third-party libraries
    chorus_logger = logging.getLogger('chorus_engine')
    chorus_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Silence noisy third-party loggers
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    
    # Suppress noisy heartbeat status access logs
    class HeartbeatStatusFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage()
            return "/heartbeat/status" not in message

    logging.getLogger("uvicorn.access").addFilter(HeartbeatStatusFilter())

    # Log startup info (only __main__ logger needs file handler in parent process)
    startup_logger = logging.getLogger(__name__)
    if debug:
        startup_logger.info(f"[STARTUP] Server log file: {log_file}")
    startup_logger.info(f"[STARTUP] Logging configured: level={level}, chorus_engine logger level={chorus_logger.level}")
    
    return file_handler, log_file


def main():
    """Run the FastAPI server."""
    # Load system config to get debug flag
    from chorus_engine.config import ConfigLoader
    try:
        loader = ConfigLoader()
        system_config = loader.load_system_config()
        debug_mode = system_config.debug
    except Exception as e:
        # Use basic logging since logger isn't configured yet
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.getLogger(__name__).warning(f"[STARTUP WARNING] Could not load system config: {e}, using debug=False")
        debug_mode = False
    
    # Setup logging with debug flag from config
    file_handler, log_file = setup_logging(debug=debug_mode)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Chorus Engine server (debug mode: {debug_mode})...")
    logger.info(f"Server will listen on {system_config.api_host}:{system_config.api_port}")
    
    # Run server
    uvicorn.run(
        "chorus_engine.api.app:app",
        host=system_config.api_host,
        port=system_config.api_port,
        reload=False,  # Disable hot reload - use manual restart for changes
        log_level="info",  # Use info level for uvicorn itself
        log_config=None,  # Don't modify uvicorn's log config - use our basicConfig
    )


if __name__ == "__main__":
    main()
