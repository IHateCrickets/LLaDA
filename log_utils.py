import os
import datetime
import sys
from pathlib import Path

# Import debug settings
# Assuming debug_utils.py is in the same directory or Python path
try:
    from debug_utils import (DEBUG_MODE, VERBOSE, SUPPRESS_WARNINGS,
                             QUIET_MODE, LOGFILE)
except ImportError:
    print("Warning: debug_utils.py not found. Using default logging settings.")
    # Define defaults if import fails
    DEBUG_MODE = 0
    VERBOSE = 0
    SUPPRESS_WARNINGS = 0
    QUIET_MODE = 1
    LOGFILE = "logs/app.log"

def log(level="INFO", message=None):
    """
    Log messages with timestamps and color-coded output based on severity level.

    Args:
        level: Severity level (ERROR, WARNING, SUCCESS, INFO, DEBUG)
        message: The message to log
    """
    # Handle the case where only message is provided
    if message is None:
        message = level
        level = "INFO"

    # Ensure level is uppercase for dictionary lookup and consistency
    level = level.upper()

    # Only log DEBUG messages if DEBUG_MODE is enabled
    if level == "DEBUG" and not DEBUG_MODE:
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add color based on level
    colors = {
        "ERROR": "\033[1;31m",    # Red
        "WARNING": "\033[1;33m",  # Yellow
        "SUCCESS": "\033[1;32m",  # Green
        "INFO": "\033[1;34m",     # Blue
        "DEBUG": "\033[1;35m",    # Magenta
    }
    color = colors.get(level, "\033[36m")  # Default to Cyan

    # Ensure log directory exists
    try:
        log_path = Path(LOGFILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file_path = str(log_path.resolve()) # Get absolute path for writing
    except Exception as e:
        print(f"\033[1;31mError creating log directory/file '{LOGFILE}': {e}\033[0m", file=sys.stderr)
        return # Cannot log to file if path is invalid

    # Always log to file without colors
    try:
        with open(log_file_path, "a", encoding='utf-8') as f: # Specify encoding
            f.write(f"{timestamp} - [{level}] {message}\n")
    except Exception as e:
         print(f"\033[1;31mError writing to log file '{log_file_path}': {e}\033[0m", file=sys.stderr)

    # Check if we should suppress this message in console
    suppress_message = False

    # If in quiet mode, only show errors
    if QUIET_MODE and level != "ERROR":
        suppress_message = True

    # If suppressing warnings and this is a warning
    if SUPPRESS_WARNINGS and level == "WARNING":
        suppress_message = True

    # Log to console with colors if verbose and not suppressed
    if VERBOSE and not suppress_message:
        try:
            print(f"{color}{timestamp} - [{level}] {message}\033[0m")
        except Exception as e:
             # Fallback for potential encoding issues in console
             print(f"{timestamp} - [{level}] {message} (Error printing with color: {e})")

# Alias for log function to maintain compatibility
log_msg = log
