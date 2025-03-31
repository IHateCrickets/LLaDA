import os
import sys
import inspect
import traceback
import functools

# Debug and output settings
DEBUG_MODE = 0      # Set to 1 to enable enhanced debugging, 0 to disable
DEBUG_TRAP = 0      # Controls whether the DEBUG trap is active

# Output control settings
VERBOSE = 0             # Enable verbose console output
SUPPRESS_WARNINGS = 0   # Show all warnings
QUIET_MODE = 1          # Normal console output

# Default log file (Will be used by log_utils)
# Consider making this configurable or relative if needed elsewhere
LOGFILE = "logs/app.log" # Changed to relative path

def debug_decorator(func):
    """Decorator to provide debug information similar to bash's DEBUG trap"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if DEBUG_MODE and DEBUG_TRAP:
            try:
                # Get the current frame first
                current_frame = inspect.currentframe()
                calling_frame = current_frame.f_back if current_frame else None # Check if current_frame exists

                if calling_frame:
                    line_number = calling_frame.f_lineno
                    caller_function_info = inspect.getframeinfo(calling_frame)
                    caller_function = caller_function_info.function
                    # Use ANSI escape codes for color
                    print(f"\033[1;35m[DEBUG][{line_number}][{caller_function}]:\033[0m Calling {func.__name__}()")
                else:
                     print(f"\033[1;35m[DEBUG]:\033[0m Calling {func.__name__}() from unknown context")
            except Exception as e:
                 print(f"\033[1;31m[DEBUG TRAP ERROR]: {e}\033[0m")
        return func(*args, **kwargs)
    return wrapper

# Enable more verbose exception handling if in debug mode
if DEBUG_MODE:
    def excepthook(exc_type, exc_value, exc_traceback):
        # Use ANSI escape codes for color
        print("\033[1;31m[EXCEPTION]:\033[0m")
        traceback.print_exception(exc_type, exc_value, exc_traceback)

    sys.excepthook = excepthook