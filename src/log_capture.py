"""
Log capture utility that saves all print output to file immediately.
Excludes progress bars (tqdm) from logging.
"""
import sys
import os
from datetime import datetime
import re

class TeeOutput:
    """
    Captures stdout/stderr and writes to both console and file immediately.
    Filters out tqdm progress bar output.
    """
    
    def __init__(self, log_file_path, is_stderr=False):
        if is_stderr:
            self.terminal = sys.stderr
        else:
            self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'a' if is_stderr else 'w', encoding='utf-8', buffering=1)  # Line buffered
        self.log_file_path = log_file_path
        self.is_stderr = is_stderr
        
    def write(self, message):
        # Filter out tqdm progress bars (they contain carriage returns and overwrite)
        # tqdm typically writes lines with \r and percentage indicators
        if self._is_tqdm_output(message):
            # Still show in terminal but don't log to file
            self.terminal.write(message)
            self.terminal.flush()
            return
        
        # Write to terminal
        self.terminal.write(message)
        self.terminal.flush()
        
        # Write to file immediately (no buffering)
        if message:  # Only write non-empty messages
            self.log_file.write(message)
            self.log_file.flush()  # Immediate write - no memory buffering
    
    def _is_tqdm_output(self, message):
        """
        Detect if message is from tqdm progress bar.
        tqdm outputs typically contain:
        - Carriage returns (\r) at the start
        - Percentage indicators (100%)
        - Progress bar characters (|, █, etc.)
        - "it/s" or "s/it" speed indicators
        """
        if not message or not message.strip():
            return False
        
        # Check for tqdm patterns with carriage return
        if '\r' in message:
            # tqdm typically starts with \r and has percentage or progress bar
            if '%' in message or '|' in message or '█' in message or '▉' in message:
                return True
            
            # Check for tqdm speed indicators: "it/s" or "s/it"
            if re.search(r'\d+\.?\d*\s*(it/s|s/it)', message, re.IGNORECASE):
                return True
            
            # Check for tqdm's typical format: "Epoch X/Y: 100%|████████|"
            if re.search(r'\d+%\|[█▉▊▋▌▍▎▏\s]+\|', message):
                return True
        
        # Check for Optuna's tqdm format: "[XX:XX<XX:XX, X.XXit/s]"
        if re.search(r'\[\d+:\d+<[\d:]+,?\s*\d+\.\d+it/s\]', message):
            return True
        
        return False
    
    def flush(self):
        """Ensure both streams are flushed"""
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        """Close the log file"""
        if self.log_file:
            self.log_file.close()

class LogCapture:
    """
    Context manager for capturing all print output to a file.
    Filters out tqdm progress bars but logs everything else immediately.
    """
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        self.tee_stdout = None
        self.tee_stderr = None
        self.original_stdout = None
        self.original_stderr = None
        self.log_file_path = None
        
    def __enter__(self):
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(self.results_dir, f"training_log_{timestamp}.txt")
        
        # Save original stdout and stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create Tee for stdout (main output)
        self.tee_stdout = TeeOutput(self.log_file_path, is_stderr=False)
        
        # Create Tee for stderr (errors, but we'll filter tqdm)
        self.tee_stderr = TeeOutput(self.log_file_path, is_stderr=True)
        
        # Redirect stdout and stderr to our Tee
        sys.stdout = self.tee_stdout
        sys.stderr = self.tee_stderr
        
        # Write header to log file
        header = f"\n{'='*70}\n"
        header += f"Training Log Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"Log File: {self.log_file_path}\n"
        header += f"{'='*70}\n\n"
        self.tee_stdout.write(header)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original stdout and stderr
        if self.original_stdout:
            sys.stdout = self.original_stdout
        if self.original_stderr:
            sys.stderr = self.original_stderr
        
        # Write footer and close
        if self.tee_stdout:
            footer = f"\n{'='*70}\n"
            footer += f"Training Log Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            if exc_type:
                footer += f"Exit Status: ERROR - {exc_type.__name__}: {exc_val}\n"
            else:
                footer += f"Exit Status: SUCCESS\n"
            footer += f"{'='*70}\n"
            self.tee_stdout.write(footer)
            self.tee_stdout.close()
        
        if self.tee_stderr:
            self.tee_stderr.close()
        
        return False  # Don't suppress exceptions
    
    def get_log_path(self):
        """Get the path to the log file"""
        return self.log_file_path

