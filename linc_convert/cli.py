"""Root command line entry point."""

import os
import logging
import sys
from cyclopts import App
from utils.logging import configure_logging

help = "Collection of conversion scripts for LINC datasets"
main = App("linc-convert", help=help)

# Configure logging
logger = configure_logging()


command = sys.argv
idx = None 
output_dir = None

try:
    if "--out" in command:
        idx = command.index("--out") + 1
    if not idx or idx >= len(command):
        raise IndexError("No value provided for '--inp' or '--out'.")
    output_dir = command[idx]

except ValueError as e:
    print(e)
except IndexError:
    print("Error: No value provided after '--out' or '--inp'.")


if output_dir:
    if os.path.isdir(output_dir): 
        log_dir = output_dir
    else:
        log_dir = os.path.dirname(output_dir)
    os.makedirs(log_dir, exist_ok=True)
else:
    log_dir = "./"


logging.basicConfig(
    filename=os.path.join(log_dir, 'command_history.json'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

message = " ".join(command)
logging.info(f"Executed command: {message}")  
