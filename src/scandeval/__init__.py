"""
.. include:: ../../README.md
"""

import importlib.metadata
import logging
import os
import warnings

from dotenv import load_dotenv
from termcolor import colored

from .utils import HiddenPrints, block_terminal_output

# The warnings and prints hidden here come from the `bitsandbytes` package, when no GPU
# is available
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    with HiddenPrints():
        from .benchmarker import Benchmarker

# Fetches the version of the package as defined in pyproject.toml
__version__ = importlib.metadata.version(__package__)


# Loads environment variables
load_dotenv()


block_terminal_output()


# Set up logging
fmt = colored("%(asctime)s", "light_blue") + " ⋅ " + colored("%(message)s", "green")
logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


# Disable parallelisation when tokenizing, as that can lead to errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Enable MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# Set amount of threads per GPU - this is the default and is only set to prevent a
# warning from showing
os.environ["OMP_NUM_THREADS"] = "1"
