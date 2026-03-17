"""Claude Code Proxy

A proxy server that enables Claude Code to work with OpenAI-compatible API providers.
"""
from dotenv import load_dotenv
#import os
#import sys
#from pathlib import Path
#
#
#def get_app_dir() -> Path:
#    if getattr(sys, "frozen", False):
#        # Nuitka / frozen app
#        return Path(sys.executable).parent
#    else:
#        # Normal Python
#        return Path(__file__).parent.parent
#
#env_path = get_app_dir() / ".env"
#print(f"Loading env from {env_path}")
#load_dotenv(env_path)
load_dotenv()
__version__ = "1.0.1"
__author__ = "Claude Code Proxy With Web Search"
