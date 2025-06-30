#!/usr/bin/env python3
"""
Wrapper script to download Llama models with SSL verification disabled.
WARNING: This disables SSL certificate verification - use only in development/workshop environments!
"""
import os
import sys
import subprocess
import warnings
import urllib3

# Suppress SSL warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set environment variables
os.environ['PYTHONWARNINGS'] = 'ignore:Unverified HTTPS request'
os.environ['URLLIB3_DISABLE_WARNINGS'] = '1'

# Create a startup script that patches httpx
startup_script = '''
import httpx
import ssl

# Patch httpx to disable SSL verification
_original_async_init = httpx.AsyncClient.__init__
_original_sync_init = httpx.Client.__init__

def _patched_async_init(self, *args, **kwargs):
      kwargs['verify'] = False
      return _original_async_init(self, *args, **kwargs)

def _patched_sync_init(self, *args, **kwargs):
      kwargs['verify'] = False
      return _original_sync_init(self, *args, **kwargs)

httpx.AsyncClient.__init__ = _patched_async_init
httpx.Client.__init__ = _patched_sync_init

# Also create an unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Now import and run the main llama CLI
from llama_stack.cli.llama import main
main()
'''

# Write to temporary file
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
      f.write(startup_script)
      temp_file = f.name

try:
      # Run Python with our startup script
      cmd = [sys.executable, temp_file] + sys.argv[1:]
      
      # Run the command - it will prompt for URL if needed
      subprocess.run(cmd)
finally:
      # Clean up
      os.unlink(temp_file)
