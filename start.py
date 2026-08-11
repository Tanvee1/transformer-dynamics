#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-command launcher for Transformer Dynamics Lab.
Starts the FastAPI server on port 8050 and opens the browser interface.
"""

import sys
import os
import subprocess
import webbrowser
import time

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "backend")
    
    print("=" * 65)
    print("🚀 Launching Transformer Dynamics Lab Platform...")
    print("=" * 65)
    
    # Ensure backend directory is in PYTHONPATH
    sys.path.insert(0, backend_dir)
    os.environ["PYTHONPATH"] = backend_dir
    
    host = "0.0.0.0"
    port = 8050
    url = f"http://localhost:{port}"
    
    print(f"\n🌐 Web Interface: {url}")
    print("Press Ctrl+C to terminate server.\n")
    
    # Launch uvicorn
    import uvicorn
    uvicorn.run("app:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    main()
