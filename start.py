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

def find_python_with_deps():
    candidates = [
        "/Users/tanvee/miniforge3/bin/python3",
        sys.executable,
        "python3",
        "python"
    ]
    for cand in candidates:
        try:
            res = subprocess.run([cand, "-c", "import uvicorn, torch; print('ok')"], capture_output=True, text=True)
            if res.returncode == 0:
                return cand
        except Exception:
            pass
    return sys.executable

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "backend")
    
    # Check if current python has uvicorn & torch
    try:
        import uvicorn
        import torch
    except ImportError:
        valid_py = find_python_with_deps()
        if valid_py != sys.executable:
            os.execv(valid_py, [valid_py, __file__] + sys.argv[1:])
        else:
            print("❌ Dependencies missing. Installing requirements...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", os.path.join(root_dir, "requirements.txt")])
            import uvicorn
            import torch
    
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
