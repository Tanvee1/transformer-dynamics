import sys
import os
import gradio as gr

# Add backend directory to PYTHONPATH
backend_dir = os.path.join(os.path.dirname(__file__), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import app as fastapi_app

# Create a clean Gradio interface that mounts the FastAPI web app
with gr.Blocks(title="Transformer Dynamics Lab") as demo:
    gr.HTML('<iframe src="/" style="width:100%; height:900px; border:none;"></iframe>')

# Mount FastAPI application onto Gradio for Hugging Face Spaces
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")
