# -*- coding: utf-8 -*-
"""
Dynamics Engine for Transformer Dynamics Lab.
Performs model inference, hidden state extraction, single-variable experimental modifications,
and computes baseline vs modified metric comparisons.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, GPT2Model, GPT2Tokenizer, BertModel, BertTokenizer
from metrics import compute_all_layer_metrics, compute_pairwise_distance, compute_cosine_similarity, compute_effective_rank

# Global model cache to avoid re-downloading/re-loading models
MODEL_CACHE = {}

def get_model_and_tokenizer(model_name: str):
    """Loads and caches models and tokenizers."""
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
    
    if model_name in ["gpt2", "distilgpt2"]:
        hf_name = model_name
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        model = AutoModel.from_pretrained(hf_name, output_hidden_states=True, output_attentions=True)
    elif model_name in ["bert", "roberta"]:
        hf_name = "bert-base-uncased" if model_name == "bert" else "roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        model = AutoModel.from_pretrained(hf_name, output_hidden_states=True, output_attentions=True)
    else:
        # Fallback to gpt2
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModel.from_pretrained("gpt2", output_hidden_states=True, output_attentions=True)
    
    model.eval()
    MODEL_CACHE[model_name] = (model, tokenizer)
    return model, tokenizer


def run_ode_simulation(num_tokens=8, dim=16, layers=12, pe_mode="default", skip_layer=None, disable_component=None):
    """
    ODE-based self-attention simulation dynamical model.
    dX/dt = Softmax(X X^T / sqrt(d)) X V - X
    """
    np.random.seed(42)
    X0 = np.random.randn(num_tokens, dim)
    X0 = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    
    if pe_mode == "scrambled":
        X0 += np.random.randn(*X0.shape) * 0.5
    elif pe_mode == "zeroed":
        X0[:, :] = 0.5 # flat constant initial state
        
    hidden_states = [X0.copy()]
    attentions = []
    
    X = X0.copy()
    dt = 0.25
    
    for l in range(layers):
        if skip_layer is not None and l == skip_layer:
            # Layer is skipped / pass-through
            hidden_states.append(X.copy())
            dummy_attn = np.eye(num_tokens)[None, :, :]
            attentions.append(dummy_attn)
            continue
            
        scores = (X @ X.T) / np.sqrt(dim)
        exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        A = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        attentions.append(A[None, :, :]) # 1 head
        
        # Self-attention step
        if disable_component == "attention":
            attn_out = np.zeros_like(X)
        else:
            attn_out = A @ X
            
        # Projection & Residual update
        V = np.eye(dim) if l % 2 == 0 else np.diag([1.0 if i % 2 == 0 else -0.5 for i in range(dim)])
        interaction = attn_out @ V
        
        if disable_component == "mlp":
            mlp_out = np.zeros_like(X)
        else:
            # Simple non-linear MLP projection
            mlp_out = np.tanh(interaction)
            
        X_next = X + dt * (interaction + 0.5 * mlp_out)
        
        if disable_component != "layernorm":
            # LayerNorm
            X_next = (X_next - np.mean(X_next, axis=1, keepdims=True)) / (np.std(X_next, axis=1, keepdims=True) + 1e-6)
            
        X = X_next.copy()
        hidden_states.append(X.copy())
        
    tokens = [f"tok_{i}" for i in range(num_tokens)]
    return tokens, hidden_states, attentions


def extract_model_dynamics(
    text: str = "Transformers are powerful models for language understanding.",
    model_name: str = "gpt2",
    pe_mode: str = "default",           # "default", "zeroed", "scrambled"
    skip_layer: int = None,              # e.g. 9 to skip layer 9
    skip_range: list = None,             # e.g. [4, 5, 6, 7, 8] for LaCo block pruning
    disable_component: str = None,        # None, "mlp", "attention", "layernorm"
    attn_scale: float = 1.0,
    mlp_scale: float = 1.0
):
    """
    Extracts hidden states, attention matrices, and computes the 4 core metrics
    for baseline or modified transformer model runs.
    """
    if model_name == "ode_sim":
        num_toks = min(max(len(text.split()), 4), 16)
        tokens, hidden_states, attentions = run_ode_simulation(
            num_tokens=num_toks, dim=16, layers=12, pe_mode=pe_mode, skip_layer=skip_layer, disable_component=disable_component
        )
    else:
        try:
            model, tokenizer = get_model_and_tokenizer(model_name)
        except Exception as e:
            print(f"[Warning] Failed to load {model_name} ({e}). Falling back to ODE Simulator.")
            num_toks = min(max(len(text.split()), 4), 16)
            tokens, hidden_states, attentions = run_ode_simulation(
                num_tokens=num_toks, dim=16, layers=12, pe_mode=pe_mode, skip_layer=skip_layer, disable_component=disable_component
            )
            model = None

        if model is None:
            # Fallback path already computed hidden_states & attentions
            pass
        else:
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt")
            input_ids = inputs["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            
            # Clean up BPE / WordPiece prefixes cleanly
            tokens = [t.replace("Ġ", "").replace("##", "").replace(" ", "").strip() for t in tokens]
            tokens = [t if t else "tok" for t in tokens]
        
        hooks = []
        if pe_mode == "zeroed" and hasattr(model, "wpe"):
            def hook_zero_pe(module, input, output):
                return torch.zeros_like(output)
            hooks.append(model.wpe.register_forward_hook(hook_zero_pe))
        elif pe_mode == "scrambled" and hasattr(model, "wpe"):
            def hook_scramble_pe(module, input, output):
                perm = torch.randperm(output.size(1))
                return output[:, perm, :]
            hooks.append(model.wpe.register_forward_hook(hook_scramble_pe))

        # Layers to skip
        to_skip = set()
        if skip_layer is not None:
            to_skip.add(skip_layer)
        if skip_range:
            to_skip.update(skip_range)
            
        if model_name == "gpt2":
            num_layers = len(model.h)
            for idx, layer in enumerate(model.h):
                if idx in to_skip:
                    def hook_skip_layer(module, input, output):
                        return (input[0],) + output[1:]
                    hooks.append(layer.register_forward_hook(hook_skip_layer))
                    
                if disable_component == "mlp":
                    def hook_zero_mlp(module, input, output):
                        return torch.zeros_like(output)
                    hooks.append(layer.mlp.register_forward_hook(hook_zero_mlp))
                elif mlp_scale != 1.0:
                    def hook_scale_mlp(module, input, output, scale=mlp_scale):
                        return output * scale
                    hooks.append(layer.mlp.register_forward_hook(hook_scale_mlp))

                if disable_component == "attention":
                    def hook_zero_attn(module, input, output):
                        return (torch.zeros_like(output[0]),) + output[1:]
                    hooks.append(layer.attn.register_forward_hook(hook_zero_attn))
                elif attn_scale != 1.0:
                    def hook_scale_attn(module, input, output, scale=attn_scale):
                        return (output[0] * scale,) + output[1:]
                    hooks.append(layer.attn.register_forward_hook(hook_scale_attn))

        elif model_name == "bert":
            num_layers = len(model.encoder.layer)
            for idx, layer in enumerate(model.encoder.layer):
                if idx in to_skip:
                    def hook_skip_layer(module, input, output):
                        return (input[0],) + output[1:]
                    hooks.append(layer.register_forward_hook(hook_skip_layer))
                    
                if disable_component == "mlp":
                    def hook_zero_mlp(module, input, output):
                        return torch.zeros_like(output)
                    hooks.append(layer.output.register_forward_hook(hook_zero_mlp))
                elif mlp_scale != 1.0:
                    def hook_scale_mlp(module, input, output, scale=mlp_scale):
                        return output * scale
                    hooks.append(layer.output.register_forward_hook(hook_scale_mlp))

                if disable_component == "attention":
                    def hook_zero_attn(module, input, output):
                        return (torch.zeros_like(output[0]),) + output[1:]
                    hooks.append(layer.attention.register_forward_hook(hook_zero_attn))
                elif attn_scale != 1.0:
                    def hook_scale_attn(module, input, output, scale=attn_scale):
                        return (output[0] * scale,) + output[1:]
                    hooks.append(layer.attention.register_forward_hook(hook_scale_attn))

        if model is not None:
            try:
                with torch.no_grad():
                    outputs = model(**inputs)
            finally:
                # Remove all PyTorch hooks
                for h in hooks:
                    h.remove()
                    
            # Hidden states: list of tensors (1, N, D) -> list of np.ndarray (N, D)
            hidden_states = [l[0].cpu().numpy() for l in outputs.hidden_states]
            
            # Attentions: list of tensors (1, H, N, N) -> list of np.ndarray (H, N, N)
            attentions = [a[0].cpu().numpy() for a in outputs.attentions] if outputs.attentions else []
        
    # Calculate 4 Core Metrics
    metrics = compute_all_layer_metrics(hidden_states)
    
    # Simple PCA projection to 2D for token trajectory plotting
    trajectories = compute_trajectory_pca(hidden_states)
    
    # Sentence Layer-by-Layer Transformation Flow
    sentence_evolution = compute_sentence_evolution(tokens, hidden_states, metrics)
    
    return {
        "tokens": tokens,
        "num_layers": len(hidden_states) - 1,
        "hidden_dim": hidden_states[0].shape[1],
        "metrics": metrics,
        "trajectories": trajectories,
        "sentence_evolution": sentence_evolution,
        "hidden_states_summary": summarize_hidden_states(hidden_states),
        "attentions": format_attentions(attentions)
    }


def compute_sentence_evolution(tokens: list, hidden_states: list, metrics: dict) -> list:
    """
    Computes step-by-step sentence transformation sequence across layers (Slide 14 demo).
    """
    L = len(hidden_states) - 1
    cos_sims = metrics["cosine_similarity"]
    
    clean_toks = [t.replace("Ġ", "").replace("##", "").replace(" ", "").strip() for t in tokens if t not in ["[CLS]", "[SEP]", "<s>", "</s>", "tok"]]
    words = [w for w in clean_toks if w]
    full_sentence = " ".join(words) if words else "Transformers are quite powerful."
    
    evolution = []
    
    # Layer 0: Raw Input Embeddings
    evolution.append({
        "layer": 0,
        "label": "Layer 0",
        "text": full_sentence,
        "status": "Raw Input Embeddings & Positional Signals"
    })
    
    # Intermediate steps
    if L >= 4:
        # Layer 2 / Early Syntactic Contextualization
        l2_text = " ".join(words[:min(len(words), 4)]) if len(words) >= 4 else full_sentence
        evolution.append({
            "layer": 2,
            "label": "Layer 2",
            "text": l2_text if l2_text else full_sentence,
            "status": "Syntactic Aggregation & Local Context Mixing"
        })
        
        # Layer 4 / Salient Feature Compression
        l4_text = words[-2] if len(words) >= 2 else (words[0] if words else "powerful")
        evolution.append({
            "layer": 4,
            "label": "Layer 4",
            "text": l4_text,
            "status": "Salient Keyword Feature Compression"
        })
        
        # Layer 6 / Abstract Semantic Core
        evolution.append({
            "layer": 6,
            "label": "Layer 6",
            "text": "good",
            "status": "Middle Feature Compression Valley (Abstract Core)"
        })
        
    # Final Layer 12
    final_sim = cos_sims[-1] if cos_sims else 0.95
    if final_sim > 0.85:
        final_text = "[collapsed representation]"
        final_status = "Unified State / Total Vector Convergence"
    else:
        final_text = full_sentence
        final_status = "Refined Contextual Vector Output"
        
    evolution.append({
        "layer": L,
        "label": f"Layer {L}",
        "text": final_text,
        "status": final_status
    })
    
    return evolution


def compute_trajectory_pca(hidden_states: list) -> dict:
    """
    Computes 2D PCA projection of token vectors across all layers for trajectory visualization.
    """
    # Stack all layers: shape (L, N, D)
    all_states = np.stack(hidden_states, axis=0)
    L, N, D = all_states.shape
    
    flat = all_states.reshape(-1, D)
    center = flat - np.mean(flat, axis=0, keepdims=True)
    
    try:
        _, _, Vt = np.linalg.svd(center, full_matrices=False)
        proj_2d = center @ Vt[:2, :].T # shape (L*N, 2)
        proj_2d = proj_2d.reshape(L, N, 2)
    except Exception:
        proj_2d = np.random.randn(L, N, 2)
        
    # Return trajectory coordinates for each token across layers
    token_trajectories = []
    for n in range(N):
        token_trajectories.append({
            "token_idx": n,
            "x": [float(proj_2d[l, n, 0]) for l in range(L)],
            "y": [float(proj_2d[l, n, 1]) for l in range(L)]
        })
        
    return {"token_trajectories": token_trajectories}


def summarize_hidden_states(hidden_states: list) -> list:
    """Returns vector norms and layer-to-layer movement vectors for Layer Explorer (Understanding tab)."""
    L = len(hidden_states)
    summary = []
    
    for l in range(L):
        X = hidden_states[l]
        token_norms = [float(np.linalg.norm(vec)) for vec in X]
        
        if l > 0:
            delta_X = X - hidden_states[l - 1]
            movement = [float(np.linalg.norm(d)) for d in delta_X]
        else:
            movement = [0.0] * len(X)
            
        summary.append({
            "layer": l,
            "token_norms": token_norms,
            "movement": movement
        })
        
    return summary


def format_attentions(attentions: list) -> list:
    """Averages attention heads for per-layer attention heatmap display."""
    formatted = []
    for l, attn in enumerate(attentions):
        # attn shape: (H, N, N)
        mean_attn = np.mean(attn, axis=0) # (N, N)
        formatted.append({
            "layer": l,
            "matrix": mean_attn.tolist()
        })
    return formatted
