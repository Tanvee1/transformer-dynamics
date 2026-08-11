# -*- coding: utf-8 -*-
"""
Verification tests for Transformer Dynamics Lab metrics and ODE dynamics.
"""

import numpy as np
from metrics import (
    compute_pairwise_distance,
    compute_cosine_similarity,
    compute_effective_rank,
    compute_nsi,
    compute_all_layer_metrics
)
from dynamics_engine import run_ode_simulation, extract_model_dynamics

def test_metrics_on_synthetic_data():
    print("--- Running Metrics Tests ---")
    
    # 1. Identity matrix (orthogonal vectors)
    X_ortho = np.eye(4)
    dist_ortho = compute_pairwise_distance(X_ortho)
    cos_ortho = compute_cosine_similarity(X_ortho)
    erank_ortho = compute_effective_rank(X_ortho)
    
    print(f"Ortho Matrix -> Distance: {dist_ortho:.4f}, Cosine Sim: {cos_ortho:.4f}, Effective Rank: {erank_ortho:.4f}")
    assert cos_ortho == 0.0, "Orthogonal vectors must have 0.0 mean cosine similarity"
    assert np.isclose(erank_ortho, 4.0, atol=0.01), f"Effective rank of 4x4 identity should be 4.0, got {erank_ortho}"
    
    # 2. Collapsed matrix (all identical vectors)
    X_collapsed = np.ones((4, 4))
    dist_col = compute_pairwise_distance(X_collapsed)
    cos_col = compute_cosine_similarity(X_collapsed)
    erank_col = compute_effective_rank(X_collapsed)
    
    print(f"Collapsed Matrix -> Distance: {dist_col:.4f}, Cosine Sim: {cos_col:.4f}, Effective Rank: {erank_col:.4f}")
    assert np.isclose(dist_col, 0.0), "Collapsed vectors must have 0.0 pairwise distance"
    assert np.isclose(cos_col, 1.0), "Identical vectors must have 1.0 cosine similarity"
    assert np.isclose(erank_col, 1.0, atol=0.01), f"Effective rank of rank-1 matrix should be 1.0, got {erank_col}"
    
    print("✓ All synthetic metric assertions passed!")

def test_ode_dynamics_engine():
    print("\n--- Running ODE Dynamics Engine Test ---")
    tokens, hidden_states, attentions = run_ode_simulation(num_tokens=6, dim=8, layers=10)
    metrics = compute_all_layer_metrics(hidden_states)
    
    print(f"Tokens ({len(tokens)}):", tokens)
    print(f"Hidden Layers ({len(hidden_states)}): shape {hidden_states[0].shape}")
    print("Initial vs Final Distance:", metrics["distance"][0], "->", metrics["distance"][-1])
    print("Initial vs Final Rank:", metrics["effective_rank"][0], "->", metrics["effective_rank"][-1])
    print("NSI profile:", [round(x, 3) for x in metrics["nsi"]])
    
    assert len(metrics["distance"]) == 11
    assert len(metrics["nsi"]) == 11
    print("✓ ODE Simulation Engine passed!")

if __name__ == "__main__":
    test_metrics_on_synthetic_data()
    test_ode_dynamics_engine()
