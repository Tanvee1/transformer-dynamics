# -*- coding: utf-8 -*-
"""
Metrics calculation module for Transformer Dynamics Lab.
Computes the 4 core research metrics across transformer layers:
1. Pairwise Distance
2. Cosine Similarity (SAM)
3. Effective Rank
4. Neural Stiffness Index (NSI)
"""

import numpy as np

def compute_pairwise_distance(layer_matrix: np.ndarray) -> float:
    """
    Computes average L2 pairwise distance between token vectors in a layer.
    Input shape: (N, D) where N = num_tokens, D = hidden_dim
    """
    N = layer_matrix.shape[0]
    if N <= 1:
        return 0.0
    
    # Efficient pairwise distance: ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u . v
    norms_sq = np.sum(layer_matrix ** 2, axis=1, keepdims=True)
    dot_prod = layer_matrix @ layer_matrix.T
    dist_sq = norms_sq + norms_sq.T - 2 * dot_prod
    dist_sq = np.maximum(dist_sq, 0.0) # clip numerical instabilities
    distances = np.sqrt(dist_sq)
    
    # Return mean pairwise distance (excluding self-distance)
    return float(np.sum(distances) / (N * (N - 1)))


def compute_cosine_similarity(layer_matrix: np.ndarray) -> float:
    """
    Computes average pairwise Cosine Similarity (Spectral Average Similarity) between token vectors.
    Input shape: (N, D)
    """
    N = layer_matrix.shape[0]
    if N <= 1:
        return 1.0
    
    # Normalize vectors
    norms = np.linalg.norm(layer_matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    norm_matrix = layer_matrix / norms
    
    cosine_sim_matrix = norm_matrix @ norm_matrix.T
    
    # Average off-diagonal elements
    np.fill_diagonal(cosine_sim_matrix, 0.0)
    return float(np.sum(cosine_sim_matrix) / (N * (N - 1)))


def compute_effective_rank(layer_matrix: np.ndarray) -> float:
    """
    Computes Effective Rank of the hidden state matrix X (N x D).
    erank(X) = exp( - sum p_k ln p_k ) where p_k = s_k / sum(s) from SVD.
    """
    N, D = layer_matrix.shape
    if N <= 1:
        return 1.0
    
    # Singular value decomposition
    try:
        singular_values = np.linalg.svd(layer_matrix, compute_uv=False)
    except Exception:
        return float(min(N, D))
    
    sv_sum = np.sum(singular_values)
    if sv_sum < 1e-9:
        return 1.0
    
    p = singular_values / sv_sum
    p = p[p > 1e-12] # remove zero probabilities for log
    
    entropy = -np.sum(p * np.log(p))
    erank = float(np.exp(entropy))
    return float(np.round(erank, 4))


def compute_nsi(hidden_states: list) -> list:
    """
    Computes Neural Stiffness Index (NSI) across consecutive layers.
    NSI_l = ||X^{(l+1)} - X^{(l)}||_F / (||X^{(l)}||_F + 1e-8)
    Returns a list of length L (Layer 0 NSI = 0.0)
    """
    nsi_list = [0.0] # Layer 0 NSI defined as 0
    L = len(hidden_states)
    
    for l in range(L - 1):
        X_curr = hidden_states[l]
        X_next = hidden_states[l + 1]
        
        diff_norm = np.linalg.norm(X_next - X_curr, ord='fro')
        curr_norm = np.linalg.norm(X_curr, ord='fro')
        
        stiffness = diff_norm / (curr_norm + 1e-8)
        nsi_list.append(float(stiffness))
        
    return nsi_list


def compute_all_layer_metrics(hidden_states: list) -> dict:
    """
    Computes all 4 core metrics for each layer in hidden_states.
    hidden_states: list of numpy arrays, each (N, D)
    """
    distances = []
    cos_sims = []
    effective_ranks = []
    
    for layer_X in hidden_states:
        distances.append(compute_pairwise_distance(layer_X))
        cos_sims.append(compute_cosine_similarity(layer_X))
        effective_ranks.append(compute_effective_rank(layer_X))
        
    nsi = compute_nsi(hidden_states)
    
    return {
        "distance": distances,
        "cosine_similarity": cos_sims,
        "effective_rank": effective_ranks,
        "nsi": nsi
    }
