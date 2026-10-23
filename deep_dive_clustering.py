import sys
import time
import re
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import numpy as np
from safetensors import safe_open
from tqdm import tqdm

torch.set_num_threads(8)
model_path = "Test_model/anima_turboV11.safetensors"

print("Running Deep Dive: Consecutive Block Correlation & Clustered Subspaces...\n", flush=True)

with safe_open(model_path, framework="pt", device="cpu") as f:
    num_blocks = 28
    
    # Load all 28 Self-Attention Q projections
    tensors = []
    print("Loading 28 Self-Attention Q layers with progress bar:")
    for i in tqdm(range(num_blocks), desc="Loading Layers"):
        key = f"model.diffusion_model.blocks.{i}.self_attn.q_proj.weight"
        tensors.append(f.get_tensor(key).float())
        
    stack_A = torch.stack(tensors) # [28, 2048, 2048]
    n, m, k = stack_A.shape
    frob_orig = torch.norm(stack_A, p='fro', dim=(1, 2))
    total_frob_orig = torch.sqrt(torch.sum(frob_orig ** 2)).item()

    # 1. Test Consecutive Layer Drift (A_{i+1} - A_i)
    deltas = stack_A[1:] - stack_A[:-1] # [27, 2048, 2048]
    frob_deltas = torch.norm(deltas, p='fro', dim=(1, 2))
    total_frob_deltas = torch.sqrt(torch.sum(frob_deltas ** 2)).item()
    orig_subset_frob = torch.sqrt(torch.sum(frob_orig[1:] ** 2)).item()

    print(f"\n--- [Test A] Consecutive Layer Correlation (A_{{i+1}} = A_i + Delta_i) ---")
    print(f"Norm of Consecutive Deltas vs Raw Layer Norm: {total_frob_deltas / orig_subset_frob * 100:.2f}%")

    # 2. Test Clustered Bases (4 Stage Clusters: 7 blocks each)
    # Stage 0: Blocks 0-6 (Input / Early)
    # Stage 1: Blocks 7-13 (Mid 1)
    # Stage 2: Blocks 14-20 (Mid 2)
    # Stage 3: Blocks 21-27 (Output / Late)
    cluster_size = 7
    num_clusters = n // cluster_size
    
    clustered_err_sq = 0.0
    print(f"\n--- [Test B] Clustered Grouping ({num_clusters} Stage Clusters of {cluster_size} blocks each) ---")
    for c in range(num_clusters):
        cluster_stack = stack_A[c*cluster_size : (c+1)*cluster_size]
        cluster_mean = cluster_stack.mean(dim=0)
        cluster_res = cluster_stack - cluster_mean
        clustered_err_sq += torch.sum(torch.norm(cluster_res, p='fro', dim=(1, 2))**2).item()
        
    total_cluster_err = np.sqrt(clustered_err_sq)
    cluster_var_explained = (1.0 - (total_cluster_err**2) / (total_frob_orig**2)) * 100.0
    print(f"Variance explained by 4 Clustered Stage Bases: {cluster_var_explained:.2f}%")
    print(f"Cluster Residual Frobenius Error: {total_cluster_err / total_frob_orig * 100:.2f}%")

    # 3. Test Block-Level Quantization of Residuals vs Raw Weights
    # How much smaller is the dynamic range (scale factor) of residuals?
    X_mean = stack_A.mean(dim=0)
    residuals = stack_A - X_mean
    
    orig_std = torch.std(stack_A).item()
    res_std = torch.std(residuals).item()
    print(f"\n--- [Test C] Quantization Dynamic Range ---")
    print(f"Standard deviation of Original Weights:  {orig_std:.6f}")
    print(f"Standard deviation of Residual Weights:  {res_std:.6f}")
    print(f"Variance Reduction from Mean Subtraction: {(1.0 - (res_std**2)/(orig_std**2))*100:.2f}%")
