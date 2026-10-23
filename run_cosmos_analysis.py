import sys
import time
import re
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import numpy as np
from safetensors import safe_open

# Optimize PyTorch CPU threads for 13th gen HX (6 P-cores / 12 threads)
torch.set_num_threads(8)

model_path = "Test_model/anima_turboV11.safetensors"
print(f"Loading Cosmos DiT architecture from: {model_path} ...", flush=True)

with safe_open(model_path, framework="pt", device="cpu") as f:
    all_keys = list(f.keys())
    
    # Discover all blocks
    block_indices = set()
    for k in all_keys:
        m = re.match(r"model\.diffusion_model\.blocks\.(\d+)\.", k)
        if m:
            block_indices.add(int(m.group(1)))
            
    num_blocks = len(block_indices)
    print(f"Detected Cosmos DiT architecture with {num_blocks} Transformer Blocks (0 to {num_blocks-1}).\n", flush=True)

    def analyze_tensor_group(group_name, key_template):
        t0 = time.time()
        print(f"================================================================================", flush=True)
        print(f"ANALYSIS: {group_name}", flush=True)
        print(f"Key template: {key_template}", flush=True)
        
        # Load tensors for this group
        tensors = []
        for i in range(num_blocks):
            key = key_template.format(i)
            tensors.append(f.get_tensor(key).float())
            
        stack_A = torch.stack(tensors)  # [n, m, k]
        n, m, k = stack_A.shape
        raw_mb = n * m * k * 2 / (1024**2) # FP16 size
        print(f"Shape per layer: [{m}, {k}] | Total uncompressed (FP16): {raw_mb:.2f} MB across {n} blocks", flush=True)
        
        # 1. Professor's Section 1: Mean Matrix X*
        X_mean = stack_A.mean(dim=0)
        
        frob_orig = torch.norm(stack_A, p='fro', dim=(1, 2))
        total_frob_orig = torch.sqrt(torch.sum(frob_orig ** 2)).item()
        
        residuals = stack_A - X_mean
        frob_res = torch.norm(residuals, p='fro', dim=(1, 2))
        total_frob_res = torch.sqrt(torch.sum(frob_res ** 2)).item()
        
        rel_error_mean_only = (total_frob_res / total_frob_orig) * 100.0
        var_explained = (1.0 - (total_frob_res**2) / (total_frob_orig**2)) * 100.0
        
        print(f"\n[1] Professor's Step 1: Mean Matrix X* alone", flush=True)
        print(f"  - Variance explained by single shared X*: {var_explained:.2f}%", flush=True)
        print(f"  - Relative Frobenius Error (Mean alone): {rel_error_mean_only:.2f}%", flush=True)
        
        # 2. Fast Single-Pass SVD (compute once per layer, then slice ranks!)
        print(f"\n[2] Computing SVDs across {n} blocks (single-pass decomposition)...", flush=True)
        svd_t0 = time.time()
        
        # Precompute SVDs of original matrices
        orig_svds = []
        for i in range(n):
            U, S, Vh = torch.linalg.svd(stack_A[i], full_matrices=False)
            orig_svds.append((U, S, Vh))
            
        # Precompute SVDs of residuals
        res_svds = []
        for i in range(n):
            U, S, Vh = torch.linalg.svd(residuals[i], full_matrices=False)
            res_svds.append((U, S, Vh))
            
        print(f"  SVD computed in {time.time() - svd_t0:.1f}s. Evaluating ranks...", flush=True)

        print(f"\n{'Rank (r)':<10} | {'Indep. SVD Error':<18} | {'Shared X* + LowRank':<22} | {'Storage (MB)':<15} | {'VRAM Savings':<12}", flush=True)
        print("-" * 82, flush=True)
        
        raw_bytes = n * m * k * 2
        for r in [4, 8, 16, 32, 64, 128]:
            # Independent SVD error
            err_indep_sq = 0.0
            for i in range(n):
                U, S, Vh = orig_svds[i]
                approx = (U[:, :r] * S[:r]) @ Vh[:r, :]
                err_indep_sq += torch.norm(stack_A[i] - approx, p='fro').item() ** 2
            err_indep = (np.sqrt(err_indep_sq) / total_frob_orig) * 100.0
            
            # Shared Mean + SVD on Residuals
            err_shared_sq = 0.0
            for i in range(n):
                U, S, Vh = res_svds[i]
                approx_res = (U[:, :r] * S[:r]) @ Vh[:r, :]
                err_shared_sq += torch.norm(stack_A[i] - (X_mean + approx_res), p='fro').item() ** 2
            err_shared = (np.sqrt(err_shared_sq) / total_frob_orig) * 100.0
            
            shared_bytes = (m * k + n * (m * r + r * k)) * 2
            shared_mb = shared_bytes / (1024**2)
            savings = (1.0 - shared_bytes / raw_bytes) * 100.0
            
            print(f"{r:<10} | {err_indep:>15.2f}% | {err_shared:>19.2f}% | {shared_mb:>12.2f} MB | {savings:>10.1f}%", flush=True)

        # 3. Bilinear / Diagonal Channel Scaling Test
        err_diag_sq = 0.0
        for i in range(n):
            row_norm_orig = torch.norm(stack_A[i], dim=1, keepdim=True)
            row_norm_mean = torch.norm(X_mean, dim=1, keepdim=True) + 1e-8
            p_i = row_norm_orig / row_norm_mean
            
            col_norm_orig = torch.norm(stack_A[i], dim=0, keepdim=True)
            col_norm_mean = torch.norm(X_mean, dim=0, keepdim=True) + 1e-8
            q_i = col_norm_orig / col_norm_mean
            
            scaled = p_i * X_mean * q_i
            alpha = (torch.sum(scaled * stack_A[i]) / (torch.sum(scaled ** 2) + 1e-8)).item()
            approx_diag = alpha * scaled
            err_diag_sq += torch.norm(stack_A[i] - approx_diag, p='fro').item() ** 2
            
        err_diag = (np.sqrt(err_diag_sq) / total_frob_orig) * 100.0
        diag_bytes = (m * k + n * (m + k + 1)) * 2
        diag_mb = diag_bytes / (1024**2)
        diag_savings = (1.0 - diag_bytes / raw_bytes) * 100.0
        print(f"\n[3] Diagonal Channel Scaling (A_i ~= p_i * X* * q_i):", flush=True)
        print(f"  - Relative Error:   {err_diag:.2f}%", flush=True)
        print(f"  - Compressed Size:  {diag_mb:.2f} MB (Compression Savings: {diag_savings:.1f}%)", flush=True)
        print(f"Completed in {time.time() - t0:.1f} seconds.", flush=True)
        print("================================================================================\n", flush=True)

    # 1. Self Attention Q Projection
    analyze_tensor_group("Self-Attention Q Projections", 
                         "model.diffusion_model.blocks.{}.self_attn.q_proj.weight")

    # 2. Self Attention Output Projection
    analyze_tensor_group("Self-Attention Output Projections", 
                         "model.diffusion_model.blocks.{}.self_attn.output_proj.weight")

    # 3. MLP Layer 1 (Dense Expansion: 8192 x 2048)
    analyze_tensor_group("Feed-Forward MLP Layer 1 (8192 x 2048)", 
                         "model.diffusion_model.blocks.{}.mlp.layer1.weight")

print("Cosmos DiT layer relationship analysis complete!", flush=True)
