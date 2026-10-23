import os
import sys
import numpy as np

def run_analysis(layers, layer_names=None):
    """
    Given a list of weight matrices A_1, A_2, ..., A_n (each of shape [m, k]),
    runs empirical tests comparing:
      1. Baseline: Independent Low-Rank (SVD on each layer individually)
      2. Baseline: Mean Matrix alone (Professor's Section 1)
      3. Option 1: Shared Mean X* + Low-Rank Residual (Inverted LoRA)
      4. Option 2: Shared Mean X* + Diagonal Scaling (P_i * X* * Q_i)
    """
    import torch

    n = len(layers)
    m, k = layers[0].shape
    print(f"\n==================================================")
    print(f"ANALYSIS REPORT: Inter-Layer Structural Redundancy")
    print(f"Number of layers evaluated: {n}")
    print(f"Matrix dimensions per layer: {m} x {k} ({m*k*2 / (1024**2):.2f} MB in FP16)")
    print(f"Total uncompressed size:    {n * m * k * 2 / (1024**2):.2f} MB")
    print(f"==================================================\n")

    stack_A = torch.stack(layers).float()  # [n, m, k]

    # 1. Compute Professor's Section 1: Mean Matrix X*
    X_mean = stack_A.mean(dim=0)  # [m, k]
    
    # 2. Frobenius Norm of original matrices
    frob_orig = torch.norm(stack_A, p='fro', dim=(1, 2))  # [n]
    total_frob_orig = torch.sqrt(torch.sum(frob_orig ** 2)).item()

    # Residuals from mean
    residuals = stack_A - X_mean  # [n, m, k]
    frob_res = torch.norm(residuals, p='fro', dim=(1, 2))
    total_frob_res = torch.sqrt(torch.sum(frob_res ** 2)).item()

    mean_rel_error = total_frob_res / total_frob_orig
    variance_explained_by_mean = 1.0 - (total_frob_res**2) / (total_frob_orig**2)

    print(f"--- 1. Shared Mean Matrix (Professor's Step 1) ---")
    print(f"Variance captured by shared mean X* alone: {variance_explained_by_mean * 100:.2f}%")
    print(f"Relative Frobenius error with ONLY mean:  {mean_rel_error * 100:.2f}%\n")

    # 3. Analyze Singular Value Spectrum of Originals vs Residuals
    print(f"--- 2. Low-Rank Truncation Test (Ranks 4, 8, 16, 32) ---")
    ranks = [4, 8, 16, 32]
    
    print(f"{'Rank (r)':<10} | {'Indep. SVD Error':<18} | {'Shared X* + SVD Error':<22} | {'Shared X* Storage (MB)':<22} | {'Savings %':<10}")
    print("-" * 92)

    # Size of uncompressed
    raw_bytes = n * m * k * 2  # FP16

    for r in ranks:
        # A. Independent SVD per layer
        err_indep_sq = 0.0
        for i in range(n):
            U, S, Vh = torch.linalg.svd(stack_A[i], full_matrices=False)
            A_approx = (U[:, :r] * S[:r]) @ Vh[:r, :]
            err_indep_sq += torch.norm(stack_A[i] - A_approx, p='fro').item() ** 2
        rel_err_indep = np.sqrt(err_indep_sq) / total_frob_orig

        # B. Shared Mean + SVD on Residuals
        err_shared_sq = 0.0
        for i in range(n):
            U, S, Vh = torch.linalg.svd(residuals[i], full_matrices=False)
            res_approx = (U[:, :r] * S[:r]) @ Vh[:r, :]
            approx_total = X_mean + res_approx
            err_shared_sq += torch.norm(stack_A[i] - approx_total, p='fro').item() ** 2
        rel_err_shared = np.sqrt(err_shared_sq) / total_frob_orig

        # Storage for Shared Mean + Low Rank:
        # 1 copy of X_mean [m*k] + n copies of (U [m*r] + Vh [r*k])
        shared_bytes = (m * k + n * (m * r + r * k)) * 2  # FP16
        savings = (1.0 - shared_bytes / raw_bytes) * 100.0

        print(f"{r:<10} | {rel_err_indep*100:>15.2f}% | {rel_err_shared*100:>19.2f}% | {shared_bytes/(1024**2):>19.2f} MB | {savings:>8.1f}%")

    print("\n--- 3. Diagonal Channel Scaling Test (A_i ≈ P_i * X_mean * Q_i) ---")
    # Alternating Least Squares or simple row/col norm scaling
    err_diag_sq = 0.0
    for i in range(n):
        # Initial estimate: p_i = row norms ratio, q_i = col norms ratio
        row_norm_orig = torch.norm(stack_A[i], dim=1, keepdim=True)
        row_norm_mean = torch.norm(X_mean, dim=1, keepdim=True) + 1e-8
        p_i = row_norm_orig / row_norm_mean

        col_norm_orig = torch.norm(stack_A[i], dim=0, keepdim=True)
        col_norm_mean = torch.norm(X_mean, dim=0, keepdim=True) + 1e-8
        q_i = col_norm_orig / col_norm_mean

        # Quick refinement
        scaled = p_i * X_mean * q_i
        # Fit optimal scalar alpha
        alpha = (torch.sum(scaled * stack_A[i]) / (torch.sum(scaled ** 2) + 1e-8)).item()
        approx_diag = alpha * scaled
        err_diag_sq += torch.norm(stack_A[i] - approx_diag, p='fro').item() ** 2

    rel_err_diag = np.sqrt(err_diag_sq) / total_frob_orig
    diag_bytes = (m * k + n * (m + k + 1)) * 2
    diag_savings = (1.0 - diag_bytes / raw_bytes) * 100.0
    print(f"Diagonal Scaling Relative Error: {rel_err_diag * 100:.2f}%")
    print(f"Diagonal Scaling Storage:        {diag_bytes / (1024**2):.2f} MB (Compression Savings: {diag_savings:.1f}%)")
    print("==================================================\n")

if __name__ == '__main__':
    import torch
    print(f"PyTorch version: {torch.__version__} (Device: cpu)")
    
    # Try fetching real weights from a trained transformer (e.g. ViT / DiT backbone)
    weights_loaded = False
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        
        print("Downloading a real Vision Transformer backbone (google/vit-base-patch16-224) to extract real layer weights...")
        # Download just the safetensors file (~340 MB)
        file_path = hf_hub_download(repo_id="google/vit-base-patch16-224", filename="model.safetensors")
        state_dict = load_file(file_path)
        
        # Extract attention query, key, value and MLP layers across all 12 blocks
        # Keys look like: vit.encoder.layer.0.attention.attention.query.weight
        q_weights = [state_dict[f"vit.encoder.layer.{i}.attention.attention.query.weight"] for i in range(12)]
        mlp_weights = [state_dict[f"vit.encoder.layer.{i}.intermediate.dense.weight"] for i in range(12)]
        
        print("\n>>> TEST 1: Attention Query Weights (Across all 12 layers)")
        run_analysis(q_weights)

        print("\n>>> TEST 2: MLP Dense Expansion Weights (Across all 12 layers)")
        run_analysis(mlp_weights)

        weights_loaded = True
    except Exception as e:
        print(f"Note: Could not download HuggingFace model directly ({e}).")
        print("Generating a representative deep transformer model (19 layers, hidden_dim=1536) to simulate Dual-Stream DiT blocks...")
        
        # Simulate a 19-layer dual-stream DiT setup (e.g. Flux-like dimensions: 1536 x 1536)
        # where layers share base representations with smooth cross-layer drift
        torch.manual_seed(42)
        n_layers = 19
        dim = 1536
        
        # Base template (underlying generator matrix)
        base = torch.randn(dim, dim) * 0.02
        
        # Layers evolving with a drift + low-rank variation
        layers = []
        for i in range(n_layers):
            # Slow smooth layer drift
            drift = (i / n_layers) * torch.randn(dim, dim) * 0.005
            # Low rank adaptation (rank 8)
            U = torch.randn(dim, 8)
            V = torch.randn(8, dim)
            lora = (U @ V) * (0.01 / 8)
            noise = torch.randn(dim, dim) * 0.002
            layers.append(base + drift + lora + noise)
            
        print("\n>>> SIMULATED 19-LAYER DUAL-STREAM DiT BLOCKS:")
        run_analysis(layers)

