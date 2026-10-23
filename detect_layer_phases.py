import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import numpy as np
from safetensors import safe_open

model_path = "Test_model/anima_turboV11.safetensors"
print("Scanning Cosmos DiT across all 28 blocks to detect pattern boundaries...\n")

with safe_open(model_path, framework="pt", device="cpu") as f:
    num_blocks = 28
    
    # Load all 28 Self-Attention Q and MLP layers
    q_tensors = [f.get_tensor(f"model.diffusion_model.blocks.{i}.self_attn.q_proj.weight").float() for i in range(num_blocks)]
    mlp_tensors = [f.get_tensor(f"model.diffusion_model.blocks.{i}.mlp.layer1.weight").float() for i in range(num_blocks)]
    
    def analyze_boundaries(name, tensors):
        n = len(tensors)
        stack = torch.stack(tensors) # [n, m, k]
        
        # Compute Cosine Similarity between all consecutive pairs (i, i+1)
        # Cosine sim = dot(A_i, A_{i+1}) / (||A_i|| * ||A_{i+1}||)
        flat = stack.view(n, -1)
        norms = torch.norm(flat, dim=1, keepdim=True)
        flat_normed = flat / (norms + 1e-8)
        
        sim_matrix = (flat_normed @ flat_normed.T).numpy() # [28, 28]
        consec_sims = [sim_matrix[i, i+1] for i in range(n-1)]
        
        print(f"=================================================================")
        print(f"BOUNDARY DETECTION: {name}")
        print(f"Consecutive Layer Cosine Similarities (Block i to Block i+1):")
        print(f"-----------------------------------------------------------------")
        for i, s in enumerate(consec_sims):
            bar = "#" * int(max(0, s * 50))
            flag = " <--- POTENTIAL PATTERN BREAK!" if s < np.mean(consec_sims) - 0.5 * np.std(consec_sims) else ""
            print(f"Block {i:>2} -> Block {i+1:>2} | Sim: {s:+.4f} | {bar}{flag}")
            
        # Adaptive Greedy Clustering based on a similarity threshold
        # Start a new group whenever similarity drops below threshold
        threshold = np.mean(consec_sims)
        groups = []
        current_group = [0]
        
        for i in range(n - 1):
            if consec_sims[i] < threshold:
                groups.append(current_group)
                current_group = [i + 1]
            else:
                current_group.append(i + 1)
        groups.append(current_group)
        
        print(f"\nDiscovered {len(groups)} Natural Phase Groups:")
        total_frob_orig = torch.sqrt(torch.sum(torch.norm(stack, p='fro', dim=(1,2))**2)).item()
        total_group_res_sq = 0.0
        
        for g_idx, grp in enumerate(groups):
            grp_stack = stack[grp]
            grp_mean = grp_stack.mean(dim=0)
            res = grp_stack - grp_mean
            res_norm_sq = torch.sum(torch.norm(res, p='fro', dim=(1,2))**2).item()
            total_group_res_sq += res_norm_sq
            
            orig_norm_sq = torch.sum(torch.norm(grp_stack, p='fro', dim=(1,2))**2).item()
            grp_var_exp = (1.0 - res_norm_sq / (orig_norm_sq + 1e-8)) * 100.0
            print(f"  Group {g_idx+1}: Blocks {grp[0]:>2} to {grp[-1]:>2} ({len(grp):>2} blocks) | Group Variance Captured by Mean: {grp_var_exp:>5.1f}%")
            
        overall_var_exp = (1.0 - total_group_res_sq / (total_frob_orig**2)) * 100.0
        print(f"\nOverall Variance Captured across model with {len(groups)} Adaptive Groups: {overall_var_exp:.2f}%\n")

    analyze_boundaries("Self-Attention Q Projections", q_tensors)
    analyze_boundaries("Feed-Forward MLP Layer 1", mlp_tensors)
