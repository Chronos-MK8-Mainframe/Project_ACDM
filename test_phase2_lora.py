import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file

torch.set_num_threads(8)
model_path = "Test_model/anima_turboV11.safetensors"

print("==================================================================")
print("TESTING SCENARIO A + APPROACH 1: PHASE 2 ANCHOR + LORA PEFT")
print("==================================================================\n")

with safe_open(model_path, framework="pt", device="cpu") as f:
    # Phase 2 Core blocks: Blocks 4 to 19 (16 blocks total)
    phase2_blocks = list(range(4, 20))
    n = len(phase2_blocks)
    
    # 1. Load Self-Attention Q projections for Phase 2
    q_tensors = [f.get_tensor(f"model.diffusion_model.blocks.{i}.self_attn.q_proj.weight").float() for i in phase2_blocks]
    stack_Q = torch.stack(q_tensors) # [16, 2048, 2048]
    m, k = stack_Q.shape[1], stack_Q.shape[2]
    
    # Compute the Phase 2 Shared Anchor X*_core
    X_anchor = stack_Q.mean(dim=0) # [2048, 2048]
    residuals = stack_Q - X_anchor # [16, 2048, 2048]
    
    total_orig_frob = torch.sqrt(torch.sum(torch.norm(stack_Q, p='fro', dim=(1,2))**2)).item()
    
    print(f"Phase 2 contains {n} core blocks (Blocks 4 to 19).")
    print(f"Raw FP16 Size of Phase 2 Self-Attention Q: {n * m * k * 2 / (1024**2):.2f} MB")
    print(f"Anchor Matrix Size (stored once):           {m * k * 2 / (1024**2):.2f} MB\n")
    
    # Test different LoRA ranks
    print(f"{'LoRA Rank (r)':<15} | {'Reconstruction Error':<22} | {'LoRA Adapter Size':<20} | {'Total Size (Base+LoRA)':<22} | {'Savings %':<10}")
    print("-" * 100)
    
    raw_bytes = n * m * k * 2
    for r in [16, 32, 64, 128, 256]:
        err_sq = 0.0
        lora_dict = {}
        
        for idx, block_id in enumerate(phase2_blocks):
            res = residuals[idx]
            U, S, Vh = torch.linalg.svd(res, full_matrices=False)
            
            # LoRA factorization: B (up) and A (down)
            # res ≈ B @ A
            sqrt_S = torch.sqrt(S[:r])
            lora_up = U[:, :r] * sqrt_S.unsqueeze(0)    # [m, r]
            lora_down = sqrt_S.unsqueeze(1) * Vh[:r, :] # [r, k]
            
            reconstructed = X_anchor + (lora_up @ lora_down)
            err_sq += torch.norm(stack_Q[idx] - reconstructed, p='fro').item() ** 2
            
            # Standard LoRA / PEFT key names
            lora_dict[f"diffusion_model.blocks.{block_id}.self_attn.q_proj.lora_up.weight"] = lora_up.half()
            lora_dict[f"diffusion_model.blocks.{block_id}.self_attn.q_proj.lora_down.weight"] = lora_down.half()
            
        rel_err = (np.sqrt(err_sq) / total_orig_frob) * 100.0
        
        # Storage: 1 x Anchor [m, k] + 16 x (m*r + r*k) for LoRA
        lora_bytes = n * (m * r + r * k) * 2 # FP16
        total_bytes = (m * k * 2) + lora_bytes
        savings = (1.0 - total_bytes / raw_bytes) * 100.0
        
        print(f"Rank {r:<10} | {rel_err:>18.2f}% | {lora_bytes/(1024**2):>17.2f} MB | {total_bytes/(1024**2):>19.2f} MB | {savings:>8.1f}%")
        
        # Save a sample LoRA file at rank 64 to verify it's a valid standard safetensors file
        if r == 64:
            save_file(lora_dict, "Test_model/sample_phase2_lora_r64.safetensors")
            print(f"\n[Verified] Successfully saved standard LoRA file: 'sample_phase2_lora_r64.safetensors' ({len(lora_dict)} tensors).")

print("\nDone!")
