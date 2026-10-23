import sys
import os
import time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from tqdm import tqdm

input_path = "Test_model/anima_turboV11.safetensors"
output_path = "Test_model/anima_turboV11_compressed.safetensors"

print(f"==================================================================")
print(f"FULL MODEL COMPRESSION: Cosmos DiT (anima_turboV11)")
print(f"Input:  {input_path}")
print(f"Output: {output_path}")
print(f"==================================================================\n", flush=True)

# 10 large 2D matrix targets in Phase 2 (Blocks 4 to 19)
COMPRESSIBLE_SUFFIXES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.output_proj.weight",
    "cross_attn.q_proj.weight",
    "cross_attn.k_proj.weight",
    "cross_attn.v_proj.weight",
    "cross_attn.output_proj.weight",
    "mlp.layer1.weight",
    "mlp.layer2.weight",
]

phase2_blocks = list(range(4, 20)) # 16 core blocks

with safe_open(input_path, framework="pt", device="cpu") as f:
    all_keys = list(f.keys())
    total_keys = len(all_keys)
    print(f"Total tensors in source model: {total_keys}", flush=True)
    
    compressed_dict = {}
    
    # Identify which keys belong to Phase 2 compressible groups
    phase2_compressible_keys = set()
    for block_id in phase2_blocks:
        for suffix in COMPRESSIBLE_SUFFIXES:
            phase2_compressible_keys.add(f"model.diffusion_model.blocks.{block_id}.{suffix}")
            
    # Step 1: Compress Phase 2 matrix groups (Anchor + Quantized Residuals)
    print("\n--- [Step 1/3] Compressing Phase 2 Core Blocks (4 to 19) ---", flush=True)
    for suffix in tqdm(COMPRESSIBLE_SUFFIXES, desc="Compressing Matrix Groups"):
        group_keys = [f"model.diffusion_model.blocks.{i}.{suffix}" for i in phase2_blocks]
        
        # Load all 16 tensors in this group
        tensors = [f.get_tensor(k) for k in group_keys]
        orig_dtype = tensors[0].dtype
        stack = torch.stack([t.float() for t in tensors]) # [16, m, k]
        
        # 1. Compute Shared Group Anchor X*
        anchor = stack.mean(dim=0).to(orig_dtype)
        anchor_key = f"anchors.phase2.{suffix}"
        compressed_dict[anchor_key] = anchor.contiguous()
        
        # 2. Compute Quantized Residuals (INT8) per block
        residuals = stack - anchor.float()
        for idx, block_id in enumerate(phase2_blocks):
            res = residuals[idx]
            max_val = torch.max(torch.abs(res))
            scale = max_val / 127.0
            q_delta = torch.clamp(torch.round(res / (scale + 1e-8)), -128, 127).to(torch.int8)
            
            base_prefix = f"model.diffusion_model.blocks.{block_id}.{suffix[:-7]}" # strip '.weight'
            compressed_dict[f"{base_prefix}.delta"] = q_delta.contiguous()
            compressed_dict[f"{base_prefix}.scale"] = scale.half().contiguous()
            
    print(f"Compressed {len(COMPRESSIBLE_SUFFIXES) * len(phase2_blocks)} large matrices into {len(COMPRESSIBLE_SUFFIXES)} anchors + quantized deltas.", flush=True)

    # Step 2: Copy all other tensors in original precision
    # (Phase 1 blocks 0-3, Phase 3 blocks 20-27, AdaLN modulation, layer norms, embeddings, llm_adapter)
    print("\n--- [Step 2/3] Copying Remaining Tensors (100% Original Precision) ---", flush=True)
    remaining_keys = [k for k in all_keys if k not in phase2_compressible_keys]
    
    for k in tqdm(remaining_keys, desc="Copying Uncompressed Tensors"):
        compressed_dict[k] = f.get_tensor(k).contiguous()

# Step 3: Save final production .safetensors file
print(f"\n--- [Step 3/3] Saving Full Compressed Model to Disk ---", flush=True)
t0 = time.time()
save_file(compressed_dict, output_path)
elapsed = time.time() - t0
print(f"Saved: {output_path} in {elapsed:.1f} seconds!", flush=True)

# Final Size Comparison
in_size_mb = os.path.getsize(input_path) / (1024**2)
out_size_mb = os.path.getsize(output_path) / (1024**2)
savings_mb = in_size_mb - out_size_mb
savings_pct = (1.0 - out_size_mb / in_size_mb) * 100.0

print(f"\n==================================================================")
print(f"COMPRESSION SUMMARY:")
print(f"  Original Model Size:   {in_size_mb:.2f} MB ({in_size_mb/1024:.2f} GB)")
print(f"  Compressed Model Size: {out_size_mb:.2f} MB ({out_size_mb/1024:.2f} GB)")
print(f"  Net VRAM / Disk Saved: {savings_mb:.2f} MB ({savings_mb/1024:.2f} GB)")
print(f"  Overall Reduction:     {savings_pct:.1f}%")
print(f"==================================================================")
