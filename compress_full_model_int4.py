import sys
import os
import time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

input_path = "Test_model/anima_turboV11.safetensors"
output_path = "Test_model/anima_turboV11_int4_compressed.safetensors"

print(f"==================================================================")
print(f"HIGH-FIDELITY BLOCK-32 INT4 COMPRESSION: Cosmos DiT")
print(f"Input:  {input_path}")
print(f"Output: {output_path}")
print(f"==================================================================\n", flush=True)

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

phases = {
    "phase2": list(range(4, 20)), # 16 core blocks
    "phase3": list(range(20, 28)) # 8 output blocks
}

BLOCK_SIZE = 32

with safe_open(input_path, framework="pt", device="cpu") as f:
    all_keys = list(f.keys())
    total_keys = len(all_keys)
    print(f"Total tensors in source model: {total_keys}", flush=True)
    
    compressed_dict = {}
    compressible_keys = set()

    for phase_name, block_list in phases.items():
        print(f"\n--- Compressing {phase_name.upper()} (Blocks {block_list[0]} to {block_list[-1]} - {len(block_list)} blocks) ---", flush=True)
        
        for suffix in tqdm(COMPRESSIBLE_SUFFIXES, desc=f"{phase_name.upper()} Groups"):
            group_keys = [f"model.diffusion_model.blocks.{i}.{suffix}" for i in block_list]
            for k in group_keys:
                compressible_keys.add(k)
                
            tensors = [f.get_tensor(k) for k in group_keys]
            orig_dtype = tensors[0].dtype
            stack = torch.stack([t.float() for t in tensors]) # [n_blocks, m, k]
            m, k_dim = stack.shape[1], stack.shape[2]
            
            # 1. Compute Phase Anchor X*
            anchor = stack.mean(dim=0).to(orig_dtype)
            anchor_key = f"anchors.{phase_name}.{suffix}"
            compressed_dict[anchor_key] = anchor.contiguous()
            
            # 2. Block-32 INT4 Quantization of Residuals
            residuals = stack - anchor.float()
            for idx, block_id in enumerate(block_list):
                res = residuals[idx] # [m, k_dim]
                
                # Reshape into blocks of 32 elements: [num_blocks, 32]
                res_blocked = res.view(-1, BLOCK_SIZE)
                max_val = torch.max(torch.abs(res_blocked), dim=1, keepdim=True)[0] + 1e-8
                scales = max_val / 7.0 # [num_blocks, 1]
                
                # Quantize to [-8, 7]
                q_blocked = torch.clamp(torch.round(res_blocked / scales), -8, 7).to(torch.int8)
                q_tensor = q_blocked.view(m, k_dim) # [m, k_dim]
                
                # Pack two 4-bit nibbles per byte -> [m, k_dim // 2] uint8
                u = (q_tensor + 8).to(torch.uint8)
                packed_delta = (u[:, 0::2] << 4) | (u[:, 1::2] & 0x0F)
                
                base_prefix = f"model.diffusion_model.blocks.{block_id}.{suffix[:-7]}"
                compressed_dict[f"{base_prefix}.delta_int4"] = packed_delta.contiguous()
                # Store scales as FP16 vector: [num_blocks]
                compressed_dict[f"{base_prefix}.scales_b32"] = scales.squeeze(-1).half().contiguous()

    # Step 2: Copy remaining uncompressed tensors (Phase 1, norms, embeddings, llm_adapter)
    print("\n--- Copying Uncompressed Tensors (Phase 1 + Norms + Embeddings) ---", flush=True)
    remaining_keys = [k for k in all_keys if k not in compressible_keys]
    for k in tqdm(remaining_keys, desc="Copying Untouched Tensors"):
        compressed_dict[k] = f.get_tensor(k).contiguous()

# Step 3: Save to disk
print(f"\n--- Saving High-Fidelity INT4 Safetensors Model to Disk ---", flush=True)
t0 = time.time()
save_file(compressed_dict, output_path)
elapsed = time.time() - t0
print(f"Saved: {output_path} in {elapsed:.1f} seconds!", flush=True)

# Final Size Statistics
in_size_mb = os.path.getsize(input_path) / (1024**2)
out_size_mb = os.path.getsize(output_path) / (1024**2)
savings_mb = in_size_mb - out_size_mb
savings_pct = (1.0 - out_size_mb / in_size_mb) * 100.0

print(f"\n==================================================================")
print(f"BLOCK-32 INT4 COMPRESSION COMPLETE:")
print(f"  Original Model Size:   {in_size_mb:.2f} MB ({in_size_mb/1024:.2f} GB)")
print(f"  Compressed Model Size: {out_size_mb:.2f} MB ({out_size_mb/1024:.2f} GB)")
print(f"  Net VRAM / Disk Saved: {savings_mb:.2f} MB ({savings_mb/1024:.2f} GB)")
print(f"  Total Size Reduction:  {savings_pct:.1f}%")
print(f"==================================================================")
