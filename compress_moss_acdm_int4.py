import sys
import os
import time
import re
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

input_path = "Test_model/moss_image_sft_int8_convrot.safetensors"
output_path = "Test_model/moss_image_sft_acdm_int4.safetensors"

print(f"==================================================================")
print(f"HIGH-FIDELITY BLOCK-32 INT4 COMPRESSION: MOSS-Image SFT (5.08B)")
print(f"Input:  {input_path}")
print(f"Output: {output_path}")
print(f"==================================================================\n", flush=True)

BLOCK_SIZE = 32

# 6 projection matrices per layer
PROJ_SUFFIXES = [
    "proj.q",
    "proj.k",
    "proj.v",
    "proj.o",
    "proj.ffn_down",
    "proj.ffn_up_gate",
]

# 42 total layers:
# Phase 1: Layers 0-4 (5 input adaptation layers, kept 100% original precision)
# Phase 2: Layers 5-34 (30 core layers, compressed with Phase 2 anchors)
# Phase 3: Layers 35-41 (7 output refinement layers, compressed with Phase 3 anchors)
phases = {
    "phase2": list(range(5, 35)), # 30 core layers
    "phase3": list(range(35, 42)) # 7 output layers
}

with safe_open(input_path, framework="pt", device="cpu") as f:
    all_keys = list(f.keys())
    total_keys = len(all_keys)
    print(f"Total tensors in source model: {total_keys}", flush=True)

    compressed_dict = {}
    handled_keys = set()

    # Step 1: Compress Phase 2 and Phase 3
    for phase_name, layer_list in phases.items():
        print(f"\n--- Compressing {phase_name.upper()} (Layers {layer_list[0]} to {layer_list[-1]} - {len(layer_list)} layers) ---", flush=True)
        
        for suffix in tqdm(PROJ_SUFFIXES, desc=f"{phase_name.upper()} Projections"):
            # Dequantize source INT8 -> BF16 float weights: W = weight_int8 * weight_scale
            dequantized_weights = []
            for l in layer_list:
                w_key = f"layers.{l}.{suffix}.weight"
                s_key = f"layers.{l}.{suffix}.weight_scale"
                cq_key = f"layers.{l}.{suffix}.comfy_quant"
                
                handled_keys.add(w_key)
                handled_keys.add(s_key)
                handled_keys.add(cq_key)
                
                w_i8 = f.get_tensor(w_key).float()
                s_val = f.get_tensor(s_key).float()
                w_float = (w_i8 * s_val).to(torch.bfloat16)
                dequantized_weights.append(w_float)
                
            stack = torch.stack(dequantized_weights) # [n_layers, m, k]
            m, k_dim = stack.shape[1], stack.shape[2]

            # 1. Compute Phase Anchor X*
            anchor = stack.float().mean(dim=0).to(torch.bfloat16)
            anchor_key = f"anchors.{phase_name}.{suffix}"
            compressed_dict[anchor_key] = anchor.contiguous()

            # 2. Block-32 INT4 Quantization of Residuals
            residuals = stack.float() - anchor.float()
            for idx, layer_id in enumerate(layer_list):
                res = residuals[idx] # [m, k_dim]
                
                # Reshape to [num_blocks, 32]
                res_blocked = res.view(-1, BLOCK_SIZE)
                max_val = torch.max(torch.abs(res_blocked), dim=1, keepdim=True)[0] + 1e-8
                scales = max_val / 7.0 # [-8, 7]
                
                q_blocked = torch.clamp(torch.round(res_blocked / scales), -8, 7).to(torch.int8)
                q_tensor = q_blocked.view(m, k_dim)
                
                # Pack two 4-bit nibbles per byte -> [m, k_dim // 2] uint8
                u = (q_tensor + 8).to(torch.uint8)
                packed_delta = (u[:, 0::2] << 4) | (u[:, 1::2] & 0x0F)
                
                base_prefix = f"layers.{layer_id}.{suffix}"
                compressed_dict[f"{base_prefix}.delta_int4"] = packed_delta.contiguous()
                compressed_dict[f"{base_prefix}.scales_b32"] = scales.squeeze(-1).half().contiguous()

    # Step 2: Copy remaining uncompressed tensors (Phase 1 layers 0-4, layer norms, embeddings, video modulation)
    print("\n--- Copying Untouched Tensors (Phase 1 + Norms + Video Embeddings) ---", flush=True)
    remaining_keys = [k for k in all_keys if k not in handled_keys]
    for k in tqdm(remaining_keys, desc="Copying Untouched"):
        compressed_dict[k] = f.get_tensor(k).contiguous()

# Step 3: Save final compressed model to disk
print(f"\n--- Saving MOSS-Image ACDM INT4 Safetensors Model to Disk ---", flush=True)
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
print(f"MOSS-IMAGE SFT (5.08B) ACDM COMPRESSION COMPLETE:")
print(f"  Original ConvRot INT8 Size: {in_size_mb:.2f} MB ({in_size_mb/1024:.2f} GB)")
print(f"  ACDM Block-32 INT4 Size:   {out_size_mb:.2f} MB ({out_size_mb/1024:.2f} GB)")
print(f"  Net VRAM / Disk Saved:     {savings_mb:.2f} MB ({savings_mb/1024:.2f} GB)")
print(f"  Total Size Reduction:      {savings_pct:.1f}%")
print(f"==================================================================")
