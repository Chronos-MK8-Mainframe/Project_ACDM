import os
import re
import torch
from safetensors.torch import load_file

try:
    import folder_paths
    import comfy.sd
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False

class DualStreamDiTLoader:
    """
    ComfyUI Custom Node: DualStreamDiTLoader
    Universal Loader for Anchor-Compressed DiT models (ACDM).
    Supports Cosmos DiT (Anima), MOSS-Image (Wan2.1), Flux.1, and SD3.
    Reconstructs Block-32 INT4 / INT8 deltas on-the-fly into standard ComfyUI MODEL objects.
    """
    @classmethod
    def INPUT_TYPES(s):
        files = []
        if COMFY_AVAILABLE:
            try:
                files.extend(folder_paths.get_filename_list("diffusion_models"))
            except Exception:
                pass
            try:
                files.extend(folder_paths.get_filename_list("checkpoints"))
            except Exception:
                pass
        if not files:
            files = [
                "moss_image_sft_acdm_int4.safetensors",
                "anima_turboV11_int4_compressed.safetensors",
                "anima_turboV11_compressed.safetensors", 
                "anima_turboV11.safetensors"
            ]

        return {
            "required": {
                "model_name": (sorted(list(set(files))), ),
                "precision": (["default", "bf16", "fp16", "fp32"], {"default": "default"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_compressed_model"
    CATEGORY = "DualStreamCompression"

    def load_compressed_model(self, model_name, precision="default"):
        # Resolve full path to model file
        if COMFY_AVAILABLE:
            model_path = folder_paths.get_full_path("diffusion_models", model_name)
            if model_path is None:
                model_path = folder_paths.get_full_path("checkpoints", model_name)
        else:
            model_path = model_name

        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_name}")

        print(f"[DualStreamCompression] Loading compressed model from: {model_path} ...")
        raw_state_dict = load_file(model_path)
        reconstructed_dict = {}

        # 1. Detect if model utilizes ComfyUI's ConvRot / comfy_quant runtime
        is_convrot_model = any(".comfy_quant" in k for k in raw_state_dict.keys())
        sample_cq = None
        if is_convrot_model:
            for k in raw_state_dict.keys():
                if k.endswith(".comfy_quant"):
                    sample_cq = raw_state_dict[k]
                    break
            if sample_cq is None:
                cq_json = '{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'
                sample_cq = torch.tensor(list(cq_json.encode('utf-8')), dtype=torch.uint8)
            print("[DualStreamCompression] Detected ConvRot quantization metadata. Re-injecting comfy_quant hooks for stock kernels.")

        # 2. Index all phase anchors
        anchors = {}
        for k in list(raw_state_dict.keys()):
            if k.startswith("anchors."):
                anchors[k] = raw_state_dict[k]

        print(f"[DualStreamCompression] Discovered {len(anchors)} shared anchor bases.")

        # 3. Universal Dynamic Reconstruction
        reconstructed_count = 0
        for k in list(raw_state_dict.keys()):
            if k.startswith("anchors."):
                continue

            # Case A: Block-32 INT4 Packed Deltas
            if k.endswith(".delta_int4"):
                packed = raw_state_dict[k]
                base_prefix = k[:-11] # strip '.delta_int4'
                base_weight_key = base_prefix + ".weight"
                scale_key = base_prefix + ".scales_b32"

                # Universal Regex Parser: matches blocks/layers/double_blocks
                m = re.search(r"(blocks|layers|double_blocks|single_blocks)\.(\d+)\.(.+)", base_prefix)
                if not m:
                    raise ValueError(f"Could not parse layer structure from key: {base_prefix}")
                
                group_type, layer_num_str, suffix = m.groups()
                layer_id = int(layer_num_str)

                # Match anchor dynamically across phases
                matching = [ak for ak in anchors if ak.endswith(suffix) or ak.endswith(suffix + ".weight")]
                if len(matching) > 1:
                    phase_candidates = [ak for ak in matching if ("phase2" in ak and layer_id < 35) or ("phase3" in ak and layer_id >= 20)]
                    anchor_key = phase_candidates[0] if phase_candidates else matching[0]
                elif len(matching) == 1:
                    anchor_key = matching[0]
                else:
                    raise KeyError(f"No matching anchor found for suffix: {suffix}")

                anchor = anchors[anchor_key]

                # Fast vectorized 4-bit unpacking
                high = (packed >> 4) & 0x0F
                low = packed & 0x0F
                m_dim, half_k = packed.shape
                k_dim = half_k * 2
                unpacked = torch.stack([high, low], dim=-1).view(m_dim, k_dim)
                q_delta = unpacked.float() - 8.0 # [-8, 7]

                # Block-32 scale dequantization
                if scale_key in raw_state_dict:
                    scales = raw_state_dict[scale_key].float() # [num_blocks]
                    q_blocked = q_delta.view(-1, 32)
                    res_blocked = q_blocked * scales.unsqueeze(-1)
                    reconstructed_delta = res_blocked.view(m_dim, k_dim)
                else:
                    fallback_scale = raw_state_dict[base_prefix + ".scale"].float()
                    reconstructed_delta = q_delta * fallback_scale

                reconstructed_weight = (anchor.float() + reconstructed_delta).to(anchor.dtype)
                
                # Check target output format
                if is_convrot_model and precision == "default":
                    # Reconstruct into ComfyUI's native INT8 ConvRot layout
                    # This tells ComfyUI to apply online Hadamard rotation to activations!
                    row_scale = (torch.max(torch.abs(reconstructed_weight), dim=-1, keepdim=True)[0] + 1e-8) / 127.0
                    w_i8 = torch.clamp(torch.round(reconstructed_weight / row_scale), -128, 127).to(torch.int8)

                    reconstructed_dict[f"{base_prefix}.weight"] = w_i8
                    reconstructed_dict[f"{base_prefix}.weight_scale"] = row_scale.to(torch.float32)
                    reconstructed_dict[f"{base_prefix}.comfy_quant"] = sample_cq.clone()
                else:
                    if precision == "bf16":
                        reconstructed_weight = reconstructed_weight.to(torch.bfloat16)
                    elif precision == "fp16":
                        reconstructed_weight = reconstructed_weight.to(torch.float16)
                    elif precision == "fp32":
                        reconstructed_weight = reconstructed_weight.to(torch.float32)

                    reconstructed_dict[base_weight_key] = reconstructed_weight

                reconstructed_count += 1

            # Case B: Legacy INT8 Deltas (Anima)
            elif k.endswith(".delta"):
                base_prefix = k[:-6]
                base_weight_key = base_prefix + ".weight"
                scale_key = base_prefix + ".scale"

                m = re.search(r"(blocks|layers|double_blocks|single_blocks)\.(\d+)\.(.+)", base_prefix)
                if m:
                    suffix = m.groups()[2]
                    matching = [ak for ak in anchors if ak.endswith(suffix) or ak.endswith(suffix + ".weight")]
                    anchor_key = matching[0] if matching else list(anchors.keys())[0]
                else:
                    anchor_key = list(anchors.keys())[0]

                anchor = anchors[anchor_key]
                delta = raw_state_dict[k].float()
                scale = raw_state_dict[scale_key].float()

                reconstructed_weight = (anchor.float() + (delta * scale)).to(anchor.dtype)
                if precision == "bf16":
                    reconstructed_weight = reconstructed_weight.to(torch.bfloat16)
                elif precision == "fp16":
                    reconstructed_weight = reconstructed_weight.to(torch.float16)
                elif precision == "fp32":
                    reconstructed_weight = reconstructed_weight.to(torch.float32)

                reconstructed_dict[base_weight_key] = reconstructed_weight
                reconstructed_count += 1

            elif k.endswith(".scales_b32") or k.endswith(".scale"):
                continue
            else:
                # Direct uncompressed tensor (Phase 1, norms, embeddings, video modulation)
                tensor = raw_state_dict[k]
                if precision == "bf16" and tensor.is_floating_point():
                    tensor = tensor.to(torch.bfloat16)
                elif precision == "fp16" and tensor.is_floating_point():
                    tensor = tensor.to(torch.float16)
                elif precision == "fp32" and tensor.is_floating_point():
                    tensor = tensor.to(torch.float32)

                reconstructed_dict[k] = tensor

        print(f"[DualStreamCompression] Successfully reconstructed {reconstructed_count} compressed layers.")
        print(f"[DualStreamCompression] Total tensors in final state_dict: {len(reconstructed_dict)} (100% complete!)")

        if COMFY_AVAILABLE:
            model = comfy.sd.load_diffusion_model_state_dict(reconstructed_dict)
            return (model,)
        else:
            return (reconstructed_dict,)


NODE_CLASS_MAPPINGS = {
    "DualStreamDiTLoader": DualStreamDiTLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DualStreamDiTLoader": "Load Compressed DiT (Dual-Stream) 🚀"
}
