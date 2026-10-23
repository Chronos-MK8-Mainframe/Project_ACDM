import os
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
    Loads an anchor-compressed DiT model (.safetensors) with Block-32 INT4 / INT8 deltas
    and reconstructs the weights on-the-fly into a standard ComfyUI MODEL object compatible with KSampler.
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

        # 1. Index all phase anchors
        anchors = {}
        for k in list(raw_state_dict.keys()):
            if k.startswith("anchors."):
                anchors[k] = raw_state_dict[k]

        print(f"[DualStreamCompression] Discovered {len(anchors)} shared anchor bases.")

        # 2. Reconstruct all layers
        reconstructed_count = 0
        for k in list(raw_state_dict.keys()):
            if k.startswith("anchors."):
                continue

            # INT4 Block-32 Packed Deltas
            if k.endswith(".delta_int4"):
                packed = raw_state_dict[k]
                base_prefix = k[:-11] # strip '.delta_int4'
                base_weight_key = base_prefix + ".weight"
                scale_key = base_prefix + ".scales_b32"

                parts = base_weight_key.split(".")
                block_id = int(parts[3])
                suffix = ".".join(parts[4:])

                phase_name = "phase2" if (4 <= block_id <= 19) else "phase3"
                anchor_key = f"anchors.{phase_name}.{suffix}"
                anchor = anchors[anchor_key]

                # Fast vectorized 4-bit unpacking
                high = (packed >> 4) & 0x0F
                low = packed & 0x0F
                m_dim, half_k = packed.shape
                k_dim = half_k * 2
                unpacked = torch.stack([high, low], dim=-1).view(m_dim, k_dim)
                q_delta = unpacked.float() - 8.0 # [-8, 7]

                # Block-32 dequantization
                if scale_key in raw_state_dict:
                    scales = raw_state_dict[scale_key].float() # [num_blocks]
                    q_blocked = q_delta.view(-1, 32)
                    res_blocked = q_blocked * scales.unsqueeze(-1)
                    reconstructed_delta = res_blocked.view(m_dim, k_dim)
                else:
                    # Fallback if per-tensor scale
                    fallback_scale = raw_state_dict[base_prefix + ".scale"].float()
                    reconstructed_delta = q_delta * fallback_scale

                reconstructed_weight = (anchor.float() + reconstructed_delta).to(anchor.dtype)
                
                if precision == "bf16":
                    reconstructed_weight = reconstructed_weight.to(torch.bfloat16)
                elif precision == "fp16":
                    reconstructed_weight = reconstructed_weight.to(torch.float16)
                elif precision == "fp32":
                    reconstructed_weight = reconstructed_weight.to(torch.float32)

                reconstructed_dict[base_weight_key] = reconstructed_weight
                reconstructed_count += 1

            # INT8 Deltas (legacy compatibility)
            elif k.endswith(".delta"):
                base_prefix = k[:-6]
                base_weight_key = base_prefix + ".weight"
                scale_key = base_prefix + ".scale"

                parts = base_weight_key.split(".")
                suffix = ".".join(parts[4:])
                anchor_key = f"anchors.phase2.{suffix}"
                anchor = anchors.get(anchor_key, list(anchors.values())[0])

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
                # Direct uncompressed tensor (Phase 1, norms, embeddings, llm_adapter)
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
