---
license: other
license_name: proprietary-research
tags:
- text-to-image
- diffusion-transformers
- dit
- cosmos
- compression
- acdm
- int4
- comfyui
base_model: Test_model/anima_turboV11.safetensors
---

# ACDM: Anima Turbo V11 (Block-32 INT4 Anchor-Compressed)

> 🔴 **EXPERIMENTAL RESEARCH CHECKPOINT — NOT FOR COMMERCIAL DEPLOYMENT**  
> *All rights reserved. Research in progress.*

This model is an **ACDM (Anchor-Compressed Delta Model)** checkpoint of `anima_turboV11` (NVIDIA Cosmos DiT architecture).

## Model Details
* **Original Model Size:** 3.90 GB
* **ACDM Compressed Size:** **2.11 GB (45.9% reduction)**
* **Precision Format:** Lossless FP16 Phase Anchors + Micro-Scaled Block-32 INT4 Deltas
* **Cosine Similarity:** > 99.5% across all 240 compressed layers
* **Total Tensors:** 685 / 685 fully intact

## How to Load in ComfyUI
1. Install the custom node: [`ComfyUI_DualStreamCompression`](https://github.com/) into your `ComfyUI/custom_nodes/` directory.
2. Download `anima_turboV11_int4_compressed.safetensors` into `ComfyUI/models/diffusion_models/`.
3. In ComfyUI, add the node **`Load Compressed DiT (Dual-Stream) 🚀`** and connect its output to your **`KSampler`**.
