# ACDM: Anchor-Compressed Delta Models for Diffusion Transformers

<div align="center">

# 🔴 RESEARCH IN PROGRESS — EXPERIMENTAL WORK 🔴
### ⚠️ DO NOT USE IN PRODUCTION. THIS IS AN ACTIVE RESEARCH PROJECT AND WILL TAKE TIME TO PROGRESS. ⚠️

***All Rights Reserved. No open-source license is granted. Any use, commercial distribution, reproduction, or deployment of this work without explicit written permission from the author/host is strictly prohibited.***

</div>

---

## 📌 Executive Summary

**ACDM (Anchor-Compressed Delta Models)** is a novel structural compression architecture designed specifically for deep **Diffusion Transformers (DiTs)** (such as NVIDIA Cosmos, Flux.1, and Stable Diffusion 3).

Unlike traditional quantization methods (such as standard AWQ, GPTQ, or GGUF Q4) that treat every weight matrix in isolation, **ACDM exploits cross-layer depth redundancy**:
1. **Dynamic Phase Anchors ($X^*$):** Extracts shared, high-precision (FP16/BF16) geometric foundation anchors across coherent depth regimes of the transformer.
2. **Micro-Scaled Residual Deltas ($\Delta_i$):** Quantizes only the zero-centered layer-specific variation using **Block-32 INT4 micro-quantization** (4 bits per weight with 32-element grouping).
3. **Pristine I/O Protection:** Keeps sensitive input patch adapters and output decoders at 100% original precision to avoid prompt comprehension drift and visual collapse.

Tested empirically on the 4-Billion parameter **NVIDIA Cosmos DiT architecture (`anima_turboV11`)**, ACDM cuts the model size **from 3.90 GB down to 2.11 GB (45.9% VRAM reduction)** while maintaining **>99.5% cosine similarity** and producing crisp, artifact-free visual generations.

---

## 🔬 Mathematical Formulation

Given $n$ weight matrices $A_1, A_2, \dots, A_n \in \mathbb{R}^{m \times k}$ across a functional transformer phase:

$$X^* = \frac{1}{n} \sum_{i=1}^{n} A_i$$

$$\Delta_i = A_i - X^*$$

$$\hat{A}_i = X^*_{\text{FP16}} + \mathcal{Q}_4(\Delta_i)$$

Where $\mathcal{Q}_4(\cdot)$ represents signed 4-bit block quantization with per-32-element dynamic scaling:
$$\mathcal{Q}_4(x) = \text{clamp}\left(\text{round}\left(\frac{x}{S_{\text{block}}}\right), -8, 7\right) \times S_{\text{block}}$$

Because the high-energy DC structural component ($X^*$) is preserved in **lossless FP16**, the 4-bit rounding noise is strictly zero-mean high-frequency perturbation, which the diffusion model's iterative denoising process naturally filters out.

---

## 📊 Empirical Benchmarks (NVIDIA Cosmos DiT)

| Metric | Raw Uncompressed (BF16) | Standard INT8 Deltas | **ACDM Block-32 INT4 (Ours)** |
| :--- | :---: | :---: | :---: |
| **Model Disk Size** | 3.90 GB (3,988 MB) | 3.07 GB (3,148 MB) | **2.11 GB (2,158 MB)** |
| **VRAM / Disk Saved** | 0.0 MB | 840 MB (21.1%) | **1,830 MB (45.9% Saved)** |
| **Cosine Similarity (Core)**| 100.0% | 99.96% | **99.52%** |
| **Cosine Similarity (Output)**| 100.0% | 100.0% | **99.68%** |
| **Total Reconstructed Keys** | 685 / 685 | 685 / 685 | **685 / 685 (100% Complete)** |
| **Zero-Frequency Drift** | None | None | **Zero (Anchor is exact)** |

---

## 📥 Download Compressed Model (ACDM Checkpoint)

> [!IMPORTANT]
> **Model Weights Download Link:**
> 
> 🔗 **[CLICK HERE TO DOWNLOAD ACDM MODEL WEIGHTS (PASTE LINK HERE)](<!-- PASTE_YOUR_MODEL_LINK_HERE -->)**
> 
> *(Once downloaded, place `anima_turboV11_int4_compressed.safetensors` inside `ComfyUI/models/diffusion_models/`)*

---

## 🖼️ Visual Comparison & Inference Results

Side-by-side test generations using the exact same prompt and initial seed:

| Original Uncompressed (`3.90 GB`) | ACDM INT4 Compressed (`2.11 GB`) |
| :---: | :---: |
| <img src="Test_Anima/Anima_OG_Uncompressed_.png" width="400" /> | <img src="Test_Anima/Anima_int4_compressed_.png" width="400" /> |
| *Original baseline generation* | *ACDM INT4: Sharp lines, zero color banding, zero face distortion* |

### Notes on Generation Dynamics:
* **Visual Fidelity:** Notice the sunset lighting, sky gradients, hair strands, and lighting reflections in the ACDM generation. Unlike raw INT4 which causes muddy color banding and facial melting, ACDM preserves clean gradients and razor-sharp line art.
* **Trajectory Divergence:** Due to the Butterfly Effect in numerical ODE/SDE integration across 20-30 sampling steps, a 2% weight variation causes the trajectory to settle into a neighboring local attractor on the prompt manifold (e.g. collared shirt + jacket vs. casual top).

---

## 🚀 ComfyUI Custom Node Integration

A complete, working ComfyUI custom node package is included in [`ComfyUI_DualStreamCompression/`](./ComfyUI_DualStreamCompression).

### Installation:
1. Copy the `ComfyUI_DualStreamCompression` folder into your `ComfyUI/custom_nodes/` directory:
   ```bash
   cp -r ComfyUI_DualStreamCompression /path/to/ComfyUI/custom_nodes/
   ```
2. Place the compressed checkpoint (`anima_turboV11_int4_compressed.safetensors`) into:
   ```bash
   ComfyUI/models/diffusion_models/
   ```
3. In ComfyUI, add the node:
   **`Load Compressed DiT (Dual-Stream) 🚀`**
4. Connect the **`MODEL`** output wire directly to your standard **`KSampler`**.

---

## 🛠️ Repository Contents

* `ComfyUI_DualStreamCompression/`: The ready-to-use ComfyUI custom node package.
* `compress_full_model_int4.py`: The production compression engine for converting models into ACDM format.
* `verify_int4_reconstruction.py`: Rigorous verification script validating all 685 tensors and cosine similarity.
* `test_cuda_inference.py`: PyTorch script verifying GPU CUDA execution.
* `detect_layer_phases.py`: Automated change-point detection algorithm identifying natural depth regimes.
* `upload_to_huggingface.py`: One-click utility script to stage and publish the ACDM model to Hugging Face Hub.

---

## 📋 Road Map & Future To-Dos

- [ ] 🔴 **[HIGH PRIORITY] Universal ACDM ComfyUI Node Pack:**  
  Develop a fully interactive ComfyUI node pack capable of taking **ANY arbitrary diffusion model** (Flux.1, SD3, Wan2.1, SDXL) directly within the ComfyUI canvas and automatically converting it into an optimized ACDM `.safetensors` file.  
  *(Current Status: Concept Stage — Nowhere near started or completed; will require significant algorithmic automation and research).*
- [ ] **On-the-Fly Fused Triton / CUDA Dequantization Kernel:**  
  Write a custom GPU kernel that pins the shared FP16 anchor in L2 Cache / SRAM and streams INT4 deltas directly to Tensor Cores, eliminating any RAM reconstruction overhead.
- [ ] **Scaling to 12B–14B Architectures:**  
  Apply ACDM to massive models (Flux.1 12B and Wan2.1 14B Video) to allow them to fit entirely inside consumer 12GB–16GB VRAM GPUs without PCIe CPU offloading.
- [ ] **Automated Quality Benchmark Suite:**  
  Run standardized ImageReward, PickScore, and CLIP aesthetic evals across 1,000 prompts comparing ACDM against standard GGUF Q4 and uncompressed FP16.

---

## ⚖️ License & Terms of Use

**NO LICENSE GRANTED.**

Copyright © 2026. All rights reserved.

This repository, its architectural designs, algorithms, code, and documentation are strictly proprietary research materials. No permission is granted to copy, distribute, modify, reverse engineer, sublicense, or use this work for any purpose (commercial or non-commercial) without explicit prior written authorization from the author/host.
