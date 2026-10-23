import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from safetensors import safe_open

model_path = "Test_model/anima_turboV11.safetensors"
print(f"Inspecting metadata & keys from: {model_path} ...")

with safe_open(model_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    print(f"Total tensors in file: {len(keys)}")
    print("\nSample keys (first 25):")
    for k in keys[:25]:
        tensor_slice = f.get_slice(k)
        print(f"  {k} -> shape: {tensor_slice.get_shape()}, dtype: {tensor_slice.get_dtype()}")

    # Find common patterns in keys (e.g. blocks, double_blocks, single_blocks, unet, dit, etc.)
    prefixes = set()
    for k in keys:
        parts = k.split('.')
        if len(parts) > 2:
            prefixes.add('.'.join(parts[:3]))
    
    print("\nCommon key prefixes:")
    for p in sorted(list(prefixes))[:20]:
        print(f"  {p}")
