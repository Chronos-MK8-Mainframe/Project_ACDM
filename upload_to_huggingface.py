"""
One-Click Upload Script for Hugging Face Hub
Uploads the ACDM INT4 compressed checkpoint, model card, and test images.

Usage:
  1. Set your token: $env:HF_TOKEN="your_hf_write_token"
  2. Run: python upload_to_huggingface.py --repo your-username/anima_turbo_v11_acdm_int4
"""

import os
import sys
import argparse

def upload(repo_id, token=None, private=False):
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Error: huggingface_hub is required. Install it using: pip install huggingface_hub")
        sys.exit(1)

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        print("\n[!] Hugging Face API Token not found.")
        token = input("Please enter your Hugging Face write token: ").strip()
        if not token:
            print("Token is required to upload. Aborting.")
            sys.exit(1)

    api = HfApi(token=token)
    
    print(f"\n[1/3] Ensuring repository exists: {repo_id} (private={private}) ...")
    create_repo(repo_id=repo_id, repo_type="model", token=token, private=private, exist_ok=True)
    print("      Repository verified!")

    model_file = "Test_model/anima_turboV11_int4_compressed.safetensors"
    if not os.path.exists(model_file):
        print(f"Error: Model file '{model_file}' not found.")
        sys.exit(1)

    print(f"\n[2/3] Uploading Model Card & Preview Images ...")
    if os.path.exists("MODEL_CARD.md"):
        api.upload_file(
            path_or_fileobj="MODEL_CARD.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add ACDM Model Card documentation"
        )
        print("      Uploaded README.md (Model Card)")

    preview_img = "Test_Anima/Anima_int4_compressed_.png"
    if os.path.exists(preview_img):
        api.upload_file(
            path_or_fileobj=preview_img,
            path_in_repo="preview_int4_generation.png",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add ACDM sample generation preview image"
        )
        print("      Uploaded preview_int4_generation.png")

    file_size_gb = os.path.getsize(model_file) / (1024**3)
    print(f"\n[3/3] Uploading ACDM Model Checkpoint: {model_file} ({file_size_gb:.2f} GB) ...")
    print("      (This may take a few minutes depending on your internet upload speed)")
    
    api.upload_file(
        path_or_fileobj=model_file,
        path_in_repo="anima_turboV11_int4_compressed.safetensors",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload Anima Turbo V11 ACDM Block-32 INT4 checkpoint"
    )
    
    print("\n" + "="*70)
    print(f"🎉 SUCCESS! Model successfully published to Hugging Face Hub:")
    print(f"   https://huggingface.co/{repo_id}")
    print("="*70 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="One-Click Hugging Face Uploader for ACDM")
    parser.add_argument("--repo", type=str, default="", help="Hugging Face repo ID (e.g. username/anima-acdm-int4)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face write token (or set HF_TOKEN)")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    args = parser.parse_args()

    repo = args.repo
    if not repo:
        repo = input("Enter target Hugging Face repository ID (e.g. your-username/anima_turbo_acdm_int4): ").strip()
        if not repo:
            print("Repository ID is required. Aborting.")
            sys.exit(1)

    upload(repo_id=repo, token=args.token, private=args.private)
