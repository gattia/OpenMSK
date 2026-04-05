#!/usr/bin/env python3
"""
Download NSM models from HuggingFace Hub.
The repository is gated and requires authentication.

Usage:
    python download_nsm_models.py --token YOUR_HF_TOKEN

Or set the HF_TOKEN environment variable:
    export HF_TOKEN=YOUR_HF_TOKEN
    python download_nsm_models.py
"""

import argparse
import os
from pathlib import Path


def download_nsm_models(token=None, local_dir="./NSM_MODELS"):
    """Download NSM models from HuggingFace Hub."""
    
    try:
        from huggingface_hub import snapshot_download, login
    except ImportError:
        print("❌ huggingface_hub not installed. Install with:")
        print("   pip install huggingface_hub")
        return False
    
    # Get token from argument or environment
    if token is None:
        token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    
    if not token:
        print("❌ HuggingFace token required for gated repository.")
        print("\nTo get a token:")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Create a new token with 'read' access")
        print("3. Accept the terms at https://huggingface.co/aagatti/ShapeMedKnee")
        print("\nThen run:")
        print("   python download_nsm_models.py --token YOUR_TOKEN")
        print("\nOr set the environment variable:")
        print("   export HF_TOKEN=YOUR_TOKEN")
        print("   python download_nsm_models.py")
        return False
    
    print(f"📥 Downloading NSM models from aagatti/ShapeMedKnee...")
    print(f"📁 Target directory: {local_dir}")
    
    # Create directory if it doesn't exist
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Login with token
        login(token=token)
        
        snapshot_download(
            repo_id="aagatti/ShapeMedKnee",
            local_dir=local_dir,
            repo_type="model"
        )
        
        print("✅ Download completed successfully!")
        print(f"📂 NSM models are now available in: {local_dir}/")
        
        # List downloaded contents
        print("\nDownloaded files:")
        for root, dirs, files in os.walk(local_dir):
            level = root.replace(local_dir, "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files[:5]:
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\nMake sure you have:")
        print("1. Accepted the terms at https://huggingface.co/aagatti/ShapeMedKnee")
        print("2. A valid HuggingFace token with read access")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download NSM models from HuggingFace (requires authentication)"
    )
    parser.add_argument(
        "--token",
        help="HuggingFace access token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--dir",
        default="./NSM_MODELS",
        help="Local directory to download to (default: ./NSM_MODELS)"
    )
    
    args = parser.parse_args()
    
    success = download_nsm_models(args.token, args.dir)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()





