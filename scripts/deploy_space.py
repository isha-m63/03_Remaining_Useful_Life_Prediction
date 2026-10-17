"""
Deploy the Streamlit demo to a Hugging Face Docker Space.

Usage:
    export HF_TOKEN=hf_xxx            # token with write access
    python scripts/deploy_space.py <hf-username>/<space-name>

Only the files the Docker image needs are uploaded (not the whole repo).
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
FILES = ["Dockerfile", "requirements-demo.txt", "streamlit_app.py"]
DIRS = ["demo_assets"]


def main(repo_id: str) -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in FILES:
            shutil.copy(ROOT / f, tmp / f)
        for d in DIRS:
            shutil.copytree(ROOT / d, tmp / d)
        shutil.copy(ROOT / "deploy" / "space_README.md", tmp / "README.md")
        api.upload_folder(folder_path=str(tmp), repo_id=repo_id, repo_type="space",
                          commit_message="Deploy Streamlit demo")
    print(f"https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
