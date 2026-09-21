from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "Yu-star/RLMF-m6APred"
MODEL_REVISION = "bd3aeedbd947c3bca4cf24fdfcc767681aca6ada"


def main() -> None:
    snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=PROJECT_ROOT,
        allow_patterns=["pretrained/**", "weights/**"],
    )
    print(f"Downloaded {MODEL_REPO}@{MODEL_REVISION} into {PROJECT_ROOT}")


if __name__ == "__main__":
    main()
