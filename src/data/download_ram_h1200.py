"""Download and audit the official RAM-H1200 dataset from Hugging Face.

Repository: https://huggingface.co/datasets/TokyoTechMagicYang/RAM-H1200-v1
Benchmark Code: https://github.com/YSongxiao/RAM-H1200
License: CC BY-NC-SA 4.0
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict

from src.data.generate_sample_fixtures import generate_ram_h1200_fixtures

HF_DATASET_ID = "TokyoTechMagicYang/RAM-H1200-v1"


def fetch_remote_dataset_manifest(repo_id: str = HF_DATASET_ID) -> Dict[str, Any]:
    """Fetch public dataset manifest from Hugging Face API and verify splits & patient separation."""
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    req = urllib.request.urlopen(url, timeout=15)
    data = json.loads(req.read().decode("utf-8"))
    files = [s["rfilename"] for s in data.get("siblings", [])]

    split_stats: Dict[str, Any] = {}
    split_patients: Dict[str, set] = {}
    for split in ("train", "val", "test"):
        bmps = [
            f.split("/")[-1]
            for f in files
            if f.startswith(f"Segmentation/{split}/") and f.endswith(".bmp")
        ]
        patients = {"_".join(b.split("_")[:3]) for b in bmps}
        centers = sorted({b.split("_")[1] for b in bmps if "_" in b})
        split_patients[split] = patients
        split_stats[split] = {
            "num_images": len(bmps),
            "num_unique_patients": len(patients),
            "centers": centers,
        }

    overlap_train_val = len(split_patients["train"] & split_patients["val"])
    overlap_train_test = len(split_patients["train"] & split_patients["test"])
    overlap_val_test = len(split_patients["val"] & split_patients["test"])
    all_patients = split_patients["train"] | split_patients["val"] | split_patients["test"]

    return {
        "repo_id": repo_id,
        "gated": data.get("gated", "auto"),
        "total_repository_files": len(files),
        "total_full_hand_radiographs": sum(s["num_images"] for s in split_stats.values()),
        "total_unique_patients": len(all_patients),
        "splits": split_stats,
        "patient_overlap": {
            "train_val": overlap_train_val,
            "train_test": overlap_train_test,
            "val_test": overlap_val_test,
            "zero_leakage_verified": (overlap_train_val + overlap_train_test + overlap_val_test) == 0,
        },
    }


def download_ram_h1200(
    target_dir: str | Path = "dataset/raw",
    token: str | None = None,
    fallback_to_fixtures: bool = True,
) -> Dict[str, Any]:
    """Download RAM-H1200-v1 via huggingface_hub or fall back to RAM-H1200 local fixtures."""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    manifest: Dict[str, Any] = {}
    try:
        manifest = fetch_remote_dataset_manifest(HF_DATASET_ID)
        manifest_out = Path("dataset/annotations/ram_h1200_remote_audit.json")
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception as exc:
        manifest = {"warning": f"Could not query live HF manifest: {exc}"}

    if hf_token:
        try:
            from huggingface_hub import snapshot_download

            downloaded_path = snapshot_download(
                repo_id=HF_DATASET_ID,
                repo_type="dataset",
                local_dir=str(target_path),
                token=hf_token,
            )
            return {
                "status": "downloaded_from_huggingface",
                "local_dir": downloaded_path,
                "remote_audit": manifest,
            }
        except Exception as exc:
            if not fallback_to_fixtures:
                raise
            fixture_info = generate_ram_h1200_fixtures(".")
            return {
                "status": "fallback_fixtures_generated",
                "reason": f"Hugging Face download failed ({exc}). Generated RAM-H1200 fixtures.",
                "fixture_summary": fixture_info,
                "remote_audit": manifest,
            }

    if fallback_to_fixtures:
        fixture_info = generate_ram_h1200_fixtures(".")
        return {
            "status": "fallback_fixtures_generated",
            "reason": (
                "TokyoTechMagicYang/RAM-H1200-v1 is a gated Hugging Face dataset (Gated: auto). "
                "Set HF_TOKEN after accepting the license at "
                "https://huggingface.co/datasets/TokyoTechMagicYang/RAM-H1200-v1 to download the full 2.75 GB dataset. "
                "Generated RAM-H1200-compatible local fixtures for immediate training and testing."
            ),
            "fixture_summary": fixture_info,
            "remote_audit": manifest,
        }

    raise RuntimeError("HF_TOKEN not provided for gated dataset TokyoTechMagicYang/RAM-H1200-v1.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or initialize RAM-H1200 dataset")
    parser.add_argument("--target-dir", default="dataset/raw", help="Target directory for raw dataset")
    parser.add_argument("--token", default=None, help="Hugging Face access token")
    parser.add_argument("--no-fallback", action="store_true", help="Fail instead of generating fixtures")
    args = parser.parse_args()

    result = download_ram_h1200(
        target_dir=args.target_dir,
        token=args.token,
        fallback_to_fixtures=not args.no_fallback,
    )
    print(json.dumps(result, indent=2))
