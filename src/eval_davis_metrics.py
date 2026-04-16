import sys; sys.path.append("../extensions/vggt")

from typing import Dict, List

import os
import copy
import json
import argparse
import numpy as np

import torch
from torch import Tensor
from safetensors.torch import load_file

import sys; sys.path.append(".")  # for src modules
from src.options import opt_dict
from src.models import SplatRecon


CKPT_PATH = "resources/movies_ckpt.safetensors"
DATA_DIR = "resources/DAVIS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MoVieS PSNR / SSIM / LPIPS on DAVIS samples")
    parser.add_argument("--name", type=str, nargs="+", default=None, help="One or more DAVIS sample names")
    parser.add_argument("--all", action="store_true", help="Evaluate all DAVIS npz samples in the data directory")
    parser.add_argument("--ckpt-path", type=str, default=CKPT_PATH, help="Path to the MoVieS checkpoint")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Directory containing DAVIS npz files")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output json path")
    parser.add_argument("--summary-json", type=str, default=None, help="Optional summary json path for multi-sample eval")
    return parser.parse_args()


def load_eval_batch(npz_path: str, device: str) -> Dict[str, Tensor]:
    npz = np.load(npz_path)

    images = torch.from_numpy(npz["images"]).float().unsqueeze(0)
    C2W = torch.from_numpy(npz["C2W"]).float().unsqueeze(0)
    fxfycxcy = torch.from_numpy(npz["fxfycxcy"]).float().unsqueeze(0)

    total_frames = images.shape[1]
    H, W = images.shape[-2:]
    timesteps = torch.linspace(0, 1, steps=total_frames).unsqueeze(0)

    dummy_depth = torch.zeros((1, total_frames, H, W), dtype=torch.float32)
    dummy_depth_mask = torch.zeros_like(dummy_depth)
    zero_weight = torch.zeros((1,), dtype=torch.float32)

    batch = {
        "image": images.to(device),
        "C2W": C2W.to(device),
        "fxfycxcy": fxfycxcy.to(device),
        "timestep": timesteps.to(device),
        "depth": dummy_depth.to(device),
        "depth_mask": dummy_depth_mask.to(device),
        "depth_weight": zero_weight.to(device),
        "motion_weight": zero_weight.to(device),
    }
    return batch


def list_davis_samples(data_dir: str) -> List[str]:
    return sorted([
        os.path.splitext(name)[0]
        for name in os.listdir(data_dir)
        if name.endswith(".npz")
    ])


def resolve_sample_names(args: argparse.Namespace) -> List[str]:
    if args.all:
        sample_names = list_davis_samples(args.data_dir)
    elif args.name is not None:
        sample_names = args.name
    else:
        raise ValueError("Specify `--name <sample>` / `--name <sample1> <sample2>` or use `--all`.")

    if len(sample_names) == 0:
        raise ValueError(f"No DAVIS samples found in {args.data_dir}")
    return sample_names


@torch.inference_mode()
def evaluate_sample(model: SplatRecon, opt, npz_path: str, sample_name: str, device: str) -> Dict[str, float]:
    batch = load_eval_batch(npz_path, device)
    total_frames = batch["image"].shape[1]
    if total_frames <= opt.num_output_frames:
        raise ValueError(
            f"Need more than {opt.num_output_frames} frames to evaluate, got {total_frames} in {npz_path}"
        )

    outputs = model.compute_loss(batch)
    return {
        "name": sample_name,
        "num_total_frames": int(total_frames),
        "num_input_frames": int(total_frames - opt.num_output_frames),
        "num_target_frames": int(opt.num_output_frames),
        "metric_definition": "held-out frame reconstruction using first frames as input and last frames as targets",
        "psnr": float(outputs["psnr"].item()),
        "ssim": float(outputs["ssim"].item()),
        "lpips": float(outputs["lpips"].item()),
    }


@torch.inference_mode()
def main(args: argparse.Namespace):
    sample_names = resolve_sample_names(args)
    for sample_name in sample_names:
        npz_path = os.path.join(args.data_dir, f"{sample_name}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Missing DAVIS sample: {npz_path}")

    opt = copy.deepcopy(opt_dict["movies"])
    opt.depth_weight = 0.
    opt.motion_weight = 0.

    model = SplatRecon(opt)
    model.load_state_dict(load_file(args.ckpt_path), strict=True)
    model.eval().to(args.device)

    all_results = []
    for sample_name in sample_names:
        npz_path = os.path.join(args.data_dir, f"{sample_name}.npz")
        result = evaluate_sample(model, opt, npz_path, sample_name, args.device)
        result["checkpoint"] = args.ckpt_path
        all_results.append(result)

        sample_output_json = args.output_json if len(sample_names) == 1 and args.output_json is not None \
            else os.path.join("out", sample_name, "recon_metrics.json")
        os.makedirs(os.path.dirname(sample_output_json), exist_ok=True)
        with open(sample_output_json, "w") as f:
            json.dump(result, f, indent=2)

    if len(all_results) == 1:
        print(json.dumps(all_results[0], indent=2))
        return

    summary = {
        "checkpoint": args.ckpt_path,
        "num_samples": len(all_results),
        "sample_names": sample_names,
        "mean_psnr": float(np.mean([result["psnr"] for result in all_results])),
        "mean_ssim": float(np.mean([result["ssim"] for result in all_results])),
        "mean_lpips": float(np.mean([result["lpips"] for result in all_results])),
        "results": all_results,
    }

    summary_json = args.summary_json or os.path.join("out", "davis_metrics_summary.json")
    os.makedirs(os.path.dirname(summary_json), exist_ok=True)
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(parse_args())
