import sys; sys.path.append("../extensions/vggt")

from typing import Dict, Iterable, List, Tuple

import os
import io
import copy
import json
import tarfile
import argparse

import numpy as np
import requests
import torch
import torch.nn.functional as tF
from PIL import Image
from torch import Tensor
from safetensors.torch import load_file
from huggingface_hub import hf_hub_url

import sys; sys.path.append(".")  # for src modules
from src.options import opt_dict
from src.models import SplatRecon


CKPT_PATH = "resources/movies_ckpt.safetensors"
DEFAULT_REPO_ID = "chenguolin/re_tt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MoVieS on a small streamed RealEstate10K subset")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id")
    parser.add_argument("--part", type=str, default="re_tt.part0.tar", help="Dataset tar shard to stream")
    parser.add_argument("--num-videos", type=int, default=3, help="Number of videos to evaluate from the shard")
    parser.add_argument("--ckpt-path", type=str, default=CKPT_PATH, help="Path to the MoVieS checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--output-json", type=str, default="out/re10k_hf_subset_metrics.json", help="Output json path")
    return parser.parse_args()


def iter_videos_from_tar(url: str) -> Iterable[Tuple[str, Dict[str, Dict[str, object]]]]:
    current_uid = None
    current_frames: Dict[str, Dict[str, object]] = {}

    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        tar = tarfile.open(fileobj=response.raw, mode="r|*")
        for member in tar:
            if not member.isfile():
                continue
            if "/" not in member.name:
                continue
            uid, filename = member.name.split("/", 1)
            if current_uid is None:
                current_uid = uid
            elif uid != current_uid:
                yield current_uid, current_frames
                current_uid, current_frames = uid, {}

            frame_name, ext = os.path.splitext(filename)
            frame_entry = current_frames.setdefault(frame_name, {})
            file_obj = tar.extractfile(member)
            if file_obj is None:
                continue
            if ext == ".jpg":
                frame_entry["jpg_bytes"] = file_obj.read()
            elif ext == ".npz":
                npz = np.load(io.BytesIO(file_obj.read()))
                frame_entry["camera_pose"] = npz["camera_pose"]
                frame_entry["fxfycxcy"] = npz["fxfycxcy"]

    if current_uid is not None and current_frames:
        yield current_uid, current_frames


def select_eval_indices(num_frames: int, opt) -> Tuple[List[int], List[int]]:
    max_gap = opt.dataset_args["re10k"]["max_bounded_gap"]
    clip_frame_idxs = np.arange(max_gap if num_frames >= max_gap else num_frames, dtype=int)
    input_indices = clip_frame_idxs[np.linspace(0, len(clip_frame_idxs) - 1, opt.num_input_frames, dtype=int)].tolist()
    output_indices = clip_frame_idxs[np.linspace(0, len(clip_frame_idxs) - 1, opt.num_output_frames, dtype=int)].tolist()
    return input_indices, output_indices


def decode_image(image_bytes: bytes) -> Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1)


def preprocess_images_and_intrinsics(images: Tensor, fxfycxcy: Tensor, opt) -> Tuple[Tensor, Tensor]:
    H, W = images.shape[-2:]
    new_H, new_W = opt.input_res
    aspect_ratio = H / W

    if aspect_ratio <= 1.:
        new_W = int(max(new_H, new_W))
        new_H = int(aspect_ratio * new_W)
    else:
        new_H = int(max(new_H, new_W))
        new_W = int(round(new_H / aspect_ratio))
    new_H = new_H // opt.size_divisor * opt.size_divisor
    new_W = new_W // opt.size_divisor * opt.size_divisor

    scale_factor = max(new_H / (opt.crop_resize_ratio[1] * H), new_W / (opt.crop_resize_ratio[1] * W))
    scaled_H, scaled_W = round(H * scale_factor), round(W * scale_factor)

    images = tF.interpolate(images, size=(scaled_H, scaled_W), mode="bicubic", align_corners=False).clamp(0., 1.)
    top = max((scaled_H - new_H) // 2, 0)
    left = max((scaled_W - new_W) // 2, 0)
    images = images[:, :, top:top + new_H, left:left + new_W]

    fxfycxcy = fxfycxcy.clone()
    fxfycxcy[:, 0] *= (scaled_W / new_W)
    fxfycxcy[:, 1] *= (scaled_H / new_H)
    return images, fxfycxcy


def build_batch(frames: Dict[str, Dict[str, object]], opt, device: str) -> Dict[str, Tensor]:
    frame_names = sorted(frames.keys())
    input_indices, output_indices = select_eval_indices(len(frame_names), opt)
    selected_indices = input_indices + output_indices

    images, poses, intrinsics = [], [], []
    for idx in selected_indices:
        frame = frames[frame_names[idx]]
        images.append(decode_image(frame["jpg_bytes"]))
        poses.append(torch.from_numpy(frame["camera_pose"]).float())
        intrinsics.append(torch.from_numpy(frame["fxfycxcy"]).float())

    images = torch.stack(images, dim=0)
    C2W = torch.stack(poses, dim=0)
    fxfycxcy = torch.stack(intrinsics, dim=0)
    images, fxfycxcy = preprocess_images_and_intrinsics(images, fxfycxcy, opt)

    timesteps = torch.tensor(selected_indices, dtype=torch.float32)
    timesteps = (timesteps - timesteps.min()) / (timesteps.max() - timesteps.min())

    total_frames = images.shape[0]
    dummy_depth = torch.zeros((1, total_frames, images.shape[-2], images.shape[-1]), dtype=torch.float32, device=device)
    zero_weight = torch.zeros((1,), dtype=torch.float32, device=device)

    return {
        "image": images.unsqueeze(0).to(device),
        "C2W": C2W.unsqueeze(0).to(device),
        "fxfycxcy": fxfycxcy.unsqueeze(0).to(device),
        "timestep": timesteps.unsqueeze(0).to(device),
        "depth": dummy_depth,
        "depth_mask": dummy_depth,
        "depth_weight": zero_weight,
        "motion_weight": zero_weight,
    }


def evaluate_video(model: SplatRecon, frames: Dict[str, Dict[str, object]], uid: str, opt, device: str) -> Dict[str, object]:
    input_indices, output_indices = select_eval_indices(len(frames), opt)
    batch = build_batch(frames, opt, device)

    with torch.inference_mode():
        outputs = model.compute_loss(batch)

    return {
        "uid": uid,
        "num_frames_available": len(frames),
        "num_input_frames": opt.num_input_frames,
        "num_target_frames": opt.num_output_frames,
        "input_indices": input_indices,
        "target_indices": output_indices,
        "psnr": float(outputs["psnr"].item()),
        "ssim": float(outputs["ssim"].item()),
        "ssim_x100": float(outputs["ssim"].item() * 100.0),
        "lpips": float(outputs["lpips"].item()),
    }


@torch.inference_mode()
def main(args: argparse.Namespace):
    opt = copy.deepcopy(opt_dict["movies"])
    opt.depth_weight = 0.
    opt.motion_weight = 0.

    model = SplatRecon(opt)
    model.load_state_dict(load_file(args.ckpt_path), strict=True)
    model.eval().to(args.device)

    tar_url = hf_hub_url(args.repo_id, args.part, repo_type="dataset")

    results = []
    for uid, frames in iter_videos_from_tar(tar_url):
        if len(results) >= args.num_videos:
            break
        if len(frames) < (opt.num_input_frames + opt.num_output_frames):
            continue
        results.append(evaluate_video(model, frames, uid, opt, args.device))

    if len(results) == 0:
        raise RuntimeError(f"No valid videos were evaluated from {tar_url}")

    summary = {
        "checkpoint": args.ckpt_path,
        "dataset_repo": args.repo_id,
        "dataset_part": args.part,
        "num_videos_evaluated": len(results),
        "metric_definition": "held-out frame reconstruction on streamed RealEstate10K clips using embedded per-frame poses",
        "sampling_protocol": "match Re10kDataset eval-style uniform frame selection from the first max_gap clip",
        "preprocessing_note": "uses the same resize/center-crop intrinsics adjustment as BaseDataset eval; depth and motion supervision are disabled",
        "mean_psnr": float(np.mean([result["psnr"] for result in results])),
        "mean_ssim": float(np.mean([result["ssim"] for result in results])),
        "mean_ssim_x100": float(np.mean([result["ssim_x100"] for result in results])),
        "mean_lpips": float(np.mean([result["lpips"] for result in results])),
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(parse_args())
