import sys; sys.path.append("../extensions/vggt")

from typing import Dict, List, Tuple

import argparse
import copy
import csv
import json
import os
import time
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np
import torch
from einops import rearrange
from lpips import LPIPS
from pytorch_msssim import ssim as SSIM
from safetensors.torch import load_file
from torch import Tensor

import sys; sys.path.append(".")
from src.options import opt_dict
from src.models import SplatRecon
from src.utils import inverse_c2w, tensor_to_video


CKPT_PATH = "resources/movies_ckpt.safetensors"
PATCH_SIZE = 14
SCENES = [
    "Jumping",
    "Skating",
    "Truck",
    "DynamicFace",
    "Umbrella",
    "Balloon1",
    "Balloon2",
    "Teadybear",
    "Playground",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MoVieS on the NVIDIA dynamic scene benchmark")
    parser.add_argument("--scene", type=str, nargs="+", choices=SCENES, help="One or more benchmark scenes")
    parser.add_argument("--all-scenes", action="store_true", help="Evaluate all benchmark scenes")
    parser.add_argument("--data-root", type=str, default="datasets/nvidia", help="Root directory containing extracted scenes")
    parser.add_argument("--ckpt-path", type=str, default=CKPT_PATH, help="Path to the MoVieS checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--image-height", type=int, default=379, help="Evaluation height")
    parser.add_argument("--image-width", type=int, default=672, help="Evaluation width")
    parser.add_argument(
        "--frames-chunk-size",
        type=int,
        default=16,
        help="Chunk size passed to the VGGT / DPT heads during inference",
    )
    parser.add_argument(
        "--eval-camera-id",
        type=int,
        default=1,
        choices=list(range(1, 13)),
        help="Fixed target camera id for evaluation. The canonical protocol uses camera 1.",
    )
    parser.add_argument("--save-videos", action="store_true", help="Save input / prediction / GT videos for each scene")
    parser.add_argument("--output-json", type=str, default=None, help="Optional aggregate json path")
    return parser.parse_args()


def resolve_scenes(args: argparse.Namespace) -> List[str]:
    if args.all_scenes:
        return SCENES
    if args.scene:
        return args.scene
    raise ValueError("Specify --scene <name> ... or --all-scenes")


def default_scene_output_json(scene: str, eval_camera_id: int) -> str:
    return os.path.join("out", "nvidia", scene, f"nvidia_metrics_cam{eval_camera_id:02d}.json")


def default_aggregate_output_json(eval_camera_id: int) -> str:
    return os.path.join("out", "nvidia", f"nvidia_metrics_all_cam{eval_camera_id:02d}.json")


def snap_to_patch_multiple(size: int, patch_size: int = PATCH_SIZE) -> int:
    snapped = (size // patch_size) * patch_size
    if snapped <= 0:
        raise ValueError(f"Invalid image size {size}; must be >= {patch_size}")
    return snapped


def resolve_multiview_dir(scene_dir: str) -> str:
    multiview_dir = os.path.join(scene_dir, "multiview_GT")
    nested_multiview_dir = os.path.join(multiview_dir, "multiview_GT")
    if os.path.isdir(nested_multiview_dir):
        return nested_multiview_dir
    return multiview_dir


def apply_world_transform(C2W: Tensor, transform: Tensor) -> Tensor:
    return transform.unsqueeze(0) @ C2W


def read_matrix(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32)


def load_camera(scene_dir: str, camera_id: int) -> Tuple[np.ndarray, np.ndarray]:
    camera_dir = os.path.join(scene_dir, "calibration", f"cam{camera_id:02d}")
    intrinsic = read_matrix(os.path.join(camera_dir, "intrinsic.txt"))
    extrinsic = read_matrix(os.path.join(camera_dir, "extrinsic.txt"))

    center = extrinsic[0]
    rotation = extrinsic[1:]

    C2W = np.eye(4, dtype=np.float32)
    C2W[:3, :3] = rotation.T
    C2W[:3, 3] = center
    return C2W, intrinsic


def resize_image_and_intrinsics(
    image: np.ndarray,
    intrin: np.ndarray,
    output_height: int,
    output_width: int,
) -> Tuple[Tensor, np.ndarray]:
    input_height, input_width = image.shape[:2]
    scale_x = output_width / input_width
    scale_y = output_height / input_height

    image = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_AREA)
    intrin = intrin.copy()
    intrin[0, :] *= scale_x
    intrin[1, :] *= scale_y
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return image, intrin


def intrin_to_fxfycxcy(intrin: np.ndarray, image_height: int, image_width: int) -> Tensor:
    return torch.tensor(
        [
            intrin[0, 0] / image_width,
            intrin[1, 1] / image_height,
            intrin[0, 2] / image_width,
            intrin[1, 2] / image_height,
        ],
        dtype=torch.float32,
    )


def load_source_stack(scene_dir: str, output_height: int, output_width: int) -> Tuple[Tensor, Tensor, Tensor]:
    images, C2Ws, intrinsics = [], [], []
    for camera_id in range(1, 13):
        image_path = os.path.join(scene_dir, "input_images", f"cam{camera_id:02d}.jpg")
        image = iio.imread(image_path)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]

        C2W, intrin = load_camera(scene_dir, camera_id)
        image, intrin = resize_image_and_intrinsics(image, intrin, output_height, output_width)

        images.append(image)
        C2Ws.append(torch.from_numpy(C2W).float())
        intrinsics.append(intrin_to_fxfycxcy(intrin, output_height, output_width))

    return torch.stack(images, dim=0), torch.stack(C2Ws, dim=0), torch.stack(intrinsics, dim=0)


def load_target_stack(
    scene_dir: str,
    camera_id: int,
    output_height: int,
    output_width: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    images, C2Ws, intrinsics = [], [], []
    C2W, intrin = load_camera(scene_dir, camera_id)
    multiview_dir = resolve_multiview_dir(scene_dir)
    for timestep in range(1, 13):
        image_path = os.path.join(multiview_dir, f"{timestep:08d}", f"cam{camera_id:02d}.jpg")
        image = iio.imread(image_path)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]

        image_resized, intrin_resized = resize_image_and_intrinsics(image, intrin, output_height, output_width)
        images.append(image_resized)
        C2Ws.append(torch.from_numpy(C2W).float())
        intrinsics.append(intrin_to_fxfycxcy(intrin_resized, output_height, output_width))

    return torch.stack(images, dim=0), torch.stack(C2Ws, dim=0), torch.stack(intrinsics, dim=0)


def verify_round_robin_alignment(scene_dir: str) -> Dict[str, object]:
    best_matching_time_ids = []
    best_mse = []
    multiview_dir = resolve_multiview_dir(scene_dir)

    for camera_id in range(1, 13):
        source = iio.imread(os.path.join(scene_dir, "input_images", f"cam{camera_id:02d}.jpg")).astype(np.float32)
        if source.ndim == 2:
            source = np.repeat(source[..., None], 3, axis=-1)
        if source.shape[-1] == 4:
            source = source[..., :3]

        current_best_timestep = None
        current_best_mse = None
        for timestep in range(1, 13):
            target = iio.imread(
                os.path.join(multiview_dir, f"{timestep:08d}", f"cam{camera_id:02d}.jpg")
            ).astype(np.float32)
            if target.ndim == 2:
                target = np.repeat(target[..., None], 3, axis=-1)
            if target.shape[-1] == 4:
                target = target[..., :3]
            mse = float(np.mean((source - target) ** 2))
            if current_best_mse is None or mse < current_best_mse:
                current_best_mse = mse
                current_best_timestep = timestep

        best_matching_time_ids.append(int(current_best_timestep))
        best_mse.append(float(current_best_mse))

    expected_time_ids = list(range(1, 13))
    return {
        "expected_time_ids": expected_time_ids,
        "best_matching_time_ids": best_matching_time_ids,
        "all_best_matches_follow_round_robin_diagonal": best_matching_time_ids == expected_time_ids,
        "best_match_mse_per_camera": best_mse,
        "mean_best_match_mse": float(np.mean(best_mse)),
    }


def get_frame_model_outputs(
    backbone_outputs: Dict[str, Tensor],
    pred_motions: Tensor | None = None,
    pred_motion_gs: List[Dict[str, Tensor]] | None = None,
    frame_idx: int | None = None,
) -> Dict[str, Tensor]:
    model_outputs = dict(backbone_outputs)
    if frame_idx is not None:
        if pred_motions is not None:
            model_outputs["offset"] = pred_motions[:, frame_idx, :, :3, ...]
        if pred_motion_gs is not None:
            model_outputs.update(pred_motion_gs[frame_idx])
    return model_outputs


def render_target_camera_sequence(
    model: SplatRecon,
    backbone_outputs: Dict[str, Tensor],
    pred_motions: Tensor | None,
    pred_motion_gs: List[Dict[str, Tensor]] | None,
    input_C2W: Tensor,
    input_fxfycxcy: Tensor,
    target_C2W: Tensor,
    target_fxfycxcy: Tensor,
) -> Tensor:
    frame_count = target_C2W.shape[1]
    render_outputs_list: List[Dict[str, Tensor]] = []
    for frame_idx in range(frame_count):
        frame_outputs = get_frame_model_outputs(backbone_outputs, pred_motions, pred_motion_gs, frame_idx)
        render_outputs_list.append(
            model.gs_renderer.render(
                frame_outputs,
                input_C2W,
                input_fxfycxcy,
                target_C2W[:, frame_idx:frame_idx + 1, ...],
                target_fxfycxcy[:, frame_idx:frame_idx + 1, ...],
            )
        )
    return torch.cat([render_outputs_list[i]["image"] for i in range(frame_count)], dim=1)


def compute_metrics(model: SplatRecon, pred_images: Tensor, gt_images: Tensor) -> Dict[str, float]:
    frame_count = pred_images.shape[1]
    psnr = -10.0 * torch.log10(torch.mean((gt_images - pred_images) ** 2, dim=(1, 2, 3, 4)))
    ssim = SSIM(
        rearrange(gt_images, "b f c h w -> (b f) c h w"),
        rearrange(pred_images, "b f c h w -> (b f) c h w"),
        data_range=1.0,
        size_average=False,
    )
    ssim = rearrange(ssim, "(b f) -> b f", f=frame_count).mean(dim=1)
    lpips = model.lpips_loss(
        rearrange(gt_images, "b f c h w -> (b f) c h w") * 2.0 - 1.0,
        rearrange(pred_images, "b f c h w -> (b f) c h w") * 2.0 - 1.0,
    )
    lpips = rearrange(lpips, "(b f) c h w -> b f c h w", f=frame_count).mean(dim=(1, 2, 3, 4))
    return {
        "psnr": float(psnr.item()),
        "ssim": float(ssim.item()),
        "ssim_x100": float(ssim.item() * 100.0),
        "lpips": float(lpips.item()),
    }


def save_scene_videos(output_dir: str, source_images: Tensor, render_images: Tensor, target_images: Tensor) -> None:
    iio.mimwrite(os.path.join(output_dir, "input_round_robin.mp4"), tensor_to_video(source_images.unsqueeze(0)), macro_block_size=1)
    iio.mimwrite(os.path.join(output_dir, "render_cam_eval.mp4"), tensor_to_video(render_images), macro_block_size=1)
    iio.mimwrite(os.path.join(output_dir, "target_cam_eval.mp4"), tensor_to_video(target_images.unsqueeze(0)), macro_block_size=1)


def evaluate_scene(
    scene: str,
    model: SplatRecon,
    args: argparse.Namespace,
    autocast_dtype: torch.dtype,
    autocast_device: str,
) -> Dict[str, object]:
    scene_dir = os.path.join(args.data_root, scene)
    if not os.path.exists(scene_dir):
        raise FileNotFoundError(f"NVIDIA scene not found: {scene_dir}")

    scene_start = time.perf_counter()
    source_images, source_C2W, source_fxfycxcy = load_source_stack(
        scene_dir,
        args.effective_image_height,
        args.effective_image_width,
    )
    target_images, target_C2W, target_fxfycxcy = load_target_stack(
        scene_dir,
        args.eval_camera_id,
        args.effective_image_height,
        args.effective_image_width,
    )

    canonical_transform = inverse_c2w(source_C2W[0])
    source_C2W = apply_world_transform(source_C2W, canonical_transform)
    target_C2W = apply_world_transform(target_C2W, canonical_transform)
    round_robin_check = verify_round_robin_alignment(scene_dir)

    input_images = source_images.unsqueeze(0).to(args.device)
    input_C2W = source_C2W.unsqueeze(0).to(args.device)
    input_fxfycxcy = source_fxfycxcy.unsqueeze(0).to(args.device)
    input_timesteps = torch.linspace(0.0, 1.0, steps=source_images.shape[0], dtype=torch.float32).unsqueeze(0).to(args.device)

    target_images_batched = target_images.unsqueeze(0).to(args.device)
    target_C2W_batched = target_C2W.unsqueeze(0).to(args.device)
    target_fxfycxcy_batched = target_fxfycxcy.unsqueeze(0).to(args.device)

    backbone_start = time.perf_counter()
    print(f"[NVIDIA eval] scene={scene} stage=backbone start", flush=True)
    with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=args.device.startswith("cuda")):
        backbone_outputs, pred_motions, pred_motion_gs = model.backbone(
            input_images.to(dtype=autocast_dtype),
            input_C2W.to(dtype=autocast_dtype),
            input_fxfycxcy.to(dtype=autocast_dtype),
            input_timesteps.to(dtype=autocast_dtype),
            input_timesteps.to(dtype=autocast_dtype),
            frames_chunk_size=args.frames_chunk_size,
        )
    print(
        f"[NVIDIA eval] scene={scene} stage=backbone done elapsed_sec={time.perf_counter() - backbone_start:.2f}",
        flush=True,
    )

    render_start = time.perf_counter()
    print(f"[NVIDIA eval] scene={scene} stage=render start", flush=True)
    render_images = render_target_camera_sequence(
        model,
        backbone_outputs,
        pred_motions,
        pred_motion_gs,
        input_C2W.to(dtype=autocast_dtype),
        input_fxfycxcy.to(dtype=autocast_dtype),
        target_C2W_batched.to(dtype=autocast_dtype),
        target_fxfycxcy_batched.to(dtype=autocast_dtype),
    ).float()
    metrics = compute_metrics(model, render_images, target_images_batched.float())
    render_elapsed = time.perf_counter() - render_start
    print(
        f"[NVIDIA eval] scene={scene} stage=render done elapsed_sec={render_elapsed:.2f} "
        f"psnr={metrics['psnr']:.4f} ssim={metrics['ssim']:.4f} lpips={metrics['lpips']:.4f}",
        flush=True,
    )

    output_dir = os.path.join("out", "nvidia", scene)
    os.makedirs(output_dir, exist_ok=True)
    if args.save_videos:
        save_scene_videos(output_dir, source_images, render_images.cpu(), target_images)

    summary = {
        "scene": scene,
        "checkpoint": args.ckpt_path,
        "data_root": args.data_root,
        "metric_definition": "NVIDIA dynamic scene benchmark local eval with round-robin source frames and a fixed target camera over time",
        "metric_scope": "full-frame RGB metrics without dynamic-object masks",
        "protocol_kind": "round_robin_input_fixed_camera_target",
        "sampling_protocol": {
            "num_input_frames": 12,
            "input_camera_ids": list(range(1, 13)),
            "input_time_ids": list(range(1, 13)),
            "eval_camera_id": int(args.eval_camera_id),
            "target_time_ids": list(range(1, 13)),
            "requested_image_size": [int(args.image_height), int(args.image_width)],
            "effective_image_size": [int(args.effective_image_height), int(args.effective_image_width)],
            "frames_chunk_size": int(args.frames_chunk_size),
        },
        "assumptions": [
            "Uses the official Yoon et al. NVIDIA dynamic scene archive layout with input_images/, multiview_GT/, and calibration/.",
            "Interprets input_images/camXX.jpg as the round-robin source frame from camera XX at timestep XX.",
            "Evaluates a fixed target camera across all 12 timesteps, following the DynNeRF / BTimer-style NVIDIA protocol.",
            "Rescales images and camera intrinsics to the nearest VGGT patch-compatible resolution not exceeding the requested size.",
            "Does not use the provided dynamic-object foreground masks when computing metrics, matching the MoVieS paper note that no video masks are used in these experiments.",
            "Canonicalizes poses by the first source camera before inference, matching the repo's other posed-video inference utilities.",
        ],
        "round_robin_alignment_check": round_robin_check,
        "metrics": metrics,
        "elapsed_sec": float(time.perf_counter() - scene_start),
    }

    output_json = default_scene_output_json(scene, args.eval_camera_id)
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[NVIDIA eval] scene={scene} stage=complete output={output_json}", flush=True)

    return summary


def save_aggregate_results(results: List[Dict[str, object]], output_json: str) -> None:
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    aggregate = {
        "scene_count": len(results),
        "scenes": results,
        "overall_mean_psnr": float(np.mean([result["metrics"]["psnr"] for result in results])),
        "overall_mean_ssim": float(np.mean([result["metrics"]["ssim"] for result in results])),
        "overall_mean_ssim_x100": float(np.mean([result["metrics"]["ssim_x100"] for result in results])),
        "overall_mean_lpips": float(np.mean([result["metrics"]["lpips"] for result in results])),
    }
    with open(output_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene", "psnr", "ssim", "ssim_x100", "lpips", "elapsed_sec"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "scene": result["scene"],
                    "psnr": result["metrics"]["psnr"],
                    "ssim": result["metrics"]["ssim"],
                    "ssim_x100": result["metrics"]["ssim_x100"],
                    "lpips": result["metrics"]["lpips"],
                    "elapsed_sec": result["elapsed_sec"],
                }
            )


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    scenes = resolve_scenes(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU for NVIDIA evaluation.", flush=True)
        args.device = "cpu"

    args.effective_image_height = snap_to_patch_multiple(args.image_height)
    args.effective_image_width = snap_to_patch_multiple(args.image_width)
    if (args.effective_image_height, args.effective_image_width) != (args.image_height, args.image_width):
        print(
            "[NVIDIA eval] requested_resolution="
            f"{args.image_height}x{args.image_width} effective_resolution="
            f"{args.effective_image_height}x{args.effective_image_width} "
            f"(snapped to multiples of {PATCH_SIZE} for VGGT patch embedding)",
            flush=True,
        )

    opt = copy.deepcopy(opt_dict["movies"])
    opt.input_res = (args.effective_image_height, args.effective_image_width)
    opt.depth_weight = 0.0
    opt.motion_weight = 0.0

    model = SplatRecon(opt)
    model.load_state_dict(load_file(args.ckpt_path), strict=True)
    model.eval().to(args.device)

    autocast_dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    autocast_device = "cuda" if args.device.startswith("cuda") else "cpu"

    results = []
    for scene in scenes:
        results.append(evaluate_scene(scene, model, args, autocast_dtype, autocast_device))

    aggregate_output_json = args.output_json or default_aggregate_output_json(args.eval_camera_id)
    save_aggregate_results(results, aggregate_output_json)
    print(f"[NVIDIA eval] aggregate_output={aggregate_output_json}", flush=True)


if __name__ == "__main__":
    main(parse_args())
