import sys; sys.path.append("../extensions/vggt")

from typing import Any, Dict, List, Sequence, Tuple

import os
import copy
import json
import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as tF
import imageio.v2 as iio
from lpips import LPIPS
from scipy.signal import convolve2d
from torch import Tensor
from einops import rearrange
from safetensors.torch import load_file
from pytorch_msssim import ssim as SSIM

import sys; sys.path.append(".")  # for src modules
from src.options import opt_dict
from src.models import SplatRecon
from src.utils import inverse_c2w


CKPT_PATH = "resources/movies_ckpt.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MoVieS on a local DyCheck iPhone scene")
    parser.add_argument("--scene", type=str, required=True, help="DyCheck iPhone scene name, e.g. apple")
    parser.add_argument("--data-root", type=str, default="datasets/iphone", help="Root directory containing DyCheck iPhone scenes")
    parser.add_argument("--ckpt-path", type=str, default=CKPT_PATH, help="Path to the MoVieS checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--image-size", type=int, default=518, help="Square evaluation resolution")
    parser.add_argument("--clip-length", type=int, default=65, help="Contiguous clip length before sub-sampling")
    parser.add_argument("--num-frames", type=int, default=13, help="Number of uniformly sampled source frames")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for clip sampling")
    parser.add_argument("--eval-camera-ids", type=int, nargs="+", default=[1, 2], help="Held-out camera ids for evaluation")
    parser.add_argument(
        "--metric-mask",
        type=str,
        choices=["none", "covisible", "both"],
        default="none",
        help="Use no mask, the DyCheck covisibility mask, or report both masked and unmasked metrics",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Optional output json path")
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def build_frame_index(scene_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset_dict = load_json(os.path.join(scene_dir, "dataset.json"))
    frame_names = np.array(dataset_dict["ids"])
    metadata_dict = load_json(os.path.join(scene_dir, "metadata.json"))
    time_ids = np.array([metadata_dict[name]["warp_id"] for name in frame_names], dtype=np.int32)
    camera_ids = np.array([metadata_dict[name]["camera_id"] for name in frame_names], dtype=np.int32)
    return frame_names, time_ids, camera_ids


def build_frame_name_map(frame_names: np.ndarray, time_ids: np.ndarray, camera_ids: np.ndarray) -> Dict[Tuple[int, int], str]:
    return {
        (int(time_id), int(camera_id)): frame_name
        for frame_name, time_id, camera_id in zip(frame_names, time_ids, camera_ids)
    }


def split_consecutive_runs(sorted_values: Sequence[int]) -> List[List[int]]:
    if len(sorted_values) == 0:
        return []
    runs = [[int(sorted_values[0])]]
    for value in sorted_values[1:]:
        value = int(value)
        if value == runs[-1][-1] + 1:
            runs[-1].append(value)
        else:
            runs.append([value])
    return runs


def select_clip_time_ids(
    frame_name_map: Dict[Tuple[int, int], str],
    eval_camera_ids: Sequence[int],
    clip_length: int,
    seed: int,
) -> List[int]:
    available_times = sorted({
        time_id
        for (time_id, camera_id) in frame_name_map.keys()
        if camera_id == 0 and all((time_id, eval_camera_id) in frame_name_map for eval_camera_id in eval_camera_ids)
    })

    runs = [run for run in split_consecutive_runs(available_times) if len(run) >= clip_length]
    if len(runs) == 0:
        raise ValueError(
            f"Could not find a contiguous {clip_length}-frame clip shared by source camera 0 and eval cameras {list(eval_camera_ids)}"
        )

    rng = np.random.default_rng(seed)
    run = runs[int(rng.integers(len(runs)))]
    start_offset = int(rng.integers(len(run) - clip_length + 1))
    return run[start_offset:start_offset + clip_length]


def sample_time_ids_from_clip(clip_time_ids: Sequence[int], num_frames: int) -> List[int]:
    indices = np.linspace(0, len(clip_time_ids) - 1, num_frames, dtype=int)
    return [int(clip_time_ids[idx]) for idx in indices]


def load_scene_norm(scene_dir: str) -> Tuple[np.ndarray, float]:
    scene_dict = load_json(os.path.join(scene_dir, "scene.json"))
    center = np.array(scene_dict["center"], dtype=np.float32)
    scale = float(scene_dict["scale"])
    return center, scale


def load_factor(scene_dir: str, default_factor: int = 2) -> int:
    extra_path = os.path.join(scene_dir, "extra.json")
    if os.path.exists(extra_path):
        return int(load_json(extra_path)["factor"])
    return default_factor


def load_camera_dict(camera_path: str) -> Dict[str, Any]:
    camera_dict = load_json(camera_path)
    if "tangential" in camera_dict:
        camera_dict["tangential_distortion"] = camera_dict["tangential"]
    return camera_dict


def undistort_rgb(image: np.ndarray, intrin: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    if np.allclose(distortion, 0):
        return image
    return cv2.undistort(image, intrin, distortion)


def undistort_mask(mask: np.ndarray, intrin: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    if np.allclose(distortion, 0):
        return mask
    H, W = mask.shape[:2]
    map_x, map_y = cv2.initUndistortRectifyMap(
        intrin, distortion, None, intrin, (W, H), cv2.CV_32FC1,
    )
    return cv2.remap(mask.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_NEAREST)


def undistort_image_domain_intrinsics(intrin: np.ndarray) -> np.ndarray:
    intrin = intrin.copy()
    # Match DyCheck's Camera.undistort_image_domain(), which disables skew and
    # distortion while keeping focal length and principal point in the same
    # image domain.
    intrin[0, 1] = 0.0
    return intrin


def camera_dict_to_matrices(camera_dict: Dict[str, Any], center: np.ndarray, scale: float, factor: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    orientation = np.asarray(camera_dict["orientation"], dtype=np.float32)  # world-to-camera rotation
    position = np.asarray(camera_dict["position"], dtype=np.float32)
    position = (position - center) * scale

    focal = float(camera_dict["focal_length"]) / factor
    principal_point = np.asarray(camera_dict["principal_point"], dtype=np.float32) / factor
    image_size = np.asarray(camera_dict["image_size"], dtype=np.float32) / factor
    skew = float(camera_dict["skew"]) / factor
    pixel_aspect_ratio = float(camera_dict["pixel_aspect_ratio"])
    radial = np.asarray(camera_dict["radial_distortion"], dtype=np.float32)
    tangential = np.asarray(camera_dict["tangential_distortion"], dtype=np.float32)

    intrin = np.array(
        [
            [focal, skew, principal_point[0]],
            [0.0, focal * pixel_aspect_ratio, principal_point[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    distortion = np.concatenate([radial[:2], tangential, radial[-1:]], axis=0).astype(np.float32)

    C2W = np.eye(4, dtype=np.float32)
    C2W[:3, :3] = orientation.T
    C2W[:3, 3] = position

    # Keep the parser's image-domain scaling behavior consistent with the official DyCheck loader.
    image_size = np.round(image_size).astype(np.int32)
    return C2W, intrin, distortion


def resize_crop_image_and_intrinsics(
    image: np.ndarray,
    intrin: np.ndarray,
    output_size: int,
) -> Tuple[Tensor, np.ndarray]:
    H, W = image.shape[:2]
    scale = max(output_size / H, output_size / W)
    scaled_H, scaled_W = round(H * scale), round(W * scale)

    image = cv2.resize(image, (scaled_W, scaled_H), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    intrin = intrin.copy()
    intrin[0, :] *= (scaled_W / W)
    intrin[1, :] *= (scaled_H / H)

    top = max((scaled_H - output_size) // 2, 0)
    left = max((scaled_W - output_size) // 2, 0)
    image = image[top:top + output_size, left:left + output_size]
    intrin[0, 2] -= left
    intrin[1, 2] -= top

    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return image, intrin


def resize_crop_mask(mask: np.ndarray, output_size: int) -> Tensor:
    H, W = mask.shape[:2]
    scale = max(output_size / H, output_size / W)
    scaled_H, scaled_W = round(H * scale), round(W * scale)

    mask = cv2.resize(mask.astype(np.float32), (scaled_W, scaled_H), interpolation=cv2.INTER_NEAREST)

    top = max((scaled_H - output_size) // 2, 0)
    left = max((scaled_W - output_size) // 2, 0)
    mask = mask[top:top + output_size, left:left + output_size]
    return torch.from_numpy((mask > 0.5).astype(np.float32))


def intrin_to_fxfycxcy(intrin: np.ndarray, image_size: int) -> Tensor:
    return torch.tensor(
        [
            intrin[0, 0] / image_size,
            intrin[1, 1] / image_size,
            intrin[0, 2] / image_size,
            intrin[1, 2] / image_size,
        ],
        dtype=torch.float32,
    )


def load_frame(
    scene_dir: str,
    frame_name: str,
    center: np.ndarray,
    scale: float,
    factor: int,
    image_size: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    image_path = os.path.join(scene_dir, "rgb", f"{factor}x", frame_name + ".png")
    camera_path = os.path.join(scene_dir, "camera", frame_name + ".json")

    image = iio.imread(image_path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]

    camera_dict = load_camera_dict(camera_path)
    C2W, intrin, distortion = camera_dict_to_matrices(camera_dict, center, scale, factor)
    image = undistort_rgb(image, intrin, distortion)
    intrin = undistort_image_domain_intrinsics(intrin)
    image, intrin = resize_crop_image_and_intrinsics(image, intrin, image_size)

    return image, torch.from_numpy(C2W).float(), intrin_to_fxfycxcy(intrin, image_size)


def load_covisible_mask(
    scene_dir: str,
    frame_name: str,
    center: np.ndarray,
    scale: float,
    factor: int,
    image_size: int,
) -> Tensor:
    mask_path = os.path.join(scene_dir, "covisible", f"{factor}x", "val", frame_name + ".png")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Missing DyCheck covisibility mask: {mask_path}. "
            "Re-download the subset with --include-covisible."
        )

    camera_path = os.path.join(scene_dir, "camera", frame_name + ".json")
    mask = iio.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.float32)

    camera_dict = load_camera_dict(camera_path)
    _, intrin, distortion = camera_dict_to_matrices(camera_dict, center, scale, factor)
    mask = undistort_mask(mask, intrin, distortion)
    mask = resize_crop_mask(mask, image_size)
    return mask


def load_camera_time_stack(
    scene_dir: str,
    frame_name_map: Dict[Tuple[int, int], str],
    time_ids: Sequence[int],
    camera_id: int,
    center: np.ndarray,
    scale: float,
    factor: int,
    image_size: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    images, C2Ws, intrinsics = [], [], []
    for time_id in time_ids:
        frame_name = frame_name_map[(int(time_id), int(camera_id))]
        image, C2W, fxfycxcy = load_frame(scene_dir, frame_name, center, scale, factor, image_size)
        images.append(image)
        C2Ws.append(C2W)
        intrinsics.append(fxfycxcy)
    return torch.stack(images, dim=0), torch.stack(C2Ws, dim=0), torch.stack(intrinsics, dim=0)


def load_covisible_time_stack(
    scene_dir: str,
    frame_name_map: Dict[Tuple[int, int], str],
    time_ids: Sequence[int],
    camera_id: int,
    center: np.ndarray,
    scale: float,
    factor: int,
    image_size: int,
) -> Tensor:
    masks = []
    for time_id in time_ids:
        frame_name = frame_name_map[(int(time_id), int(camera_id))]
        masks.append(load_covisible_mask(scene_dir, frame_name, center, scale, factor, image_size))
    return torch.stack(masks, dim=0)


def apply_world_transform(C2W: Tensor, transform: Tensor) -> Tensor:
    return transform.unsqueeze(0) @ C2W


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
    F_out = target_C2W.shape[1]
    render_outputs_list: List[Dict[str, Tensor]] = []
    for i in range(F_out):
        frame_outputs = get_frame_model_outputs(backbone_outputs, pred_motions, pred_motion_gs, i)
        render_outputs_list.append(
            model.gs_renderer.render(
                frame_outputs,
                input_C2W,
                input_fxfycxcy,
                target_C2W[:, i:i + 1, ...],
                target_fxfycxcy[:, i:i + 1, ...],
            )
        )
    return torch.cat([render_outputs_list[i]["image"] for i in range(F_out)], dim=1)


def compute_metrics(
    model: SplatRecon,
    pred_images: Tensor,
    gt_images: Tensor,
) -> Dict[str, float]:
    F = pred_images.shape[1]
    psnr = -10 * torch.log10(torch.mean((gt_images - pred_images) ** 2, dim=(1, 2, 3, 4)))
    ssim = SSIM(
        rearrange(gt_images, "b f c h w -> (b f) c h w"),
        rearrange(pred_images, "b f c h w -> (b f) c h w"),
        data_range=1.,
        size_average=False,
    )
    ssim = rearrange(ssim, "(b f) -> b f", f=F).mean(dim=1)
    lpips = model.lpips_loss(
        rearrange(gt_images, "b f c h w -> (b f) c h w") * 2. - 1.,
        rearrange(pred_images, "b f c h w -> (b f) c h w") * 2. - 1.,
    )
    lpips = rearrange(lpips, "(b f) c h w -> b f c h w", f=F).mean(dim=(1, 2, 3, 4))

    return {
        "psnr": float(psnr.item()),
        "ssim": float(ssim.item()),
        "ssim_x100": float(ssim.item() * 100.0),
        "lpips": float(lpips.item()),
    }


def gaussian_kernel(filter_size: int = 11, filter_sigma: float = 1.5) -> np.ndarray:
    coords = np.arange(filter_size, dtype=np.float32) - filter_size // 2
    kernel_1d = np.exp(-(coords ** 2) / (2 * filter_sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    return np.outer(kernel_1d, kernel_1d).astype(np.float32)


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float((values * mask).sum() / np.clip(mask.sum(), 1e-6, None))


def compute_masked_ssim(
    pred_image: np.ndarray,
    gt_image: np.ndarray,
    mask: np.ndarray,
    kernel: np.ndarray,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    mask = mask.astype(np.float32)
    local_weight = convolve2d(mask, kernel, mode="same", boundary="symm")
    local_weight = np.clip(local_weight, 1e-6, None)
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    channel_ssim = []

    for channel in range(gt_image.shape[-1]):
        x = gt_image[..., channel].astype(np.float32)
        y = pred_image[..., channel].astype(np.float32)

        mu_x = convolve2d(x * mask, kernel, mode="same", boundary="symm") / local_weight
        mu_y = convolve2d(y * mask, kernel, mode="same", boundary="symm") / local_weight
        sigma_x = convolve2d((x ** 2) * mask, kernel, mode="same", boundary="symm") / local_weight - mu_x ** 2
        sigma_y = convolve2d((y ** 2) * mask, kernel, mode="same", boundary="symm") / local_weight - mu_y ** 2
        sigma_xy = convolve2d((x * y) * mask, kernel, mode="same", boundary="symm") / local_weight - mu_x * mu_y

        sigma_x = np.maximum(sigma_x, 0.0)
        sigma_y = np.maximum(sigma_y, 0.0)
        sigma_xy = np.clip(sigma_xy, -np.sqrt(sigma_x * sigma_y + 1e-12), np.sqrt(sigma_x * sigma_y + 1e-12))

        numer = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denom = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        channel_ssim.append(numer / np.clip(denom, 1e-6, None))

    ssim_map = np.mean(np.stack(channel_ssim, axis=0), axis=0)
    return masked_mean(ssim_map, mask)


def compute_masked_metrics(
    spatial_lpips_loss: LPIPS,
    pred_images: Tensor,
    gt_images: Tensor,
    covisible_masks: Tensor,
) -> Dict[str, float]:
    B, F = pred_images.shape[:2]
    pred_np = rearrange(pred_images.detach().cpu(), "b f c h w -> b f h w c").numpy()
    gt_np = rearrange(gt_images.detach().cpu(), "b f c h w -> b f h w c").numpy()
    mask_np = rearrange(covisible_masks.detach().cpu(), "b f 1 h w -> b f h w").numpy()
    kernel = gaussian_kernel()

    batch_psnr, batch_ssim, batch_coverage = [], [], []
    for batch_idx in range(B):
        frame_psnr, frame_ssim, frame_coverage = [], [], []
        for frame_idx in range(F):
            mask = (mask_np[batch_idx, frame_idx] > 0.5).astype(np.float32)
            frame_coverage.append(float(mask.mean()))

            mse = np.mean((gt_np[batch_idx, frame_idx] - pred_np[batch_idx, frame_idx]) ** 2, axis=-1)
            frame_psnr.append(float(-10.0 * np.log10(max(masked_mean(mse, mask), 1e-10))))
            frame_ssim.append(compute_masked_ssim(pred_np[batch_idx, frame_idx], gt_np[batch_idx, frame_idx], mask, kernel))

        batch_psnr.append(float(np.mean(frame_psnr)))
        batch_ssim.append(float(np.mean(frame_ssim)))
        batch_coverage.append(float(np.mean(frame_coverage)))

    pred_masked = pred_images * covisible_masks
    gt_masked = gt_images * covisible_masks
    lpips_map = spatial_lpips_loss(
        rearrange(gt_masked, "b f c h w -> (b f) c h w") * 2.0 - 1.0,
        rearrange(pred_masked, "b f c h w -> (b f) c h w") * 2.0 - 1.0,
    )
    flat_mask = rearrange(covisible_masks, "b f 1 h w -> (b f) 1 h w").to(lpips_map.dtype)
    if flat_mask.shape[-2:] != lpips_map.shape[-2:]:
        flat_mask = tF.interpolate(flat_mask, size=lpips_map.shape[-2:], mode="nearest")
    masked_lpips = (lpips_map * flat_mask).sum(dim=(1, 2, 3)) / flat_mask.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    masked_lpips = rearrange(masked_lpips, "(b f) -> b f", f=F).mean(dim=1)

    mpsnr = torch.tensor(batch_psnr, dtype=torch.float32)
    mssim = torch.tensor(batch_ssim, dtype=torch.float32)
    mask_coverage = torch.tensor(batch_coverage, dtype=torch.float32)
    return {
        "mpsnr": float(mpsnr.mean().item()),
        "mssim": float(mssim.mean().item()),
        "mssim_x100": float(mssim.mean().item() * 100.0),
        "mlpips": float(masked_lpips.mean().item()),
        "mean_covisible_coverage": float(mask_coverage.mean().item()),
    }


def default_output_json(scene: str, metric_mask: str) -> str:
    filename = {
        "none": "dycheck_metrics.json",
        "covisible": "dycheck_metrics_covisible.json",
        "both": "dycheck_metrics_both.json",
    }[metric_mask]
    return os.path.join("out", "dycheck", scene, filename)


def get_protocol_kind(eval_camera_ids: Sequence[int]) -> str:
    unique_eval_camera_ids = sorted(set(int(camera_id) for camera_id in eval_camera_ids))
    if unique_eval_camera_ids == [0]:
        return "same_view_reconstruction"
    return "held_out_novel_view"


@torch.inference_mode()
def main(args: argparse.Namespace):
    start_time = time.perf_counter()
    scene_dir = os.path.join(args.data_root, args.scene)
    if not os.path.exists(scene_dir):
        raise FileNotFoundError(f"DyCheck scene not found: {scene_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU for DyCheck evaluation.")
        args.device = "cpu"

    frame_names, time_ids, camera_ids = build_frame_index(scene_dir)
    frame_name_map = build_frame_name_map(frame_names, time_ids, camera_ids)
    center, scale = load_scene_norm(scene_dir)
    factor = load_factor(scene_dir)
    protocol_kind = get_protocol_kind(args.eval_camera_ids)

    clip_time_ids = select_clip_time_ids(frame_name_map, args.eval_camera_ids, args.clip_length, args.seed)
    sampled_time_ids = sample_time_ids_from_clip(clip_time_ids, args.num_frames)
    print(
        f"[DyCheck] scene={args.scene} factor={factor} clip_start={clip_time_ids[0]} "
        f"clip_end={clip_time_ids[-1]} sampled_frames={len(sampled_time_ids)}",
        flush=True,
    )

    source_images, source_C2W, source_fxfycxcy = load_camera_time_stack(
        scene_dir, frame_name_map, sampled_time_ids, 0, center, scale, factor, args.image_size
    )
    canonical_transform = inverse_c2w(source_C2W[0])
    source_C2W = apply_world_transform(source_C2W, canonical_transform)

    target_stacks = {}
    for camera_id in args.eval_camera_ids:
        target_images, target_C2W, target_fxfycxcy = load_camera_time_stack(
            scene_dir, frame_name_map, sampled_time_ids, camera_id, center, scale, factor, args.image_size
        )
        target_C2W = apply_world_transform(target_C2W, canonical_transform)
        target_stacks[camera_id] = {
            "images": target_images,
            "C2W": target_C2W,
            "fxfycxcy": target_fxfycxcy,
        }
        if args.metric_mask in {"covisible", "both"}:
            target_stacks[camera_id]["covisible"] = load_covisible_time_stack(
                scene_dir, frame_name_map, sampled_time_ids, camera_id, center, scale, factor, args.image_size
            )

    opt = copy.deepcopy(opt_dict["movies"])
    opt.input_res = (args.image_size, args.image_size)
    opt.depth_weight = 0.
    opt.motion_weight = 0.

    model = SplatRecon(opt)
    model.load_state_dict(load_file(args.ckpt_path), strict=True)
    model.eval().to(args.device)
    spatial_lpips_loss = None
    if args.metric_mask in {"covisible", "both"}:
        spatial_lpips_loss = LPIPS(net="vgg", spatial=True).eval().to(args.device)

    input_images = source_images.unsqueeze(0).to(args.device)
    input_C2W = source_C2W.unsqueeze(0).to(args.device)
    input_fxfycxcy = source_fxfycxcy.unsqueeze(0).to(args.device)
    input_timesteps = torch.tensor(sampled_time_ids, dtype=torch.float32)
    input_timesteps = ((input_timesteps - input_timesteps.min()) / (input_timesteps.max() - input_timesteps.min())).unsqueeze(0).to(args.device)

    autocast_dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    autocast_device = "cuda" if args.device.startswith("cuda") else "cpu"
    backbone_start = time.perf_counter()
    print(f"[DyCheck] scene={args.scene} stage=backbone start", flush=True)
    with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=args.device.startswith("cuda")):
        backbone_outputs, pred_motions, pred_motion_gs = model.backbone(
            input_images.to(dtype=autocast_dtype),
            input_C2W.to(dtype=autocast_dtype),
            input_fxfycxcy.to(dtype=autocast_dtype),
            input_timesteps.to(dtype=autocast_dtype),
            input_timesteps.to(dtype=autocast_dtype),
            frames_chunk_size=opt.frames_chunk_size,
        )
    print(
        f"[DyCheck] scene={args.scene} stage=backbone done "
        f"elapsed_sec={time.perf_counter() - backbone_start:.2f}",
        flush=True,
    )

    camera_results = []
    for camera_id in args.eval_camera_ids:
        camera_start = time.perf_counter()
        print(f"[DyCheck] scene={args.scene} camera={camera_id} stage=render start", flush=True)
        target_images = target_stacks[camera_id]["images"].unsqueeze(0).to(args.device)
        target_C2W = target_stacks[camera_id]["C2W"].unsqueeze(0).to(args.device)
        target_fxfycxcy = target_stacks[camera_id]["fxfycxcy"].unsqueeze(0).to(args.device)

        render_images = render_target_camera_sequence(
            model,
            backbone_outputs,
            pred_motions,
            pred_motion_gs,
            input_C2W.to(dtype=autocast_dtype),
            input_fxfycxcy.to(dtype=autocast_dtype),
            target_C2W.to(dtype=autocast_dtype),
            target_fxfycxcy.to(dtype=autocast_dtype),
        ).float()

        metrics: Dict[str, float] = {}
        if args.metric_mask in {"none", "both"}:
            metrics.update(compute_metrics(model, render_images, target_images.float()))
        if args.metric_mask in {"covisible", "both"}:
            covisible_masks = target_stacks[camera_id]["covisible"].unsqueeze(0).unsqueeze(2).to(args.device)
            metrics.update(compute_masked_metrics(spatial_lpips_loss, render_images, target_images.float(), covisible_masks))
        metrics.update({
            "camera_id": int(camera_id),
            "num_target_frames": int(target_images.shape[1]),
        })
        camera_results.append(metrics)
        metric_parts = []
        if "psnr" in metrics:
            metric_parts.append(
                f"psnr={metrics['psnr']:.4f} ssim={metrics['ssim']:.4f} lpips={metrics['lpips']:.4f}"
            )
        if "mpsnr" in metrics:
            metric_parts.append(
                f"mpsnr={metrics['mpsnr']:.4f} mssim={metrics['mssim']:.4f} mlpips={metrics['mlpips']:.4f}"
            )
        print(
            f"[DyCheck] scene={args.scene} camera={camera_id} stage=render done "
            f"elapsed_sec={time.perf_counter() - camera_start:.2f} "
            + " ".join(metric_parts),
            flush=True,
        )

    assumptions = [
        "Uses local DyCheck iPhone scene in Nerfies-style format.",
        "Applies scene.json center/scale normalization and canonicalizes the first source camera.",
        "Undistorts RGBs and switches the intrinsics to DyCheck's undistorted image domain before resize/crop.",
        "Aggregates metrics over the selected timestamps for each held-out camera.",
    ]
    metric_definition = "DyCheck iPhone local eval with camera_id 0 as source and other cameras as held-out novel views"
    if protocol_kind == "same_view_reconstruction":
        metric_definition = "DyCheck iPhone local eval with camera_id 0 used as both source and target views"
        assumptions.append(
            "This scene was evaluated in same-view reconstruction mode because the local metadata does not expose held-out target cameras with the requested overlap."
        )
    if args.metric_mask == "none":
        metric_scope = "full-frame unmasked RGB metrics on held-out cameras"
        if protocol_kind == "same_view_reconstruction":
            metric_scope = "full-frame unmasked RGB metrics on source camera 0"
        assumptions.append(
            "Does not use DyCheck co-visibility masks, so these numbers are not directly comparable to the official masked benchmark."
        )
    elif args.metric_mask == "covisible":
        metric_scope = "DyCheck covisible-masked RGB metrics on held-out cameras"
        if protocol_kind == "same_view_reconstruction":
            metric_scope = "DyCheck covisible-masked RGB metrics on source camera 0"
        assumptions.append(
            "Loads covisibility masks from covisible/<factor>x/val/*.png, undistorts them with nearest-neighbor remapping, and crops them with the RGB frames."
        )
    else:
        metric_scope = "both full-frame unmasked and DyCheck covisible-masked RGB metrics on held-out cameras"
        if protocol_kind == "same_view_reconstruction":
            metric_scope = "both full-frame unmasked and DyCheck covisible-masked RGB metrics on source camera 0"
        assumptions.extend([
            "Reports both the original full-frame metrics and a covisible-masked variant in the same file.",
            "Loads covisibility masks from covisible/<factor>x/val/*.png, undistorts them with nearest-neighbor remapping, and crops them with the RGB frames.",
        ])

    summary = {
        "scene": args.scene,
        "checkpoint": args.ckpt_path,
        "data_root": args.data_root,
        "metric_definition": metric_definition,
        "metric_scope": metric_scope,
        "metric_mask_mode": args.metric_mask,
        "protocol_kind": protocol_kind,
        "sampling_protocol": {
            "clip_length": int(args.clip_length),
            "num_frames": int(args.num_frames),
            "image_size": [int(args.image_size), int(args.image_size)],
            "seed": int(args.seed),
            "clip_time_ids": clip_time_ids,
            "sampled_time_ids": sampled_time_ids,
            "source_camera_id": 0,
            "eval_camera_ids": [int(camera_id) for camera_id in args.eval_camera_ids],
        },
        "assumptions": assumptions,
        "per_camera": camera_results,
    }
    if args.metric_mask in {"none", "both"}:
        summary.update({
            "overall_mean_psnr": float(np.mean([result["psnr"] for result in camera_results])),
            "overall_mean_ssim": float(np.mean([result["ssim"] for result in camera_results])),
            "overall_mean_ssim_x100": float(np.mean([result["ssim_x100"] for result in camera_results])),
            "overall_mean_lpips": float(np.mean([result["lpips"] for result in camera_results])),
        })
    if args.metric_mask in {"covisible", "both"}:
        summary.update({
            "overall_mean_mpsnr": float(np.mean([result["mpsnr"] for result in camera_results])),
            "overall_mean_mssim": float(np.mean([result["mssim"] for result in camera_results])),
            "overall_mean_mssim_x100": float(np.mean([result["mssim_x100"] for result in camera_results])),
            "overall_mean_mlpips": float(np.mean([result["mlpips"] for result in camera_results])),
            "overall_mean_covisible_coverage": float(np.mean([result["mean_covisible_coverage"] for result in camera_results])),
        })

    output_json = args.output_json or default_output_json(args.scene, args.metric_mask)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[DyCheck] scene={args.scene} stage=complete elapsed_sec={time.perf_counter() - start_time:.2f} "
        f"output={output_json}",
        flush=True,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(parse_args())
