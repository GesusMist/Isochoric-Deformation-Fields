import sys; sys.path.append("../extensions/vggt")

from typing import Any, Dict, Optional

import os
import argparse
import json
import numpy as np
import imageio.v2 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import Tensor
from safetensors.torch import load_file

import sys; sys.path.append(".")  # for src modules
from src.options import opt_dict
from src.models import SplatRecon
from src.models.gs_render.gs_util import GaussianModel
from src.utils import *


CKPT_PATH = "resources/movies_ckpt.safetensors"
DATA_DIR = "resources/DAVIS"


def concat_videos(video_paths: list[str], output_path: str) -> None:
    videos = [iio.mimread(video_path) for video_path in video_paths]
    videos = np.concatenate(videos, axis=0)  # (total_frames, H, W, 3)
    iio.mimwrite(output_path, videos, macro_block_size=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoVieS NVS inference")
    parser.add_argument("--name", type=str, required=True, help="DAVIS sample name")
    return parser.parse_args()


def get_frame_model_outputs(
    backbone_outputs: Dict[str, Tensor],
    pred_motions: Optional[Tensor] = None,
    pred_motion_gs: Optional[list[Dict[str, Tensor]]] = None,
    frame_idx: Optional[int] = None,
) -> Dict[str, Tensor]:
    model_outputs = dict(backbone_outputs)
    if frame_idx is not None:
        if pred_motions is not None:
            model_outputs["offset"] = pred_motions[:, frame_idx, :, :3, ...]
        if pred_motion_gs is not None:
            model_outputs.update(pred_motion_gs[frame_idx])
    return model_outputs


def export_gaussians_as_ply(
    model: SplatRecon,
    model_outputs: Dict[str, Tensor],
    input_C2W: Tensor,
    input_fxfycxcy: Tensor,
    render_C2W: Tensor,
    render_fxfycxcy: Tensor,
    output_path: str,
) -> int:
    export_outputs = model.gs_renderer.render(
        model_outputs,
        input_C2W,
        input_fxfycxcy,
        render_C2W,
        render_fxfycxcy,
        return_pc=True,
    )
    pc: GaussianModel = export_outputs["pc"][0]
    pc.save_ply(
        output_path,
        opacity_threshold=model.opt.opacity_threshold,
        compatible=True,
    )
    return int(pc.xyz.shape[0])


def compute_volume_stats(
    model: SplatRecon,
    backbone_outputs: Dict[str, Tensor],
    pred_motion_gs: Optional[list[Dict[str, Tensor]]],
    output_timesteps: Tensor,
) -> Dict[str, Any]:
    scale_activation = model.gs_renderer.scale_activation
    opacity_activation = model.gs_renderer.opacity_activation
    canonical_scale = scale_activation(backbone_outputs["scale"].float())  # (B, F_in, 3, H, W)
    canonical_opacity = opacity_activation(backbone_outputs["opacity"].float()).squeeze(2)  # (B, F_in, H, W)

    if pred_motion_gs is not None and len(pred_motion_gs) > 0 and "motion_scale" in pred_motion_gs[0]:
        deformed_scale = torch.stack([
            scale_activation(pred_motion_gs[i]["motion_scale"].float())
            for i in range(output_timesteps.shape[1])
        ], dim=1)  # (B, F_out, F_in, 3, H, W)
        deformed_scale_tensor = "pred_motion_gs[i]['motion_scale']"
        deformed_opacity = torch.stack([
            opacity_activation(pred_motion_gs[i]["motion_opacity"].float()).squeeze(2)
            for i in range(output_timesteps.shape[1])
        ], dim=1)  # (B, F_out, F_in, H, W)
        deformed_opacity_tensor = "pred_motion_gs[i]['motion_opacity']"
    elif "dynamic_splat" in backbone_outputs:
        deformed_scale = torch.stack([
            scale_activation(backbone_outputs["dynamic_splat"][i]["scale"].float())
            for i in range(output_timesteps.shape[1])
        ], dim=1)  # (B, F_out, F_in, 3, H, W)
        deformed_scale_tensor = "backbone_outputs['dynamic_splat'][i]['scale']"
        deformed_opacity = torch.stack([
            opacity_activation(backbone_outputs["dynamic_splat"][i]["opacity"].float()).squeeze(2)
            for i in range(output_timesteps.shape[1])
        ], dim=1)  # (B, F_out, F_in, H, W)
        deformed_opacity_tensor = "backbone_outputs['dynamic_splat'][i]['opacity']"
    else:
        deformed_scale = canonical_scale.unsqueeze(1).repeat(1, output_timesteps.shape[1], 1, 1, 1, 1)
        deformed_scale_tensor = "backbone_outputs['scale']"
        deformed_opacity = canonical_opacity.unsqueeze(1).repeat(1, output_timesteps.shape[1], 1, 1, 1)
        deformed_opacity_tensor = "backbone_outputs['opacity']"

    canonical_log_volume = torch.log(canonical_scale.clamp_min(model.opt.vol_log_eps)).sum(dim=2)  # (B, F_in, H, W)
    deformed_log_volume = torch.log(deformed_scale.clamp_min(model.opt.vol_log_eps)).sum(dim=3)  # (B, F_out, F_in, H, W)
    drift = torch.abs(deformed_log_volume - canonical_log_volume.unsqueeze(1))  # (B, F_out, F_in, H, W)

    frame_mean_drift = drift.mean(dim=(2, 3, 4))[0].cpu()  # (F_out,)
    drift_flat = drift[0].reshape(output_timesteps.shape[1], -1).cpu().numpy()  # (F_out, N)
    all_drift = drift_flat.reshape(-1)

    percentile_levels = [50, 90, 95, 99]
    per_frame_percentiles = {
        f"p{q}": np.quantile(drift_flat, q / 100.0, axis=1).tolist()
        for q in percentile_levels
    }
    overall_percentiles = {
        f"p{q}": float(np.quantile(all_drift, q / 100.0))
        for q in percentile_levels
    }
    hist_counts, hist_bin_edges = np.histogram(all_drift, bins=40)

    visible_mask = deformed_opacity > model.opt.opacity_threshold  # (B, F_out, F_in, H, W)
    visible_counts_per_frame = visible_mask[0].sum(dim=(1, 2, 3))
    visible_mean_drift_per_frame = (
        (drift[0] * visible_mask[0].float()).sum(dim=(1, 2, 3)) /
        visible_counts_per_frame.to(drift.dtype).clamp_min(1.)
    ).cpu()
    visible_counts_per_frame = visible_counts_per_frame.cpu()
    visible_drift = drift[0][visible_mask[0]].cpu().numpy()
    if visible_drift.size > 0:
        visible_overall_mean = float(visible_drift.mean())
        visible_overall_std = float(visible_drift.std())
        visible_overall_max = float(visible_drift.max())
    else:
        visible_overall_mean = 0.0
        visible_overall_std = 0.0
        visible_overall_max = 0.0

    return {
        "canonical_scale_tensor": "backbone_outputs['scale']",
        "deformed_scale_tensor": deformed_scale_tensor,
        "canonical_scale_shape": list(canonical_scale.shape),
        "deformed_scale_shape": list(deformed_scale.shape),
        "comparison_space": "dense_pre_render_indices_(input_frame, pixel_y, pixel_x)",
        "one_to_one_aligned_dense_indices": True,
        "compared_after_motion_mask_blend": pred_motion_gs is not None,
        "compared_after_renderer_pruning": False,
        "compared_after_renderer_voxelization": False,
        "renderer_opacity_threshold": model.opt.opacity_threshold,
        "canonical_opacity_tensor": "backbone_outputs['opacity']",
        "deformed_opacity_tensor": deformed_opacity_tensor,
        "scale_activation": model.opt.scale_act_type,
        "log_volume_definition": "logV = sum_j log(scale_j)",
        "drift_definition": "drift = abs(logV_t - logV_0)",
        "num_gaussians": int(canonical_scale.shape[1] * canonical_scale.shape[-2] * canonical_scale.shape[-1]),
        "canonical_visible_dense_splats": int((canonical_opacity[0] > model.opt.opacity_threshold).sum().item()),
        "visible_dense_splats_per_frame": visible_counts_per_frame.tolist(),
        "timesteps": output_timesteps[0].cpu().tolist(),
        "mean_drift_per_frame": frame_mean_drift.tolist(),
        "overall_mean_drift": float(all_drift.mean()),
        "overall_std_drift": float(all_drift.std()),
        "max_drift": float(all_drift.max()),
        "visible_mean_drift_per_frame": visible_mean_drift_per_frame.tolist(),
        "visible_overall_mean_drift": visible_overall_mean,
        "visible_overall_std_drift": visible_overall_std,
        "visible_overall_max_drift": visible_overall_max,
        "per_frame_percentiles": per_frame_percentiles,
        "overall_percentiles": overall_percentiles,
        "histogram": {
            "counts": hist_counts.tolist(),
            "bin_edges": hist_bin_edges.tolist(),
        },
    }


def save_volume_plot(output_path: str, stats: Dict[str, Any]) -> None:
    frame_idxs = np.arange(len(stats["mean_drift_per_frame"]))
    frame_means = np.asarray(stats["mean_drift_per_frame"], dtype=np.float32)
    histogram = stats["histogram"]
    bin_edges = np.asarray(histogram["bin_edges"], dtype=np.float32)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts = np.asarray(histogram["counts"], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(frame_idxs, frame_means, marker="o", linewidth=2)
    axes[0].set_title("Mean Log-Volume Drift")
    axes[0].set_xlabel("Output frame")
    axes[0].set_ylabel("Mean |logV_t - logV_0|")
    axes[0].grid(alpha=0.3)

    axes[1].plot(bin_centers, counts, linewidth=2)
    axes[1].fill_between(bin_centers, counts, alpha=0.25)
    axes[1].set_title("Volume Drift Histogram")
    axes[1].set_xlabel("|logV_t - logV_0|")
    axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def main(args: argparse.Namespace):
    output_dir = f"out/{args.name}"
    gaussian_dir = os.path.join(output_dir, "gaussians")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gaussian_dir, exist_ok=True)

    # NOTE: Load pretrained MoVieS model
    opt = opt_dict["movies"]
    model = SplatRecon(opt, load_lpips=False)  # lpips is not used for inference
    model.load_state_dict(load_file(CKPT_PATH), strict=True)
    model.eval()

    # NOTE: Load a preprocessed posed video for inference; camera is normalized similar to VGGT
    npz_path = f"{DATA_DIR}/{args.name}.npz"
    npz = np.load(npz_path)
    input_images = torch.from_numpy(npz["images"]).float().unsqueeze(0)  # (1, F_in, 3, H, W)
    input_C2W = torch.from_numpy(npz["C2W"]).float().unsqueeze(0)  # (1, F_in, 4, 4)
    input_fxfycxcy = torch.from_numpy(npz["fxfycxcy"]).float().unsqueeze(0)  # (1, F_in, 4); normalized intrinsics
    input_timesteps = torch.linspace(0, 1, steps=13).unsqueeze(0)  # (1, F_in)
    output_timesteps = torch.linspace(0, 1, steps=13).unsqueeze(0)  # (1, F_out)

    iio.mimwrite(f"{output_dir}/input_video.mp4", tensor_to_video(input_images), macro_block_size=1)

    F_in, F_out = input_timesteps.shape[1], output_timesteps.shape[1]
    output_fxfycxcy = input_fxfycxcy[:, 0:1, :].repeat(1, F_out, 1)  # same intrinsics for all output frames

    device = "cuda"
    model = model.to(device)

    # NOTE: Inference with MoVieS
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        # `backbone_outputs`: static attributes, including "depth", "color", "scale", "rotation", "opacity", etc.
        # `pred_motions`: (B, F_out, F_in, 3 (xyz) + 1 (conf), H, W)
        # `pred_motion_gs`: a list of `F_out` dict of (B, F_in, C, H, W)
        backbone_outputs, pred_motions, pred_motion_gs = \
            model.backbone(
                input_images.to(device=device, dtype=torch.bfloat16),
                input_C2W.to(device=device, dtype=torch.bfloat16),
                input_fxfycxcy.to(device=device, dtype=torch.bfloat16),
                input_timesteps.to(device=device, dtype=torch.bfloat16),
                output_timesteps.to(device=device, dtype=torch.bfloat16),
                frames_chunk_size=16,
            )

    volume_stats = compute_volume_stats(model, backbone_outputs, pred_motion_gs, output_timesteps)
    save_volume_plot(f"{output_dir}/volume_drift.png", volume_stats)

    canonical_export_count = export_gaussians_as_ply(
        model,
        get_frame_model_outputs(backbone_outputs),
        input_C2W.to(device=device, dtype=torch.bfloat16),
        input_fxfycxcy.to(device=device, dtype=torch.bfloat16),
        input_C2W[:, 0:1, ...].to(device=device, dtype=torch.bfloat16),
        output_fxfycxcy[:, 0:1, ...].to(device=device, dtype=torch.bfloat16),
        f"{gaussian_dir}/canonical.ply",
    )
    time_last_export_count = export_gaussians_as_ply(
        model,
        get_frame_model_outputs(backbone_outputs, pred_motions, pred_motion_gs, F_out - 1),
        input_C2W.to(device=device, dtype=torch.bfloat16),
        input_fxfycxcy.to(device=device, dtype=torch.bfloat16),
        input_C2W[:, 0:1, ...].to(device=device, dtype=torch.bfloat16),
        output_fxfycxcy[:, 0:1, ...].to(device=device, dtype=torch.bfloat16),
        f"{gaussian_dir}/time_last.ply",
    )
    volume_stats["exported_splats"] = {
        "canonical": {
            "path": f"{gaussian_dir}/canonical.ply",
            "num_points": canonical_export_count,
        },
        "time_last": {
            "path": f"{gaussian_dir}/time_last.ply",
            "num_points": time_last_export_count,
        },
    }
    volume_stats["export_alignment_check"] = {
        "canonical_dense_visible_splats": volume_stats["canonical_visible_dense_splats"],
        "canonical_exported_splats": canonical_export_count,
        "canonical_count_match": volume_stats["canonical_visible_dense_splats"] == canonical_export_count,
        "last_timestep_dense_visible_splats": int(volume_stats["visible_dense_splats_per_frame"][-1]),
        "last_timestep_exported_splats": time_last_export_count,
        "last_timestep_count_match": int(volume_stats["visible_dense_splats_per_frame"][-1]) == time_last_export_count,
    }
    with open(f"{output_dir}/volume_stats.json", "w") as f:
        json.dump(volume_stats, f, indent=2)

    iio.mimwrite(f"{output_dir}/output_depth.mp4", tensor_to_video(
        colorize_depth(1./backbone_outputs["depth"], batch_mode=True)), macro_block_size=1)

    # NOTE: Render the output dynamic 3DGS with desired timestep and camera parameters

    # 1. Fix at the first camera, moving timesteps
    render_outputs: Dict[str, Tensor] = {}
    render_outputs_list: List[Dict[str, Tensor]] = []
    for i in range(F_out):  # for different output timesteps
        frame_outputs = get_frame_model_outputs(backbone_outputs, pred_motions, pred_motion_gs, i)
        _render_outputs = model.gs_renderer.render(
            frame_outputs,  # a dict of (B, F_in, C, H, W)
            input_C2W.to(device=device, dtype=torch.bfloat16),  # (B, F_in, 4, 4)
            input_fxfycxcy.to(device=device, dtype=torch.bfloat16),  # (B, F_in, 4)
            # NOTE: Target C2W and fxfycxcy for rendering; fixed to the first camera here
            input_C2W[:, 0:1, ...].to(device=device, dtype=torch.bfloat16),  # (B, 1, 4, 4): one view corresponding to one timestep
            output_fxfycxcy[:, 0:1, ...].to(device=device, dtype=torch.bfloat16),  # (B, 1, 4)
        )  # a dict of (B, 1, C, H, W): one view corresponding to one timestep
        render_outputs_list.append(_render_outputs)

    for k in render_outputs_list[0].keys():
        if k not in ["gaussian_usage", "voxel_ratio"]:
            render_outputs[k] = torch.cat([render_outputs_list[i][k] for i in range(F_out)], dim=1)  # (B, F_out, C, H, W)
    render_images = render_outputs["image"]  # (B, F_out, 3, H, W)

    iio.mimwrite(f"{output_dir}/output_render_camera0.mp4", tensor_to_video(render_images), macro_block_size=1)
    iio.mimwrite(f"{output_dir}/output_motion_camera0.mp4", tensor_to_video(rearrange(normalize_among_last_dims(
        rearrange(pred_motions[:, 0, :, :3, ...], "b f c h w -> b c f h w"), num_dims=3), "b c f h w -> b f c h w")), macro_block_size=1)
    iio.mimwrite(f"{output_dir}/output_motion_time0.mp4", tensor_to_video(rearrange(normalize_among_last_dims(
        rearrange(pred_motions[:, :, 0, :3, ...], "b f c h w -> b c f h w"), num_dims=3), "b c f h w -> b f c h w")), macro_block_size=1)

    # 2. Fix at the last timestep, moving cameras
    render_outputs: Dict[str, Tensor] = {}
    render_outputs_list: List[Dict[str, Tensor]] = []
    for i in range(F_out):  # for different output timesteps
        frame_outputs = get_frame_model_outputs(backbone_outputs, pred_motions, pred_motion_gs, F_out - 1)
        _render_outputs = model.gs_renderer.render(
            frame_outputs,  # a dict of (B, F_in, C, H, W)
            input_C2W.to(device=device, dtype=torch.bfloat16),  # (B, F_in, 4, 4)
            input_fxfycxcy.to(device=device, dtype=torch.bfloat16),  # (B, F_in, 4)
            # NOTE: Target C2W and fxfycxcy for rendering; moving cameras here
            input_C2W[:, i:i+1, ...].to(device=device, dtype=torch.bfloat16),  # (B, 1, 4, 4): one view corresponding to one timestep
            output_fxfycxcy[:, i:i+1, ...].to(device=device, dtype=torch.bfloat16),  # (B, 1, 4)
        )  # a dict of (B, 1, C, H, W): one view corresponding to one timestep
        render_outputs_list.append(_render_outputs)

    for k in render_outputs_list[0].keys():
        if k not in ["gaussian_usage", "voxel_ratio"]:
            render_outputs[k] = torch.cat([render_outputs_list[i][k] for i in range(F_out)], dim=1)  # (B, F_out, C, H, W)
    render_images = render_outputs["image"]  # (B, F_out, 3, H, W)

    iio.mimwrite(f"{output_dir}/output_render_time-1.mp4", tensor_to_video(render_images), macro_block_size=1)
    concat_videos(
        [
            f"{output_dir}/output_render_camera0.mp4",
            f"{output_dir}/output_render_time-1.mp4",
        ],
        f"{output_dir}/output_render.mp4",
    )


if __name__ == "__main__":
    main(parse_args())
