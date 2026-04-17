import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence


DYCHECK_IPHONE_SCENES = [
    "apple",
    "backpack",
    "block",
    "creeper",
    "handwavy",
    "haru-sit",
    "mochi-high-five",
    "paper-windmill",
    "pillow",
    "space-out",
    "spin",
    "sriracha-tree",
    "teddy",
    "wheel",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-download and evaluate DyCheck iPhone scenes with MoVieS")
    parser.add_argument("--data-root", type=str, default="datasets/iphone", help="Root directory containing DyCheck iPhone scenes")
    parser.add_argument("--ckpt-path", type=str, default="resources/movies_ckpt.safetensors", help="Path to the MoVieS checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--image-size", type=int, default=518, help="Square evaluation resolution")
    parser.add_argument("--clip-length", type=int, default=65, help="Contiguous clip length before sub-sampling")
    parser.add_argument("--num-frames", type=int, default=13, help="Number of uniformly sampled frames")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for clip sampling")
    parser.add_argument("--eval-camera-ids", type=int, nargs="+", default=[1, 2], help="Held-out camera ids for evaluation")
    parser.add_argument(
        "--metric-mask",
        type=str,
        choices=["none", "covisible", "both"],
        default="both",
        help="Evaluate with no mask, only the covisible mask, or both",
    )
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading and only run evaluation")
    parser.add_argument("--output-root", type=str, default="out/dycheck", help="Root directory for evaluation outputs")
    parser.add_argument("--summary-json", type=str, default=None, help="Optional summary json path")
    parser.add_argument("--summary-csv", type=str, default=None, help="Optional summary csv path")
    parser.add_argument("--scenes", type=str, nargs="*", default=DYCHECK_IPHONE_SCENES, help="Subset of scenes to process")
    return parser.parse_args()


def default_scene_output_path(output_root: str, scene: str, metric_mask: str) -> str:
    filename = {
        "none": "dycheck_metrics.json",
        "covisible": "dycheck_metrics_covisible.json",
        "both": "dycheck_metrics_both.json",
    }[metric_mask]
    return os.path.join(output_root, scene, filename)


def default_summary_path(output_root: str, metric_mask: str, suffix: str) -> str:
    return os.path.join(output_root, f"dycheck_metrics_{metric_mask}_all.{suffix}")


def split_consecutive_runs(sorted_values: Sequence[int]) -> List[List[int]]:
    runs: List[List[int]] = []
    for value in sorted(set(int(v) for v in sorted_values)):
        if not runs or value != runs[-1][-1] + 1:
            runs.append([value])
        else:
            runs[-1].append(value)
    return runs


def max_shared_run_length(camera_time_sets: Dict[int, set[int]], camera_ids: Sequence[int]) -> int:
    shared_times = None
    for camera_id in camera_ids:
        if shared_times is None:
            shared_times = set(camera_time_sets[camera_id])
        else:
            shared_times &= camera_time_sets[camera_id]
    if not shared_times:
        return 0
    return max((len(run) for run in split_consecutive_runs(shared_times)), default=0)


def select_scene_protocol(scene_dir: Path, clip_length: int, requested_metric_mask: str) -> Dict[str, Any]:
    with open(scene_dir / "dataset.json", "r") as f:
        dataset_dict = json.load(f)
    with open(scene_dir / "metadata.json", "r") as f:
        metadata_dict = json.load(f)

    camera_time_sets: Dict[int, set[int]] = {}
    for frame_name in dataset_dict["ids"]:
        metadata = metadata_dict[frame_name]
        camera_id = int(metadata["camera_id"])
        time_id = int(metadata["warp_id"])
        camera_time_sets.setdefault(camera_id, set()).add(time_id)

    if 0 not in camera_time_sets:
        raise ValueError(f"Scene {scene_dir.name} does not expose source camera 0")

    other_cameras = sorted(camera_id for camera_id in camera_time_sets if camera_id != 0)

    preferred_eval_sets: List[List[int]] = []
    if {1, 2}.issubset(other_cameras):
        preferred_eval_sets.append([1, 2])
    for count in [2, 1]:
        for camera_ids in combinations(other_cameras, count):
            camera_list = list(camera_ids)
            if camera_list not in preferred_eval_sets:
                preferred_eval_sets.append(camera_list)

    best_candidate: Dict[str, Any] | None = None
    for eval_camera_ids in preferred_eval_sets:
        shared_run = max_shared_run_length(camera_time_sets, [0, *eval_camera_ids])
        if shared_run < clip_length:
            continue
        candidate = {
            "protocol_kind": "held_out_novel_view",
            "eval_camera_ids": eval_camera_ids,
            "metric_mask_mode": requested_metric_mask,
            "max_shared_run": shared_run,
            "note": "",
        }
        if best_candidate is None:
            best_candidate = candidate
            continue
        prev_key = (len(best_candidate["eval_camera_ids"]), best_candidate["max_shared_run"])
        curr_key = (len(candidate["eval_camera_ids"]), candidate["max_shared_run"])
        if curr_key > prev_key:
            best_candidate = candidate
    if best_candidate is not None:
        return best_candidate

    same_view_run = max_shared_run_length(camera_time_sets, [0])
    if same_view_run >= clip_length:
        note = (
            "Fell back to same-view reconstruction because no held-out camera set "
            f"had a contiguous {clip_length}-frame overlap with source camera 0."
        )
        return {
            "protocol_kind": "same_view_reconstruction",
            "eval_camera_ids": [0],
            "metric_mask_mode": "none",
            "max_shared_run": same_view_run,
            "note": note,
        }

    raise ValueError(
        f"Scene {scene_dir.name} does not have a contiguous {clip_length}-frame run even for source camera 0"
    )


def scene_output_path(output_root: str, scene: str, protocol_kind: str, metric_mask_mode: str) -> str:
    if protocol_kind == "same_view_reconstruction":
        return os.path.join(output_root, scene, "dycheck_metrics_same_view.json")
    return default_scene_output_path(output_root, scene, metric_mask_mode)


def run_command(cmd: List[str], retries: int = 1, retry_delay_sec: float = 5.0) -> None:
    for attempt in range(1, retries + 1):
        print(f"[DyCheck Batch] running: {' '.join(cmd)}", flush=True)
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt >= retries:
                raise
            print(
                f"[DyCheck Batch] command failed attempt={attempt}/{retries}; "
                f"retrying in {retry_delay_sec:.1f}s",
                flush=True,
            )
            time.sleep(retry_delay_sec)


def write_summary_json(path: str, summary: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def write_summary_csv(path: str, completed: List[Dict[str, Any]], failures: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene",
        "status",
        "protocol_kind",
        "metric_mask_mode",
        "eval_camera_ids",
        "output_json",
        "overall_mean_psnr",
        "overall_mean_ssim",
        "overall_mean_ssim_x100",
        "overall_mean_lpips",
        "overall_mean_mpsnr",
        "overall_mean_mssim",
        "overall_mean_mssim_x100",
        "overall_mean_mlpips",
        "overall_mean_covisible_coverage",
        "error",
    ]
    rows = []
    for item in completed:
        metrics = item["metrics"]
        rows.append({
            "scene": item["scene"],
            "status": "ok",
            "protocol_kind": item["protocol_kind"],
            "metric_mask_mode": item["metric_mask_mode"],
            "eval_camera_ids": " ".join(str(camera_id) for camera_id in item["eval_camera_ids"]),
            "output_json": item["output_json"],
            "overall_mean_psnr": metrics.get("overall_mean_psnr"),
            "overall_mean_ssim": metrics.get("overall_mean_ssim"),
            "overall_mean_ssim_x100": metrics.get("overall_mean_ssim_x100"),
            "overall_mean_lpips": metrics.get("overall_mean_lpips"),
            "overall_mean_mpsnr": metrics.get("overall_mean_mpsnr"),
            "overall_mean_mssim": metrics.get("overall_mean_mssim"),
            "overall_mean_mssim_x100": metrics.get("overall_mean_mssim_x100"),
            "overall_mean_mlpips": metrics.get("overall_mean_mlpips"),
            "overall_mean_covisible_coverage": metrics.get("overall_mean_covisible_coverage"),
            "error": "",
        })
    for item in failures:
        rows.append({
            "scene": item["scene"],
            "status": "failed",
            "protocol_kind": item.get("protocol_kind", ""),
            "metric_mask_mode": item.get("metric_mask_mode", ""),
            "eval_camera_ids": " ".join(str(camera_id) for camera_id in item.get("eval_camera_ids", [])),
            "output_json": item.get("output_json", ""),
            "overall_mean_psnr": "",
            "overall_mean_ssim": "",
            "overall_mean_ssim_x100": "",
            "overall_mean_lpips": "",
            "overall_mean_mpsnr": "",
            "overall_mean_mssim": "",
            "overall_mean_mssim_x100": "",
            "overall_mean_mlpips": "",
            "overall_mean_covisible_coverage": "",
            "error": item["error"],
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(args: argparse.Namespace) -> int:
    summary_json = args.summary_json or default_summary_path(args.output_root, args.metric_mask, "json")
    summary_csv = args.summary_csv or default_summary_path(args.output_root, args.metric_mask, "csv")

    completed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    batch_start = time.perf_counter()

    for idx, scene in enumerate(args.scenes, start=1):
        scene_start = time.perf_counter()
        output_json = ""
        protocol: Dict[str, Any] = {}
        print(
            f"[DyCheck Batch] scene={scene} index={idx}/{len(args.scenes)} "
            f"metric_mask={args.metric_mask} image_size={args.image_size}",
            flush=True,
        )
        try:
            scene_dir = Path(args.data_root) / scene
            protocol = select_scene_protocol(scene_dir, args.clip_length, args.metric_mask)
            output_json = scene_output_path(
                args.output_root, scene, protocol["protocol_kind"], protocol["metric_mask_mode"]
            )
            print(
                f"[DyCheck Batch] scene={scene} protocol={protocol['protocol_kind']} "
                f"eval_cameras={protocol['eval_camera_ids']} mask={protocol['metric_mask_mode']} "
                f"max_shared_run={protocol['max_shared_run']}",
                flush=True,
            )
            if protocol["note"]:
                print(f"[DyCheck Batch] scene={scene} note={protocol['note']}", flush=True)

            if not args.skip_download:
                download_cmd = [
                    sys.executable,
                    "src/download_dycheck_iphone_subset.py",
                    "--scene", scene,
                    "--data-root", args.data_root,
                    "--clip-length", str(args.clip_length),
                    "--num-frames", str(args.num_frames),
                    "--seed", str(args.seed),
                    "--eval-camera-ids",
                    *[str(camera_id) for camera_id in protocol["eval_camera_ids"]],
                ]
                if protocol["metric_mask_mode"] in {"covisible", "both"}:
                    download_cmd.append("--include-covisible")
                run_command(download_cmd, retries=3)

            eval_cmd = [
                sys.executable,
                "src/eval_dycheck_iphone.py",
                "--scene", scene,
                "--data-root", args.data_root,
                "--ckpt-path", args.ckpt_path,
                "--device", args.device,
                "--image-size", str(args.image_size),
                "--clip-length", str(args.clip_length),
                "--num-frames", str(args.num_frames),
                "--seed", str(args.seed),
                "--eval-camera-ids",
                *[str(camera_id) for camera_id in protocol["eval_camera_ids"]],
                "--metric-mask", protocol["metric_mask_mode"],
                "--output-json", output_json,
            ]
            run_command(eval_cmd, retries=2)

            with open(output_json, "r") as f:
                metrics = json.load(f)
            completed.append({
                "scene": scene,
                "output_json": output_json,
                "elapsed_sec": time.perf_counter() - scene_start,
                "protocol_kind": protocol["protocol_kind"],
                "metric_mask_mode": protocol["metric_mask_mode"],
                "eval_camera_ids": protocol["eval_camera_ids"],
                "note": protocol["note"],
                "metrics": metrics,
            })
        except Exception as exc:
            failures.append({
                "scene": scene,
                "output_json": output_json,
                "elapsed_sec": time.perf_counter() - scene_start,
                "protocol_kind": protocol.get("protocol_kind", ""),
                "metric_mask_mode": protocol.get("metric_mask_mode", ""),
                "eval_camera_ids": protocol.get("eval_camera_ids", []),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })

        summary = {
            "metric_mask_mode": args.metric_mask,
            "image_size": int(args.image_size),
            "clip_length": int(args.clip_length),
            "num_frames": int(args.num_frames),
            "seed": int(args.seed),
            "eval_camera_ids": [int(camera_id) for camera_id in args.eval_camera_ids],
            "checkpoint": args.ckpt_path,
            "data_root": args.data_root,
            "output_root": args.output_root,
            "elapsed_sec": time.perf_counter() - batch_start,
            "scenes_requested": list(args.scenes),
            "completed_count": len(completed),
            "failure_count": len(failures),
            "completed": completed,
            "failures": failures,
        }
        write_summary_json(summary_json, summary)
        write_summary_csv(summary_csv, completed, failures)

    print(
        f"[DyCheck Batch] complete completed={len(completed)} failed={len(failures)} "
        f"elapsed_sec={time.perf_counter() - batch_start:.2f}",
        flush=True,
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
