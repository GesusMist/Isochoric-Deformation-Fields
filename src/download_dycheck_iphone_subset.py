from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from gdown.download_folder import download_folder


IPHONE_FOLDER_ID = "1cBw3CUKu2sWQfc_1LbFZGbpdQyTFzDEX"
ITEM_RE = re.compile(
    r"\[null,&quot;([A-Za-z0-9_-]{20,})&quot;\],null,null,null,&quot;([^&]+)&quot;.*?\[\[\[&quot;([^&]+?)&quot;,null,true\]\]\]",
    re.S,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a minimal DyCheck iPhone subset for local evaluation")
    parser.add_argument("--scene", type=str, required=True, help="DyCheck iPhone scene name")
    parser.add_argument("--data-root", type=str, default="datasets/iphone", help="Root directory for local DyCheck scenes")
    parser.add_argument("--clip-length", type=int, default=65, help="Contiguous clip length before sub-sampling")
    parser.add_argument("--num-frames", type=int, default=13, help="Number of uniformly sampled frames")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for clip selection")
    parser.add_argument("--eval-camera-ids", type=int, nargs="+", default=[1, 2], help="Held-out camera ids")
    parser.add_argument("--include-covisible", action="store_true", help="Also download covisibility masks for held-out eval frames")
    return parser.parse_args()


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8", "ignore")


def fetch_json_file(file_id: str) -> Dict:
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    return json.loads(fetch_text(url))


def download_file(file_id: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)


def list_folder_items(folder_id: str) -> Dict[str, Dict[str, str]]:
    html = fetch_text(f"https://drive.google.com/drive/folders/{folder_id}")
    items: Dict[str, Dict[str, str]] = {}
    for file_id, mime, name in ITEM_RE.findall(html):
        items.setdefault(name, {"id": file_id, "mime": mime})
    return items


def split_consecutive_runs(sorted_values: Sequence[int]) -> List[List[int]]:
    runs: List[List[int]] = []
    for value in sorted(set(int(v) for v in sorted_values)):
        if len(runs) == 0 or value != runs[-1][-1] + 1:
            runs.append([value])
        else:
            runs[-1].append(value)
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
            f'Could not find a contiguous {clip_length}-frame clip shared by source camera 0 and eval cameras {list(eval_camera_ids)}'
        )

    rng = np.random.default_rng(seed)
    run = runs[int(rng.integers(len(runs)))]
    start_offset = int(rng.integers(len(run) - clip_length + 1))
    return run[start_offset:start_offset + clip_length]


def sample_time_ids_from_clip(clip_time_ids: Sequence[int], num_frames: int) -> List[int]:
    indices = np.linspace(0, len(clip_time_ids) - 1, num_frames, dtype=int)
    return [int(clip_time_ids[idx]) for idx in indices]


def build_frame_name_map(dataset_dict: Dict, metadata_dict: Dict) -> Dict[Tuple[int, int], str]:
    return {
        (int(metadata_dict[frame_name]["warp_id"]), int(metadata_dict[frame_name]["camera_id"])): frame_name
        for frame_name in dataset_dict["ids"]
    }


def ensure_metadata(scene_dir: Path, scene_items: Dict[str, Dict[str, str]]) -> Tuple[Dict, Dict, Dict]:
    metadata_names = ["dataset.json", "metadata.json", "scene.json", "extra.json"]
    loaded: Dict[str, Dict] = {}
    for name in metadata_names:
        path = scene_dir / name
        if not path.exists():
            path.write_text(json.dumps(fetch_json_file(scene_items[name]["id"]), indent=2))
        loaded[name] = json.loads(path.read_text())
    return loaded["dataset.json"], loaded["metadata.json"], loaded["extra.json"]


def build_manifest_map(folder_id: str) -> Dict[str, str]:
    files = download_folder(id=folder_id, quiet=True, skip_download=True)
    return {
        file.path: file.id
        for file in files
        if file.id is not None
    }


def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    scene_root_items = list_folder_items(IPHONE_FOLDER_ID)
    if args.scene not in scene_root_items:
        raise KeyError(f"Unknown DyCheck iPhone scene: {args.scene}")

    scene_dir = data_root / args.scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_items = list_folder_items(scene_root_items[args.scene]["id"])
    dataset_dict, metadata_dict, extra_dict = ensure_metadata(scene_dir, scene_items)

    frame_name_map = build_frame_name_map(dataset_dict, metadata_dict)
    clip_time_ids = select_clip_time_ids(frame_name_map, args.eval_camera_ids, args.clip_length, args.seed)
    sampled_time_ids = sample_time_ids_from_clip(clip_time_ids, args.num_frames)

    factor = int(extra_dict["factor"])
    camera_files = build_manifest_map(scene_items["camera"]["id"])
    rgb_items = list_folder_items(scene_items["rgb"]["id"])
    rgb_files = build_manifest_map(rgb_items[f"{factor}x"]["id"])
    covisible_files: Dict[str, str] = {}
    if args.include_covisible:
        if "covisible" not in scene_items:
            raise FileNotFoundError(f"Scene {args.scene} does not expose a covisible folder")
        covisible_items = list_folder_items(scene_items["covisible"]["id"])
        covisible_files = build_manifest_map(covisible_items[f"{factor}x"]["id"])

    selected_camera_files: Dict[str, str] = {}
    selected_rgb_files: Dict[str, str] = {}
    selected_covisible_files: Dict[str, str] = {}
    for camera_id in [0, *args.eval_camera_ids]:
        for time_id in sampled_time_ids:
            frame_name = frame_name_map[(int(time_id), int(camera_id))]
            camera_name = frame_name + ".json"
            rgb_name = frame_name + ".png"
            if camera_name not in camera_files:
                raise KeyError(f"Missing camera file id for {camera_name}")
            if rgb_name not in rgb_files:
                raise KeyError(f"Missing rgb file id for {rgb_name}")

            selected_camera_files[camera_name] = camera_files[camera_name]
            selected_rgb_files[rgb_name] = rgb_files[rgb_name]
            download_file(camera_files[camera_name], scene_dir / "camera" / camera_name)
            download_file(rgb_files[rgb_name], scene_dir / "rgb" / f"{factor}x" / rgb_name)
            if args.include_covisible and camera_id != 0:
                covisible_relpath = f"val/{rgb_name}"
                if covisible_relpath not in covisible_files:
                    raise KeyError(f"Missing covisible file id for {covisible_relpath}")
                selected_covisible_files[covisible_relpath] = covisible_files[covisible_relpath]
                download_file(
                    covisible_files[covisible_relpath],
                    scene_dir / "covisible" / f"{factor}x" / "val" / rgb_name,
                )

    manifest = {
        "scene": args.scene,
        "clip_length": int(args.clip_length),
        "num_frames": int(args.num_frames),
        "seed": int(args.seed),
        "source_camera_id": 0,
        "eval_camera_ids": [int(camera_id) for camera_id in args.eval_camera_ids],
        "clip_time_ids": clip_time_ids,
        "sampled_time_ids": sampled_time_ids,
        "camera_files": selected_camera_files,
        f"rgb_{factor}x_files": selected_rgb_files,
    }
    if args.include_covisible:
        manifest[f"covisible_{factor}x_val_files"] = selected_covisible_files
    (scene_dir / "selected_eval_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps({
        "scene": args.scene,
        "factor": factor,
        "clip_time_ids": clip_time_ids,
        "sampled_time_ids": sampled_time_ids,
        "camera_file_count": len(selected_camera_files),
        "rgb_file_count": len(selected_rgb_files),
        "covisible_file_count": len(selected_covisible_files),
        "scene_dir": str(scene_dir),
    }, indent=2))


if __name__ == "__main__":
    main(parse_args())
