import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List

import requests


SCENE_URLS: Dict[str, str] = {
    "Jumping": "https://www.dropbox.com/sh/ws9khkjv7vnyub2/AADsQ5H8ixc_yEsiNarsjGhBa?dl=0",
    "Skating": "https://www.dropbox.com/sh/pp1phzamaxyl60j/AADU9y9LQSWmMpkP9yckE1ZKa?dl=0",
    "Truck": "https://www.dropbox.com/sh/svglgn553dei9dd/AAAy1lwNv29FCJ8eQzjR695Ma?dl=0",
    "DynamicFace": "https://www.dropbox.com/sh/z90byp14nxfbsbw/AAAjFRxHEwVi98gbV6gHjfLLa?dl=0",
    "Umbrella": "https://www.dropbox.com/sh/xbp9coi0lee80qy/AADG9F6fmdeQlxCZHkIUgfz0a?dl=0",
    "Balloon1": "https://www.dropbox.com/sh/qlhpjitoakghb1d/AADmUgq4nnEuVDbWOaxpjhnma?dl=0",
    "Balloon2": "https://www.dropbox.com/sh/n5w7a1evbnmxzib/AAAH09_LJ8PHfLjqOdqSjsoAa?dl=0",
    "Teadybear": "https://www.dropbox.com/sh/0182byj05p20h1p/AACPG1IxyCPVB9EOFsD_Sk39a?dl=0",
    "Playground": "https://www.dropbox.com/sh/eg04jm1wcn5enez/AAC1vDqtqhTFOqVIaUGNiJC0a?dl=0",
}

INNER_ARCHIVES = [
    "calibration.zip",
    "depth_GT.zip",
    "foreground_mask.zip",
    "input_images.zip",
    "multiview_GT.zip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the official NVIDIA dynamic scene benchmark archives")
    parser.add_argument(
        "--scene",
        type=str,
        nargs="+",
        choices=sorted(SCENE_URLS.keys()),
        help="One or more benchmark scenes to download",
    )
    parser.add_argument("--all-scenes", action="store_true", help="Download all benchmark scenes")
    parser.add_argument("--data-root", type=str, default="datasets/nvidia", help="Local dataset root")
    parser.add_argument("--skip-existing", action="store_true", help="Skip scenes that already look extracted")
    parser.add_argument("--keep-archives", action="store_true", help="Keep downloaded and extracted zip archives")
    return parser.parse_args()


def resolve_scenes(args: argparse.Namespace) -> List[str]:
    if args.all_scenes:
        return list(SCENE_URLS.keys())
    if args.scene:
        return args.scene
    raise ValueError("Specify --scene <name> ... or --all-scenes")


def ensure_download_url(url: str) -> str:
    if url.endswith("dl=0"):
        return url[:-1] + "1"
    if "dl=" not in url:
        suffix = "&" if "?" in url else "?"
        return url + suffix + "dl=1"
    return url


def scene_is_ready(scene_dir: Path) -> bool:
    multiview_dir = scene_dir / "multiview_GT"
    if not (multiview_dir / "00000012").exists() and (multiview_dir / "multiview_GT" / "00000012").exists():
        multiview_dir = multiview_dir / "multiview_GT"
    return (
        (scene_dir / "input_images" / "cam01.jpg").exists()
        and (multiview_dir / "00000012" / "cam12.jpg").exists()
        and (scene_dir / "calibration" / "cam01" / "intrinsic.txt").exists()
    )


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    with requests.get(url, stream=True, timeout=(30, 300), allow_redirects=True) as response:
        response.raise_for_status()
        total_bytes = int(response.headers.get("content-length", 0))
        downloaded_bytes = 0
        report_every_bytes = 10 * 1024 * 1024
        next_report = report_every_bytes

        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded_bytes += len(chunk)
                if downloaded_bytes >= next_report:
                    if total_bytes > 0:
                        progress = 100.0 * downloaded_bytes / total_bytes
                        print(
                            f"[NVIDIA download] {output_path.name} "
                            f"{downloaded_bytes / (1024 ** 2):.1f} / {total_bytes / (1024 ** 2):.1f} MB "
                            f"({progress:.1f}%)",
                            flush=True,
                        )
                    else:
                        print(
                            f"[NVIDIA download] {output_path.name} "
                            f"{downloaded_bytes / (1024 ** 2):.1f} MB",
                            flush=True,
                        )
                    next_report += report_every_bytes

    temp_path.replace(output_path)


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(output_dir)


def flatten_redundant_root(extract_dir: Path) -> None:
    nested_dir = extract_dir / extract_dir.name
    if not nested_dir.is_dir():
        return
    for child in nested_dir.iterdir():
        child.rename(extract_dir / child.name)
    nested_dir.rmdir()


def extract_scene_archives(scene_dir: Path, keep_archives: bool) -> None:
    for archive_name in INNER_ARCHIVES:
        archive_path = scene_dir / archive_name
        extract_dir = scene_dir / archive_name.replace(".zip", "")
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing extracted inner archive: {archive_path}")
        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            print(f"[NVIDIA download] extracting {archive_path}", flush=True)
            extract_zip(archive_path, extract_dir)
            flatten_redundant_root(extract_dir)
        if not keep_archives:
            archive_path.unlink(missing_ok=True)


def download_scene(scene: str, data_root: Path, skip_existing: bool, keep_archives: bool) -> Dict[str, object]:
    scene_dir = data_root / scene
    raw_dir = data_root / "raw"
    archive_path = raw_dir / f"{scene}.zip"

    if skip_existing and scene_is_ready(scene_dir):
        print(f"[NVIDIA download] scene={scene} status=skip_existing", flush=True)
        return {
            "scene": scene,
            "status": "skip_existing",
            "scene_dir": str(scene_dir),
        }

    scene_dir.mkdir(parents=True, exist_ok=True)
    url = ensure_download_url(SCENE_URLS[scene])

    print(f"[NVIDIA download] scene={scene} stage=download url={url}", flush=True)
    download_file(url, archive_path)

    print(f"[NVIDIA download] scene={scene} stage=extract_outer archive={archive_path}", flush=True)
    extract_zip(archive_path, scene_dir)
    extract_scene_archives(scene_dir, keep_archives=keep_archives)

    if not keep_archives:
        archive_path.unlink(missing_ok=True)

    if not scene_is_ready(scene_dir):
        raise RuntimeError(f"Scene extraction looks incomplete: {scene_dir}")

    return {
        "scene": scene,
        "status": "downloaded",
        "scene_dir": str(scene_dir),
        "kept_archives": bool(keep_archives),
    }


def main(args: argparse.Namespace) -> None:
    scenes = resolve_scenes(args)
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    results = []
    for scene in scenes:
        results.append(
            download_scene(
                scene,
                data_root,
                skip_existing=args.skip_existing,
                keep_archives=args.keep_archives,
            )
        )

    summary = {
        "data_root": str(data_root),
        "scene_count": len(results),
        "scenes": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(parse_args())
