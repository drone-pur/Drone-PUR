#!/usr/bin/env python3
"""Draw the original MIG/MVR-VG template-box protocols into a separate cache."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re

from PIL import Image, ImageDraw

from .data import (
    load_annotations,
    read_json,
    resolve_path,
    sha256,
    template_draw_config,
    write_json,
)


def render_template(source, destination, config):
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    x1, y1, x2, y2 = config["box"]
    pad = config["padding_ratio"]
    box = [max(0, x1 - pad), max(0, y1 - pad), min(1, x2 + pad), min(1, y2 + pad)]
    pixels = [
        int(box[0] * image.width),
        int(box[1] * image.height),
        int(box[2] * image.width),
        int(box[3] * image.height),
    ]
    width = max(
        config["min_line_width"], int(min(image.size) * config["line_width_ratio"])
    )
    ImageDraw.Draw(image).rectangle(pixels, outline=(255, 0, 0), width=width)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name + ".pending")
    image.save(pending, format="JPEG", quality=config["jpeg_quality"])
    pending.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--tasks", nargs="+", choices=["MIG", "MVR-VG"], default=["MIG", "MVR-VG"]
    )
    parser.add_argument(
        "--limit", type=int, help="Diagnostic only: maximum samples per selected task"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    root = args.data_root.resolve()
    output = args.output_dir.resolve() if args.output_dir else root / "prepared_images"
    if (
        output == root
        or output.is_relative_to(root / "images")
        or output.is_relative_to(root / "annotations")
    ):
        parser.error(
            "Generated templates must not be written into original images or annotations"
        )
    wanted = [
        (task, row)
        for task in args.tasks
        for row in load_annotations(root, task)[: args.limit]
    ]
    if args.dry_run:
        for task in args.tasks:
            print(f"{task}: {sum(key == task for key, _ in wanted)} template images")
        print(f"Output: {output}; original images remain unchanged")
        return
    index_path = output / "templates.json"
    index = read_json(index_path) if index_path.is_file() else {"records": {}}
    image_info = {
        row["path"]: row for row in read_json(root / "metadata/image_manifest.json")
    }
    generated = reused = 0
    for step, (task, row) in enumerate(wanted, 1):
        config = template_draw_config(root, row)
        raw = row["images"][config["image_index"] - 1]
        source = resolve_path(root, raw["path"])
        if sha256(source) != image_info[raw["path"]]["sha256"]:
            raise ValueError(f"Original image checksum mismatch: {source}")
        key = task + "/" + row["question_id"]
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", row["question_id"])
        relative = f"{task}/{safe_id}.jpg"
        destination = resolve_path(output, relative)
        previous = index["records"].get(key, {})
        fresh = (
            previous.get("draw_config") == config
            and previous.get("source_sha256") == image_info[raw["path"]]["sha256"]
            and destination.is_file()
            and sha256(destination) == previous.get("sha256")
        )
        if fresh:
            reused += 1
        else:
            render_template(source, destination, config)
            generated += 1
        index["records"][key] = {
            "source_path": raw["path"],
            "question_id": row["question_id"],
            "task": task,
            "path": relative,
            "draw_config": config,
            "source_sha256": image_info[raw["path"]]["sha256"],
            "sha256": sha256(destination),
        }
        if step % 50 == 0 or step == len(wanted):
            index["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(index_path, index)
            print(
                f"{step}/{len(wanted)}: generated={generated}, reused={reused}",
                flush=True,
            )
    print("Prepared templates are separate derivatives, not replacements for images/.")


if __name__ == "__main__":
    main()
