"""Portable, model-independent readers for the Drone-PUR benchmark."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath


TOP_FIELDS = ("dataset", "split", "task", "subtask", "total_samples", "results")
ANNOTATION_FILES = {
    "OD": "OD/test.json",
    "OC": "OC/test.json",
    "OR": "OR/test.json",
    "MIG": "MIG/test.json",
    "CJ": "CJ/test.json",
    "VG": "VG/test.json",
    "PSJ": "PSJ/test.json",
    "SI": "SI/test.json",
    "CR": "CR/test.json",
    "MMR-OC": "MMR/test_oc.json",
    "MMR-VG": "MMR/test_vg.json",
    "MMR-R": "MMR/test_reasoning.json",
    "MVR-VG": "MVR/test_vg.json",
    "MVR-R": "MVR/test_reasoning.json",
}
OPEN_TASKS = {"OD", "OC", "VG", "MIG", "MMR-OC", "MMR-VG", "MVR-VG"}
TEMPLATE_PRESETS = {
    "MIG": {
        "image_index": 1,
        "padding_ratio": 0.002,
        "line_width_ratio": 0.004,
        "min_line_width": 4,
        "jpeg_quality": 95,
    },
    "MVR-VG": {
        "image_index": 1,
        "padding_ratio": 0.002,
        "line_width_ratio": 0.006,
        "min_line_width": 4,
        "jpeg_quality": 95,
    },
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pending.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(root, relative):
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError(f"Expected a root-relative POSIX path: {relative!r}")
    root = Path(root).resolve()
    target = (root / str(path)).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Path escapes the benchmark root: {relative!r}")
    return target


def parse_box(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Expected one normalized xyxy box: {value!r}")
    box = list(map(float, value))
    if not all(math.isfinite(number) for number in box):
        raise ValueError(f"Non-finite box: {value!r}")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"Invalid normalized xyxy box: {value!r}")
    return box


def leaf_name(row):
    return row["task"] + ("-" + row["subtask"] if row.get("subtask") else "")


def template_draw_config(root, row):
    """Keep only the sample-specific box in annotations; settings live once."""
    presets = read_json(Path(root) / "metadata/template_config.json")
    config = dict(presets[leaf_name(row)])
    if config["image_index"] != 1:
        raise ValueError(
            "Templates must refer to the first image, never the query frame"
        )
    config["box"] = parse_box(row["template"]["box"])
    return config


def get_ground_truth(row):
    """Pure-choice and Yes/No GT is derived, not duplicated in stored records."""
    if "ground_truth" in row:
        return row["ground_truth"]
    if row.get("task") == "CJ":
        return {"type": "boolean", "value": row["correct_answer"] == "Yes"}
    return {"type": "choice", "option": row["correct_answer"]}


def load_annotations(root, task):
    root = Path(root)
    if task not in ANNOTATION_FILES:
        raise ValueError(
            f"Unknown task: {task}; supported tasks: {', '.join(ANNOTATION_FILES)}"
        )
    payload = read_json(resolve_path(root, "annotations/" + ANNOTATION_FILES[task]))
    if set(payload) != set(TOP_FIELDS) or leaf_name(payload) != task:
        raise ValueError("Unexpected annotation header fields")
    if payload["total_samples"] != len(payload["results"]):
        raise ValueError("Annotation sample count does not match total_samples")
    # Context is supplied in memory, without repeating it in every JSON record.
    return [
        dict(row, task=payload["task"], subtask=payload["subtask"])
        for row in payload["results"]
    ]


def model_image_paths(root, row, prepared_dir=None, require_templates=True):
    """Return ordered model inputs; never redraw a box onto a generated image."""
    root = Path(root)
    paths = [str(resolve_path(root, image["path"])) for image in row["images"]]
    if row.get("template") and require_templates:
        prepared = Path(prepared_dir) if prepared_dir else root / "prepared_images"
        index = read_json(prepared / "templates.json")
        record = index["records"].get(leaf_name(row) + "/" + row["question_id"])
        if record is None:
            raise FileNotFoundError(
                "Template not prepared; run tools/prepare_templates.py first"
            )
        config = template_draw_config(root, row)
        image_index = config["image_index"] - 1
        if (
            record.get("source_path") != row["images"][image_index]["path"]
            or record["draw_config"] != config
        ):
            raise ValueError("Prepared template does not match the current sample")
        target = resolve_path(prepared, record["path"])
        if not target.is_file():
            raise FileNotFoundError(target)
        paths[image_index] = str(target)
    if any(not Path(path).is_file() for path in paths):
        raise FileNotFoundError("A model input image is missing")
    return paths
