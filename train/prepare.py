"""Build deterministic final-recipe mixtures without modifying source annotations."""

import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import random

from ..data import TEMPLATE_PRESETS, parse_box, read_json, sha256, write_json
from .prompts import build_prompt as training_prompt
from ..tasks import TRAIN_KEYS, prompt_sample
from ..templates import render_template
from ..utils import digest

OPEN_KEYS = ("vg", "oc", "mig", "mmr_oc", "mmr_vg", "mvr_vg")


def training_box(value):
    values = json.loads(value) if isinstance(value, str) else list(value)
    parse_box(values)
    # Retain integer/float spelling in the original LM target serialization.
    return values


def sample_passes(rows, rate, rng):
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("Sampling rates must be finite and nonnegative")
    n = int(len(rows) * rate + 0.5)
    if n and not rows:
        raise ValueError("Cannot sample an empty pool")
    out = []
    while len(out) < n:
        cycle = list(rows)
        rng.shuffle(cycle)
        out.extend(cycle[: n - len(out)])
    return out


def read_rows(path):
    obj = read_json(path)
    if isinstance(obj, dict) and obj.get("split") not in (None, "train"):
        raise ValueError(f"Refusing to train on split={obj['split']!r}: {path}")
    rows = obj.get("results") if isinstance(obj, dict) else obj
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a nonempty JSON list or results list: {path}")
    return rows


def normalize_row(raw, leaf, root):
    row = prompt_sample(raw)
    if "images" in raw:
        images = [im["path"] if isinstance(im, dict) else im for im in raw["images"]]
    elif leaf == "MIG":
        images = raw["uav_path_list"]
    elif leaf.startswith("MMR-"):
        images = [raw["rgb_path"], raw["ir_path"]]
    elif leaf.startswith("MVR-"):
        images = [raw["image1_path"], raw["image2_path"]]
    else:
        images = [raw.get("uav_path") or raw.get("image_path")]
    expected = 4 if leaf == "MIG" else 2 if leaf.startswith(("MMR-", "MVR-")) else 1
    if len(images) != expected or not all(isinstance(p, str) and p for p in images):
        raise ValueError(
            f"{leaf}/{row.get('question_id')}: expected {expected} image paths"
        )
    row["images"] = [str((root / p).resolve()) for p in images]
    if not row.get("question_id") or not row.get("question"):
        raise ValueError(f"Missing question ID or text in {leaf}")
    if leaf == "CJ":
        answer = str(row["correct_answer"]).strip().lower()
        row["correct_answer"] = {
            "yes": "Yes",
            "true": "Yes",
            "no": "No",
            "false": "No",
        }.get(answer)
        if row["correct_answer"] is None:
            raise ValueError(f"Invalid CJ answer: {answer}")
    elif leaf != "OD":
        answer = str(row["correct_answer"]).strip().upper()
        if answer not in row.get("options", {}) or any(
            not str(v).strip() for v in row["options"].values()
        ):
            raise ValueError(f"{leaf}/{row['question_id']}: invalid answer/options")
        row["correct_answer"] = answer
    return row


def choice_record(raw, leaf, templates, template_input):
    key = TRAIN_KEYS[leaf]
    images = list(raw["images"])
    if leaf in TEMPLATE_PRESETS and template_input == "raw":
        reference = raw.get("template", {}).get("box")
        if reference is None:
            reference = parse_box(raw["template_coordinates"][0])
        config = dict(TEMPLATE_PRESETS[leaf], box=reference)
        name = digest(
            {"source": images[0], "config": config, "source_sha256": sha256(images[0])}
        )
        target = templates / (name + ".jpg")
        if not target.is_file():
            render_template(images[0], target, config)
        images[0] = str(target)
    text = training_prompt(raw, key)
    return {
        "task": key,
        "joint_task": leaf.split("-")[0].lower(),
        "question_id": str(raw["question_id"]),
        "correct_answer": raw["correct_answer"],
        "view": "choice",
        "aux_kind": "none",
        "images": images,
        "messages": [
            {"role": "user", "content": "<image>" * len(images) + "\n" + text},
            {"role": "assistant", "content": raw["correct_answer"]},
        ],
    }


def open_view(record, raw):
    r = copy.deepcopy(record)
    t = r["task"]
    answer = raw["options"][raw["correct_answer"]]
    if r["correct_answer"] != raw["correct_answer"]:
        raise ValueError(f"Raw/prepared answer mismatch: {t}/{r['question_id']}")
    q = raw["question"]
    intro = {
        "vg": "Locate the object described in the question in this UAV image.",
        "oc": "Count the objects specified in the question in this UAV image.",
        "mmr_oc": "Image 1 is RGB and Image 2 is aligned TIR. Combine both modalities to count the specified objects, including targets visible in only one modality. Do not double-count the same physical object.",
        "mmr_vg": "Image 1 is RGB and Image 2 is aligned TIR. Combine both modalities to locate the specified object. Return its bounding box in Image 1 (RGB) coordinates.",
        "mig": f'Four images are temporally ordered. Track the same physical object marked in Image 1, not a similar distractor. Return its box in Image {raw.get("query_frame", 3)}. Reference box in Image 1: {raw.get("template_coordinates", [None])[0]}.',
        "mvr_vg": f'Two simultaneous UAV views are shown. Match the physical object marked by the red box in Image 1 to Image 2. Reference box in Image 1: {raw.get("template_coordinates", [None])[0]}. Return Image 2 coordinates. If absent or essentially fully occluded, output exactly: not exist. Partial visibility is not absence.',
    }[t]
    if t in ("oc", "mmr_oc"):
        count = float(answer)
        if not math.isfinite(count) or count < 0 or count != int(count):
            raise ValueError(f"Invalid count: {answer}")
        if "gt_count" in raw and float(raw["gt_count"]) != count:
            raise ValueError(f"Count annotation conflict: {r['question_id']}")
        target = str(int(count))
        r["aux_kind"], r["target_count"] = "count", count
        output = "Output only the nonnegative integer count; no explanation."
    else:
        output = "Output only [x1,y1,x2,y2], normalized to [0,1], top-left then bottom-right. Directions refer to the image frame. No explanation."
        if t == "mvr_vg":
            output += " For an absent or essentially fully occluded target output only: not exist."
        if t == "mvr_vg" and str(answer).strip().lower() == "not exist":
            target, r["aux_kind"] = "not exist", "none"
        else:
            b = training_box(answer)
            target = json.dumps(b, separators=(",", ":"))
            r["aux_kind"], r["target_box"] = "box", b
    r["messages"] = [
        {
            "role": "user",
            "content": "<image>" * len(r["images"])
            + "\n"
            + intro
            + "\n\nQuestion: "
            + q
            + "\n\n"
            + output,
        },
        {"role": "assistant", "content": target},
    ]
    r["view"] = "open"
    return r


def od_record(raw, queries):
    values = raw.get("ground_truth", {}).get("boxes", raw.get("correct_answer_boxes"))
    if not isinstance(values, list):
        raise ValueError(f"Missing OD boxes: {raw['question_id']}")
    boxes = [training_box(b) for b in values]
    if len(boxes) > queries:
        return None
    boxes.sort(key=lambda b: ((b[1] + b[3]) / 2, (b[0] + b[2]) / 2))
    return {
        "task": "od",
        "joint_task": "od",
        "question_id": raw["question_id"],
        "view": "open",
        "aux_kind": "od",
        "images": raw["images"],
        "target_boxes": boxes,
        "messages": [
            {
                "role": "user",
                "content": '<image>\nDetect every object satisfying the question. Output only JSON {"boxes":[[x1,y1,x2,y2],...]}, with normalized [0,1] coordinates. Include every matching object once; use an empty list if none.\nQuestion: '
                + raw["question"],
            },
            {
                "role": "assistant",
                "content": json.dumps({"boxes": boxes}, separators=(",", ":")),
            },
        ],
    }


def od_source(path):
    for part in Path(path).parts:
        name = part.lower().replace("-", "")
        for dataset in (
            "visdrone",
            "dronevehicle",
            "dronecrowd",
            "mdmt",
            "mdot",
            "dronergbt",
            "animaldrone",
            "dronebird",
            "dronemillion",
        ):
            if (
                name.startswith(dataset)
                or dataset == "dronemillion"
                and name.startswith("m30t")
            ):
                return dataset
    return "unknown"


def select_od(rows, n, seed):
    if n > len(rows):
        raise ValueError(f"Need {n} eligible OD rows, found {len(rows)}")
    strata = defaultdict(list)
    for r in rows:
        count = len(r["target_boxes"])
        strata[
            od_source(r["images"][0]),
            0 if count == 0 else 1 if count <= 10 else 2 if count <= 30 else 3,
        ].append(r)
    rng = random.Random(seed)
    for pool in strata.values():
        rng.shuffle(pool)
    selected = []
    while len(selected) < n:
        for key in sorted(strata):
            if strata[key] and len(selected) < n:
                selected.append(strata[key].pop())
    return selected


def build_mixtures(config, pools, opens, od):
    mixtures = {}
    for index, (name, stage) in enumerate(config["stages"].items()):
        rng = random.Random(config["seed"] + index)
        rows = []
        for task, rate in stage["choice_rates"].items():
            rows.extend(sample_passes(pools[task], rate, rng))
        if name == "s3":
            rng.shuffle(rows)
            rng = random.Random(config["seed"] + index)
        for task, rate in stage["open_rates"].items():
            pool = opens[task]
            if name == "s3":
                # The published final branch sorted each pool by sample ID.
                pool = sorted(pool, key=lambda r: str(r["question_id"]))
                rows.extend(rng.sample(pool, int(len(pool) * rate + 0.5)))
            else:
                rows.extend(sample_passes(pool, rate, rng))
        if name in {"s1", "s2"}:
            rows.extend(od)
        rng.shuffle(rows)
        mixtures[name] = rows
    return mixtures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--sources",
        type=Path,
        required=True,
        help="JSON mapping public task identifiers to training JSON files",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--template-input",
        choices=("raw", "prepared"),
        default="raw",
        help="Use prepared only if input reference images already have red boxes",
    )
    parser.add_argument(
        "--choice-jsonl",
        type=Path,
        help="Optional original prepared corpus; preserves exact existing choice prompts/order/repetitions",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        help="Exclude OD images whose resolved paths occur in this benchmark",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root, out = args.data_root.resolve(), args.output.resolve()
    if out == root or out.is_relative_to(root):
        parser.error("Prepared output must be outside the source data root")
    if out.exists() and any(out.iterdir()) and not args.dry_run:
        parser.error(
            "Choose a new empty output directory; existing prepared data is immutable"
        )
    config, sources = read_json(args.config), read_json(args.sources)
    if set(sources) != set(TRAIN_KEYS) | {"OD"}:
        parser.error("Sources must specify exactly 13 choice subtasks and OD")
    raw_by_task, pools, inputs = {}, {}, {}
    for leaf, relative in sources.items():
        path = (root / relative).resolve()
        inputs[str(path)] = sha256(path)
        rows = [normalize_row(r, leaf, root) for r in read_rows(path)]
        if leaf != "PSJ" and len({r["question_id"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate question IDs in {leaf}")
        raw_by_task[leaf] = rows
    missing = {
        p
        for rows in raw_by_task.values()
        for r in rows
        for p in r["images"]
        if not Path(p).is_file()
    }
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} missing images; examples: {sorted(missing)[:5]}"
        )
    if args.choice_jsonl:
        inputs[str(args.choice_jsonl.resolve())] = sha256(args.choice_jsonl)
        pools = defaultdict(list)
        with args.choice_jsonl.open() as handle:
            for line in handle:
                row = json.loads(line)
                row["view"], row["aux_kind"] = "choice", "none"
                row["images"] = [str((root / p).resolve()) for p in row["images"]]
                if any(not Path(p).is_file() for p in row["images"]):
                    raise FileNotFoundError(
                        f"Prepared choice image missing: {row['question_id']}"
                    )
                pools[row["task"]].append(row)
    else:
        # Dry-run does not draw templates or create output directories.
        for leaf in TRAIN_KEYS:
            pools[TRAIN_KEYS[leaf]] = [
                choice_record(
                    r,
                    leaf,
                    out / "templates",
                    "prepared" if args.dry_run else args.template_input,
                )
                for r in raw_by_task[leaf]
            ]
    if set(pools) != set(TRAIN_KEYS.values()):
        raise ValueError("Choice corpus task set does not match the final recipe")
    opens = {}
    for leaf, key in TRAIN_KEYS.items():
        if key in OPEN_KEYS:
            raw = {str(r["question_id"]): r for r in raw_by_task[leaf]}
            opens[key] = [open_view(r, raw[r["question_id"]]) for r in pools[key]]
    test_paths = set()
    if args.benchmark_root:
        from ..data import ANNOTATION_FILES, load_annotations, resolve_path

        for leaf, relative in ANNOTATION_FILES.items():
            path = args.benchmark_root / "annotations" / relative
            inputs[str(path.resolve())] = sha256(path)
            test_paths.update(
                str(resolve_path(args.benchmark_root, im["path"]))
                for row in load_annotations(args.benchmark_root, leaf)
                for im in row["images"]
            )
    od, excluded = [], Counter()
    for raw in raw_by_task["OD"]:
        if raw["images"][0] in test_paths:
            excluded["test_path_overlap"] += 1
            continue
        row = od_record(raw, config["num_queries"])
        if row is None:
            excluded["over_query_capacity"] += 1
        else:
            od.append(row)
    selected = select_od(od, config["od_samples"], config["seed"])
    mixtures = build_mixtures(config, pools, opens, selected)
    inputs[str(args.config.resolve())] = sha256(args.config)
    inputs[str(args.sources.resolve())] = sha256(args.sources)
    audits = {
        name: {
            "records_per_epoch": len(rows),
            "epochs": config["stages"][name]["epochs"],
            "task_view_counts": dict(
                Counter(r["task"] + "/" + r["view"] for r in rows)
            ),
            "aux_counts": dict(Counter(r["aux_kind"] for r in rows)),
            "od_selected": sum(r["task"] == "od" for r in rows),
            "od_excluded": dict(excluded),
        }
        for name, rows in mixtures.items()
    }
    print(json.dumps(audits, indent=2))
    if args.dry_run:
        return
    for name, rows in mixtures.items():
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "train.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(
            folder / "manifest.json",
            {
                "stage": name,
                "config": config,
                "input_sha256": inputs,
                "output_sha256": sha256(path),
                "audit": audits[name],
            },
        )


if __name__ == "__main__":
    main()
