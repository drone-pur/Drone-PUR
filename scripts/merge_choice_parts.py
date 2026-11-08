"""Merge disjoint local-Qwen choice runs with full, sample-weighted scoring."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from drone_pur.data import read_json, write_json
from drone_pur.evaluate import (
    identity,
    load_samples,
    parser as eval_parser,
    save_results,
)
from drone_pur.utils import output_lock


TASK_FIELDS = {"ids", "annotations_sha256", "max_new_tokens", "input_fingerprints"}


def collect_parts(expected, pairs):
    owners, records, fingerprints = {}, {}, {}
    versions = None
    model_name = None
    common = {key: value for key, value in expected.items() if key not in TASK_FIELDS}
    for part_number, (manifest, rows) in enumerate(pairs, 1):
        info = manifest["identity"]
        if set(info) != set(expected) | {"input_fingerprints"}:
            raise ValueError(
                "Partition identity fields differ from the current evaluator"
            )
        if {
            key: value for key, value in info.items() if key not in TASK_FIELDS
        } != common:
            raise ValueError(
                "Partitions must use identical model weights, code and inference settings"
            )
        if versions is None:
            versions, model_name = manifest["versions"], manifest["model_name"]
        elif versions != manifest["versions"] or model_name != manifest["model_name"]:
            raise ValueError("Partition runtime versions or model labels differ")
        tasks = set(info["ids"])
        if not tasks or tasks - set(expected["ids"]):
            raise ValueError("Partition has no tasks or contains unknown tasks")
        for field in TASK_FIELDS:
            if set(info[field]) != tasks:
                raise ValueError(f"Partition task coverage differs in {field}")
        for task in tasks:
            if task in owners:
                raise ValueError(f"Task appears in multiple partitions: {task}")
            for field in ("ids", "annotations_sha256", "max_new_tokens"):
                if info[field][task] != expected[field][task]:
                    raise ValueError(
                        f"{task}: full-set IDs, annotations or token limits differ"
                    )
            if set(info["input_fingerprints"][task]) != set(expected["ids"][task]):
                raise ValueError(f"{task}: missing or unknown input fingerprints")
            if manifest["selected_counts"].get(task) != len(expected["ids"][task]):
                raise ValueError(
                    f"{task}: selected count does not match the full dataset"
                )
            owners[task] = part_number
            fingerprints[task] = info["input_fingerprints"][task]
        if set(manifest["selected_counts"]) != tasks:
            raise ValueError("Selected-count task coverage differs from the partition")
        if not isinstance(rows, list):
            raise ValueError("Partition results must be a list")
        for row in rows:
            task, qid = row["task"], row["question_id"]
            if task not in tasks or qid not in info["input_fingerprints"][task]:
                raise ValueError(f"Result outside the selected partition: {task}/{qid}")
            key = task, qid
            if key in records:
                raise ValueError(f"Duplicate result: {task}/{qid}")
            if (
                type(row.get("completed")) is not bool
                or row["completed"]
                and "prediction" not in row
            ):
                raise ValueError(f"Invalid completion record: {task}/{qid}")
            records[key] = row
    missing_tasks = set(expected["ids"]) - set(owners)
    if missing_tasks:
        raise ValueError(
            f"Partitions do not cover all choice tasks: {sorted(missing_tasks)}"
        )
    merged = dict(expected, input_fingerprints=fingerprints)
    return merged, records, versions, model_name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--parts", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--max-pixels", type=int)
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    p.add_argument("--device-map", default="auto")
    p.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
    )
    p.add_argument("--revision", default="main")
    p.add_argument("--base-model", type=Path)
    p.add_argument("--text-only", action="store_true")
    args = p.parse_args()
    try:
        output = args.output.resolve()
        for source in (args.model, args.base_model):
            if source is not None:
                model = source.resolve()
                if (
                    output == model
                    or output.is_relative_to(model)
                    or model.is_relative_to(output)
                ):
                    raise ValueError(
                        "Merged statistics must be separate from model directories"
                    )
        parts = [path.resolve() for path in args.parts]
        if len(parts) != len(set(parts)) or any(
            output == path or output.is_relative_to(path) or path.is_relative_to(output)
            for path in parts
        ):
            raise ValueError(
                "Source partitions and merged output must be distinct directories"
            )
        root = args.data_root.resolve()
        if output == root or any(
            output.is_relative_to(root / name)
            for name in ("images", "annotations", "metadata")
        ):
            raise ValueError("Merged output must not overwrite benchmark data")
        eval_args = eval_parser().parse_args(
            [
                "--model",
                str(args.model),
                "--data-root",
                str(root),
                "--output",
                str(output),
                "--mode",
                "choice",
                "--tasks",
                "all",
                "--dtype",
                args.dtype,
                "--device-map",
                args.device_map,
                "--attn-implementation",
                args.attn_implementation,
                "--revision",
                args.revision,
                *(
                    ["--max-pixels", str(args.max_pixels)]
                    if args.max_pixels is not None
                    else []
                ),
                *(
                    ["--max-new-tokens", str(args.max_new_tokens)]
                    if args.max_new_tokens is not None
                    else []
                ),
                *(["--base-model", str(args.base_model)] if args.base_model else []),
                *(["--text-only"] if args.text_only else []),
            ]
        )
        samples, inputs = load_samples(eval_args)
        expected = identity(eval_args, samples, inputs)
        pairs = [
            (read_json(path / "manifest.json"), read_json(path / "results.json"))
            for path in parts
        ]
        merged, records, versions, model_name = collect_parts(expected, pairs)
        missing = {
            task: sum(
                not records.get((task, qid), {}).get("completed", False) for qid in ids
            )
            for task, ids in expected["ids"].items()
        }
        missing = {task: count for task, count in missing.items() if count}
        if missing and not args.allow_incomplete:
            raise ValueError(
                f"Some samples are still missing or failed: {missing}. Finish/resume both workers, or explicitly use --allow-incomplete"
            )
        with output_lock(output):
            path = output / "manifest.json"
            if path.is_file() and read_json(path)["identity"] != merged:
                raise ValueError("Existing merged output has a different run identity")
            if not path.is_file() and any(
                (output / name).exists()
                for name in ("results.json", "summary.json", "predictions.json")
            ):
                raise ValueError("Existing result files have no matching manifest")
            write_json(
                path,
                {
                    "identity": merged,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "model_name": model_name,
                    "versions": versions,
                    "selected_counts": {
                        task: len(rows) for task, rows in samples.items()
                    },
                    "merge_kind": "disjoint_full_task_partitions",
                    "source_parts": [str(path) for path in parts],
                },
            )
            save_results(eval_args, samples, records, show=True)
        print(f"Merged statistics: {output / 'summary.csv'}")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
