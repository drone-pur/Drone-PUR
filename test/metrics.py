"""Paper-aligned choice, grounding, counting and set-detection metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from .data import (
    ANNOTATION_FILES,
    OPEN_TASKS,
    load_annotations,
    parse_box,
    read_json,
    sha256,
    write_json,
)


PROTOCOL = {
    "reference": "https://arxiv.org/abs/2610.40801",
    "choice": "Exact option matching; CJ uses Yes/No (case-insensitive).",
    "grounding": "IoU > 0.5; absent targets require not exist.",
    "mra": "Mean of abs(pred-gt)/gt < t for t = 0.05, 0.10, ..., 0.50.",
    "zero_count_extension": "For GT=0, MRA=1 only when prediction=0.",
    "od": "Maximum-cardinality one-to-one matching with IoU > 0.5; pool TP/FP/FN.",
    "invalid": "Completed invalid outputs score zero accuracy/MRA; OD uses empty predictions.",
    "mae": "Full MAE requires all counts valid; otherwise only mae_valid is defined.",
    "missing": "Missing IDs are not errors; final metrics are withheld until all IDs complete.",
    "units": "Accuracy, MRA, precision, recall and F1 are percentages; MAE is in objects.",
}


def valid_count(value):
    return (
        type(value) is int
        and value >= 0
        or type(value) is float
        and math.isfinite(value)
        and value >= 0
        and value.is_integer()
    )


def count_metrics(prediction, target):
    if not valid_count(target):
        raise ValueError(f"Invalid reference count: {target!r}")
    if not valid_count(prediction):
        return {"valid": False, "mra": 0.0, "abs_error": None}
    error = abs(int(prediction) - int(target))
    # Integer cross multiplication preserves the paper's strict boundaries.
    mra = (
        float(error == 0)
        if target == 0
        else sum(100 * error < percent * int(target) for percent in range(5, 51, 5))
        / 10
    )
    return {"valid": True, "mra": mra, "abs_error": error}


def prediction_box(value):
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(type(x) not in (int, float) for x in value)
    ):
        return None
    try:
        return parse_box(value)
    except (ValueError, TypeError, OverflowError):
        return None


def box_iou(first, second):
    area_a = (first[2] - first[0]) * (first[3] - first[1])
    area_b = (second[2] - second[0]) * (second[3] - second[1])
    inter = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    return inter / (area_a + area_b - inter)


def match_boxes(predictions, targets):
    """Maximize valid match count, not greedy IoU or summed IoU."""
    if not predictions or not targets:
        tp = 0
    else:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import maximum_bipartite_matching

        edges = [
            (i, j)
            for i, box in enumerate(predictions)
            for j, target in enumerate(targets)
            if box_iou(box, target) > 0.5
        ]
        row_indices = [i for i, _ in edges]
        col_indices = [j for _, j in edges]
        graph = csr_matrix(
            ([1] * len(edges), (row_indices, col_indices)),
            shape=(len(predictions), len(targets)),
        )
        tp = int((maximum_bipartite_matching(graph, perm_type="column") >= 0).sum())
    return {"tp": tp, "fp": len(predictions) - tp, "fn": len(targets) - tp}


def score_prediction(task, sample, prediction, mode):
    if mode == "choice":
        answer = prediction.strip().upper() if isinstance(prediction, str) else None
        candidates = {"YES", "NO"} if task == "CJ" else set(sample["options"])
        valid = answer in candidates
        return {
            "valid": valid,
            "correct": valid and answer == sample["correct_answer"].strip().upper(),
        }

    target = sample["ground_truth"]
    if target["type"] == "count":
        return count_metrics(prediction, target["count"])
    if target["type"] == "boxes":
        boxes = (
            [prediction_box(value) for value in prediction]
            if isinstance(prediction, list)
            else None
        )
        valid = boxes is not None and all(box is not None for box in boxes)
        targets = [parse_box(value) for value in target["boxes"]]
        return {"valid": valid, **match_boxes(boxes if valid else [], targets)}
    if target["type"] in {"bbox", "absent"}:
        absent = (
            isinstance(prediction, str) and prediction.strip().lower() == "not exist"
        )
        box = prediction_box(prediction)
        if target["type"] == "absent":
            return {"valid": absent or box is not None, "correct": absent, "iou": None}
        reference = parse_box(target["box"])
        overlap = box_iou(box, reference) if box is not None else None
        return {
            "valid": absent or box is not None,
            "correct": overlap is not None and overlap > 0.5,
            "iou": overlap,
        }
    raise ValueError(f"Unsupported open target: {task} / {target['type']}")


def aggregate(completed, task, mode):
    valid = [row for row in completed if row["valid"]]
    n = len(completed)
    if not n:
        metrics = None
    elif mode == "open" and task == "OD":
        tp, fp, fn = (sum(row[key] for row in completed) for key in ("tp", "fp", "fn"))
        metrics = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision_percent": 100 * tp / (tp + fp) if tp + fp else 0.0,
            "recall_percent": 100 * tp / (tp + fn) if tp + fn else 0.0,
            "micro_f1_percent": (
                200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            ),
        }
    elif mode == "open" and task in {"OC", "MMR-OC"}:
        mae = sum(row["abs_error"] for row in valid) / len(valid) if valid else None
        metrics = {
            "mra_percent": 100 * sum(row["mra"] for row in completed) / n,
            "mae": mae if len(valid) == n else None,
            "mae_valid": mae,
        }
    else:
        correct = sum(row["correct"] for row in completed)
        metrics = {"correct": correct, "accuracy_percent": 100 * correct / n}
    return metrics


def summarize(rows, task, mode):
    completed = [row for row in rows if row["completed"]]
    valid = [row for row in completed if row["valid"]]
    n = len(completed)
    metrics = aggregate(completed, task, mode)
    complete = n == len(rows)
    return {
        "n_target": len(rows),
        "n_completed": n,
        "n_missing": len(rows) - n,
        "n_valid": len(valid),
        "n_invalid": n - len(valid),
        "complete": complete,
        "metrics": metrics if complete else None,
        "completed_metrics": metrics,
        "valid_metrics": aggregate(valid, task, mode),
    }


def select_tasks(requested, mode):
    allowed = [
        task
        for task in ANNOTATION_FILES
        if (task != "OD" if mode == "choice" else task in OPEN_TASKS)
    ]
    if requested == ["all"]:
        return allowed
    tasks = []
    for name in requested:
        expanded = (
            [task for task in allowed if task.startswith(name + "-")]
            if name in {"MMR", "MVR"}
            else [name]
        )
        if not expanded or any(task not in allowed for task in expanded):
            raise ValueError(
                f"Unsupported {mode} task: {name}; choose from {', '.join(allowed)}"
            )
        for task in expanded:
            if task not in tasks:
                tasks.append(task)
    return tasks


def evaluate(root, tasks, predictions, mode):
    return evaluate_samples(
        {task: load_annotations(root, task) for task in tasks}, predictions, mode
    )


def evaluate_samples(samples, predictions, mode):
    if not isinstance(predictions, list):
        raise ValueError(
            "Predictions must be a JSON list of question_id/prediction objects"
        )
    target_ids = [sample["question_id"] for rows in samples.values() for sample in rows]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("Duplicate annotation question IDs across selected tasks")
    by_id = {}
    for item in predictions:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("question_id"), str)
            or "prediction" not in item
        ):
            raise ValueError(
                "Every completed prediction requires string question_id and prediction (null for invalid)"
            )
        qid = item["question_id"]
        if qid in by_id:
            raise ValueError(f"Duplicate prediction ID: {qid}")
        by_id[qid] = item["prediction"]
    unknown = set(by_id) - set(target_ids)
    if unknown:
        raise ValueError(
            f"Prediction IDs outside selected tasks: {', '.join(sorted(unknown)[:10])}"
        )

    report = {"mode": mode, "protocol": PROTOCOL, "subtasks": {}, "records": []}
    grouped = defaultdict(list)
    for task, items in samples.items():
        records = []
        for sample in items:
            qid = sample["question_id"]
            row = {"task": task, "question_id": qid, "completed": qid in by_id}
            if row["completed"]:
                row.update(score_prediction(task, sample, by_id[qid], mode))
            records.append(row)
        report["subtasks"][task] = summarize(records, task, mode)
        report["records"].extend(records)
        grouped[task.split("-")[0]].extend(records)
    if mode == "choice":
        report["tasks"] = {
            task: summarize(rows, task, mode) for task, rows in grouped.items()
        }
        report["overall"] = summarize(report["records"], "all", mode)
        if "PSJ" in samples:
            attributes = {
                row["question_id"]: row.get("sampled_question_type", "Unspecified")
                for row in samples["PSJ"]
            }
            grouped_attributes = defaultdict(list)
            for row in report["records"]:
                if row["task"] == "PSJ":
                    grouped_attributes[attributes[row["question_id"]]].append(row)
            report["psj_attributes"] = {
                name: summarize(rows, "PSJ", mode)
                for name, rows in grouped_attributes.items()
            }
    report["missing_ids"] = [
        row["question_id"] for row in report["records"] if not row["completed"]
    ]
    report["invalid_ids"] = [
        row["question_id"]
        for row in report["records"]
        if row["completed"] and not row["valid"]
    ]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--mode", choices=("choice", "open"), required=True)
    parser.add_argument(
        "--tasks", nargs="+", default=["all"], help="Leaf identifiers, MMR/MVR, or all"
    )
    parser.add_argument(
        "--output", type=Path, help="Save full JSON report; otherwise print it"
    )
    args = parser.parse_args()
    try:
        tasks = select_tasks(args.tasks, args.mode)
        if args.output:
            output = args.output.resolve()
            if output == args.predictions.resolve() or any(
                output.is_relative_to(args.data_root.resolve() / folder)
                for folder in ("annotations", "images", "metadata")
            ):
                raise ValueError(
                    "Output must not overwrite predictions or benchmark data"
                )
        report = evaluate(args.data_root, tasks, read_json(args.predictions), args.mode)
        report["input_checksums"] = {
            "predictions": sha256(args.predictions),
            "annotations": {
                task: sha256(args.data_root / "annotations" / ANNOTATION_FILES[task])
                for task in tasks
            },
        }
    except (ValueError, OSError, ImportError, KeyError) as exc:
        parser.error(str(exc))
    if args.output:
        write_json(args.output, report)
        for task, result in report["subtasks"].items():
            print(
                f"{task}: {result['n_completed']}/{result['n_target']} completed; "
                f"invalid={result['n_invalid']}; metrics={result['metrics']}"
            )
        if "overall" in report:
            print(f"Overall (sample-weighted): {report['overall']['metrics']}")
        print(f"Saved: {args.output}")
    else:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
