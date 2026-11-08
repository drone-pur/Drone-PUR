"""Unified model inference with preserved prompts, safe resume and weighted metrics."""

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
from importlib.util import find_spec
import json
from pathlib import Path
import sys
import time

from .backends import APIError, api_parameters, api_url, create_backend
from .data import (
    ANNOTATION_FILES,
    load_annotations,
    model_image_paths,
    read_json,
    sha256,
    write_json,
)
from .metrics import evaluate_samples
from .tasks import (
    build_prompt,
    parse_response,
    select_tasks,
    token_limit,
    uniform_subset,
)
from .utils import digest, model_identity, output_lock


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_samples(args):
    samples, inputs = {}, {}
    ids = read_json(args.sample_ids) if args.sample_ids else None
    if ids is not None and not isinstance(ids, (list, dict)):
        raise ValueError(
            "--sample-ids must contain a list of IDs or a task-to-ID-list mapping"
        )
    for task in select_tasks(args.tasks, args.mode):
        path = args.data_root / "annotations" / ANNOTATION_FILES[task]
        rows = load_annotations(args.data_root, task)
        qids = [r["question_id"] for r in rows]
        if len(qids) != len(set(qids)):
            raise ValueError(f"Duplicate IDs in {task}")
        if ids is not None:
            wanted = (
                ids.get(task, [])
                if isinstance(ids, dict)
                else [qid for qid in ids if qid in set(qids)]
            )
            if len(wanted) != len(set(wanted)) or set(wanted) - set(qids):
                raise ValueError(f"Duplicate or unknown selected IDs in {task}")
            rows = [r for r in rows if r["question_id"] in set(wanted)]
        else:
            rows = uniform_subset(rows, args.sample_every, args.max_samples)
        if not rows:
            raise ValueError(f"Empty selected subset for {task}")
        samples[task] = rows
        inputs[task] = sha256(path)
    if isinstance(ids, list):
        selected = {r["question_id"] for rows in samples.values() for r in rows}
        if set(ids) - selected:
            raise ValueError("--sample-ids contains IDs outside the selected tasks")
    return samples, inputs


def identity(args, samples, inputs):
    package = Path(__file__).parent
    training_spec = find_spec("drone_pur.training")
    if training_spec is None or training_spec.origin is None:
        raise ImportError(
            "Install this project with pip install -e . before evaluation"
        )
    source_roots = {"test": package, "train": Path(training_spec.origin).parent}
    model = (
        model_identity(args.model, args.revision)
        if args.backend != "api"
        else args.model
    )
    base = model_identity(args.base_model, args.revision) if args.base_model else None
    if isinstance(model, dict):
        args.adapter_revision = model.get("resolved_revision", args.revision)
        args.resolved_revision = (base or model).get("resolved_revision", args.revision)
    return {
        "mode": args.mode,
        "backend": args.backend,
        "model": model,
        "base_model": base,
        "revision": args.revision,
        "text_only": args.text_only,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "max_pixels": args.max_pixels,
        "api_endpoint_sha256": (
            digest(api_url(args.api_base_url)) if args.backend == "api" else None
        ),
        "api_params": api_parameters(args.api_params),
        "api_max_side": args.api_max_side,
        "chat_max_tiles": args.chat_max_tiles,
        "chat_image_size": args.chat_image_size,
        "max_new_tokens": {
            t: args.max_new_tokens or token_limit(t, args.mode) for t in samples
        },
        "annotations_sha256": inputs,
        "ids": {t: [r["question_id"] for r in rows] for t, rows in samples.items()},
        "code_sha256": {
            name + "/" + str(p.relative_to(root)): sha256(p)
            for name, root in source_roots.items()
            for p in sorted(root.rglob("*.py"))
            if name != "test" or not p.is_relative_to(source_roots["train"])
        },
    }


def report_summary(report):
    for task, result in report.get("tasks", report["subtasks"]).items():
        print(
            f"{task}: {result['n_completed']}/{result['n_target']}; valid={result['n_valid']}; "
            f"invalid={result['n_invalid']}; {result['completed_metrics']}",
            flush=True,
        )
    if "overall" in report:
        print(
            f"Overall (sample-weighted): {report['overall']['completed_metrics']}",
            flush=True,
        )


def run(args, backend_factory=create_backend):
    samples, inputs = load_samples(args)
    if args.output.resolve() == args.data_root.resolve() or any(
        args.output.resolve().is_relative_to(args.data_root.resolve() / part)
        for part in ("images", "annotations", "metadata")
    ):
        raise ValueError("Outputs must not overwrite benchmark data")
    prepared = {}
    for task, rows in samples.items():
        for row in rows:
            prompt = build_prompt(row, task, args.mode)
            paths = (
                []
                if args.text_only
                else model_image_paths(args.data_root, row, args.prepared_dir)
            )
            prepared[task, row["question_id"]] = (prompt, paths)
        print(
            f"{task}: {len(rows)} selected, prompts and ordered image paths checked",
            flush=True,
        )
    if args.check_only:
        return 0
    ident = identity(args, samples, inputs)
    # Include the exact ordered input files and rendered prompt, not just the annotation filename.
    ident["input_fingerprints"] = {
        task: {
            r["question_id"]: digest(
                {
                    "prompt": prepared[task, r["question_id"]][0],
                    "images": [
                        model_identity_image(p)
                        for p in prepared[task, r["question_id"]][1]
                    ],
                }
            )
            for r in rows
        }
        for task, rows in samples.items()
    }
    with output_lock(args.output):
        manifest_path, results_path = (
            args.output / "manifest.json",
            args.output / "results.json",
        )
        if manifest_path.is_file():
            if read_json(manifest_path)["identity"] != ident:
                raise ValueError(
                    "Run identity changed; use a new --output. Existing results will not be reused"
                )
        else:
            versions = {}
            for name in ("torch", "transformers", "peft", "Pillow", "requests"):
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    pass
            write_json(
                manifest_path,
                {
                    "identity": ident,
                    "created_at_utc": utc_now(),
                    "model_name": args.model_name or args.model,
                    "versions": versions,
                    "selected_counts": {t: len(r) for t, r in samples.items()},
                },
            )
        records = read_json(results_path) if results_path.is_file() else []
        by_id = {(r["task"], r["question_id"]): r for r in records}
        if len(by_id) != len(records):
            raise ValueError("Duplicate records in result cache")
        if args.retry_invalid:
            for key, record in by_id.items():
                if record.get("parse_status") == "invalid":
                    record["completed"] = False
        pending = [
            (task, row)
            for task, rows in samples.items()
            for row in rows
            if not by_id.get((task, row["question_id"]), {}).get("completed")
        ]
        print(
            f"Remaining: {len(pending)}/{sum(map(len, samples.values()))}", flush=True
        )
        backend = backend_factory(args) if pending else None
        errors = 0
        exit_code = 0
        try:
            for index, (task, row) in enumerate(pending, 1):
                key = task, row["question_id"]
                prompt, paths = prepared[key]
                started = time.monotonic()
                record = {
                    "task": task,
                    "question_id": row["question_id"],
                    "timestamp_utc": utc_now(),
                }
                try:
                    if hasattr(backend, "set_context"):
                        backend.set_context(task, args.mode)
                    text, info = backend.infer(
                        paths,
                        prompt,
                        args.max_new_tokens or token_limit(task, args.mode),
                    )
                    prediction = parse_response(text, task, args.mode)
                    from .metrics import score_prediction

                    valid = score_prediction(task, row, prediction, args.mode)["valid"]
                    record.update(
                        raw_response=text,
                        prediction=prediction,
                        completed=True,
                        parse_status="valid" if valid else "invalid",
                        inference_info=info,
                    )
                    errors = 0
                except Exception as exc:
                    # Upstream backends use different exception types. Keep
                    # failed inference resumable without counting it as an answer.
                    record.update(
                        completed=False,
                        error=f"{type(exc).__name__}: {exc}",
                        parse_status="not_parsed",
                    )
                    errors += 1
                record["elapsed_seconds"] = time.monotonic() - started
                by_id[key] = record
                print(
                    f"{index}/{len(pending)} {task}/{row['question_id']}: {record['parse_status']}"
                    + (f" {record['error']}" if "error" in record else ""),
                    flush=True,
                )
                if (
                    index % args.save_every == 0
                    or errors >= args.max_consecutive_errors
                ):
                    save_results(args, samples, by_id)
                if errors >= args.max_consecutive_errors:
                    print(
                        "Stopped after repeated inference errors. Fix the cause and rerun the same command to resume.",
                        file=sys.stderr,
                    )
                    exit_code = 2
                    break
        finally:
            save_results(args, samples, by_id, show=True)
        if any(
            not by_id.get((t, r["question_id"]), {}).get("completed")
            for t, rows in samples.items()
            for r in rows
        ):
            exit_code = 2
        return exit_code


def model_identity_image(path):
    value = Path(path)
    return {"path": str(value.resolve()), "sha256": sha256(value)}


def save_results(args, samples, by_id, show=False):
    records = [
        by_id[t, r["question_id"]]
        for t, rows in samples.items()
        for r in rows
        if (t, r["question_id"]) in by_id
    ]
    write_json(args.output / "results.json", records)
    predictions = [
        {"question_id": r["question_id"], "prediction": r["prediction"]}
        for r in records
        if r.get("completed")
    ]
    write_json(args.output / "predictions.json", predictions)
    report = evaluate_samples(samples, predictions, args.mode)
    write_json(args.output / "summary.json", report)
    flat = []
    for scope, results in (
        ("subtask", report["subtasks"]),
        ("task", report.get("tasks", {})),
        ("psj_attribute", report.get("psj_attributes", {})),
    ):
        for name, result in results.items():
            flat.append(
                {
                    "scope": scope,
                    "task": name,
                    "n_target": result["n_target"],
                    "n_completed": result["n_completed"],
                    "n_valid": result["n_valid"],
                    "n_invalid": result["n_invalid"],
                    "n_missing": result["n_missing"],
                    "complete": result["complete"],
                    **(result["completed_metrics"] or {}),
                    **{
                        "valid_" + k: v
                        for k, v in (result["valid_metrics"] or {}).items()
                    },
                }
            )
    if "overall" in report:
        result = report["overall"]
        flat.append(
            {
                "scope": "overall",
                "task": "all",
                "n_target": result["n_target"],
                "n_completed": result["n_completed"],
                "n_valid": result["n_valid"],
                "n_invalid": result["n_invalid"],
                "n_missing": result["n_missing"],
                "complete": result["complete"],
                **(result["completed_metrics"] or {}),
                **{"valid_" + k: v for k, v in (result["valid_metrics"] or {}).items()},
            }
        )
    columns = list(dict.fromkeys(key for row in flat for key in row))
    pending = args.output / "summary.csv.pending"
    with pending.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)
    pending.replace(args.output / "summary.csv")
    if show:
        report_summary(report)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--mode", choices=("choice", "open"), default="choice")
    p.add_argument("--tasks", nargs="+", default=["all"])
    p.add_argument(
        "--model",
        required=True,
        help="HF model ID, local full model/adapter directory, or API model ID",
    )
    p.add_argument(
        "--base-model",
        help="Required for LoRA adapters; optional saved merger modules are loaded automatically",
    )
    p.add_argument("--model-name")
    p.add_argument(
        "--backend",
        choices=("qwen", "api", "internvl", "minicpm4", "minicpm45", "llava-ov"),
        default="qwen",
    )
    p.add_argument(
        "--revision",
        default="main",
        help="Pin a HF commit for reproducible local evaluation",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepared-dir", type=Path)
    p.add_argument("--max-pixels", type=int)
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Keep indices 0,k,2k,... within each subtask",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        help="Uniformly cap each subtask after stride sampling",
    )
    p.add_argument("--sample-ids", type=Path)
    p.add_argument("--text-only", action="store_true")
    p.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    p.add_argument("--device-map", default="auto")
    p.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
    )
    p.add_argument("--chat-image-size", type=int, default=448)
    p.add_argument("--chat-max-tiles", type=int, default=6)
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Required for InternVL, MiniCPM and remote-code LLaVA; review upstream code first",
    )
    p.add_argument(
        "--api-base-url",
        help="OpenAI-compatible /v1 base URL or complete chat/completions URL",
    )
    p.add_argument("--api-key-env", default="DRONE_PUR_API_KEY")
    p.add_argument(
        "--api-params",
        help='Provider-specific JSON object, e.g. {"enable_thinking":false}',
    )
    p.add_argument("--api-timeout", type=float, default=180)
    p.add_argument("--api-retries", type=int, default=2)
    p.add_argument(
        "--api-max-side",
        type=int,
        help="Optional explicit API resize; unset sends original image bytes",
    )
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--max-consecutive-errors", type=int, default=3)
    p.add_argument(
        "--retry-invalid",
        action="store_true",
        help="Explicitly retry completed unparseable answers; can incur API costs",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Check data and prompts without loading a model or calling an API",
    )
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.backend == "api" and not args.api_base_url:
        p.error("--backend api requires --api-base-url")
    for value in (
        args.sample_every,
        args.save_every,
        args.max_consecutive_errors,
        args.chat_image_size,
        args.chat_max_tiles,
        args.api_timeout,
    ):
        if value <= 0:
            p.error(
                "Sampling, save intervals, image size, tile count and timeout must be positive"
            )
    if args.api_retries < 0 or any(
        v is not None and v <= 0
        for v in (
            args.max_samples,
            args.max_pixels,
            args.max_new_tokens,
            args.api_max_side,
        )
    ):
        p.error("Optional limits must be positive; retries must be nonnegative")
    if args.backend != "qwen" and (args.base_model or args.max_pixels):
        p.error(
            "--base-model and --max-pixels currently apply only to the Qwen backend"
        )
    try:
        return run(args)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        p.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
