#!/usr/bin/env python3
"""Single-stage DDP training; deterministic exact sample passes, batch=1 per rank.

Stage transitions load weights only; --resume restores optimizer, scheduler,
per-rank RNG and exact next data cursor. No pickle loading is used.
"""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoProcessor

from .checkpoint_io import load_tree, save_tree
from ..data import sha256 as sha
from .staged_data import Collator
from .staged_model import build_model, optimizer_groups


def window_assignments(order, offset, world, accumulation):
    """No sampler duplication: padded rank slots have zero weight, not extra OD exposure."""
    window = order[offset : offset + world * accumulation]
    return [
        [window[j + rank] if j + rank < len(window) else None for rank in range(world)]
        for j in range(0, len(window), world)
    ], len(window)


def lr_factor(completed, total, warmup):
    if completed < warmup:
        return (completed + 1) / max(1, warmup)
    return 0.5 * (
        1 + math.cos(math.pi * min(1, (completed - warmup) / max(1, total - warmup)))
    )


def validate_manifest(folder, stage):
    m = json.loads((folder / "manifest.json").read_text())
    if m["stage"] != stage or sha(folder / "train.jsonl") != m["output_sha256"]:
        raise ValueError("Stage/data SHA256 mismatch")
    for path, digest in m["input_sha256"].items():
        if not Path(path).is_file() or sha(path) != digest:
            raise ValueError(
                f"Source missing/changed: {path}; build a new dataset version"
            )
    return m


def save_checkpoint(
    folder, model, processor, optimizer, scheduler, cursor, identity, rank, world
):
    rng = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
        "python": random.getstate(),
    }
    gathered = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(rng, gathered, dst=0)
    else:
        gathered = [rng]
    if rank == 0:
        pending = folder.with_name(folder.name + ".incomplete")
        if folder.exists() or pending.exists():
            raise FileExistsError(
                f"Checkpoint already exists (or partial save): {folder}"
            )
        pending.mkdir()
        model.save_weights(pending)
        (pending / "base_config.json").write_text(
            json.dumps(model.backbone.get_base_model().config.to_dict(), indent=2)
        )
        processor.save_pretrained(pending)
        save_tree(
            pending / "training_state",
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_by_rank": gathered,
            },
        )
        (pending / "training_cursor.json").write_text(
            json.dumps({"identity": identity, **cursor}, indent=2)
        )
        # Cursor is only consumed after the whole directory has been committed.
        (pending / "COMPLETE").write_text("complete\n")
        pending.rename(folder)
        print(f"Saved {folder}", flush=True)
    if world > 1:
        dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["s1", "s2", "s3"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--grad-acc", type=int, default=4)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Stop after this completed epoch without shortening the planned LR schedule",
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--token-check",
        type=int,
        default=0,
        help="Encode this many examples per task/view; -1 means ALL. CPU, no model loaded.",
    )
    args = parser.parse_args()
    base_path = Path(args.model).expanduser().resolve()
    if not (base_path / "config.json").is_file():
        parser.error("--model must point to a downloaded local Qwen3-VL base directory")
    args.model = str(base_path)
    if args.save_steps is not None and args.save_steps < 1:
        parser.error("--save-steps must be positive")
    if args.init_checkpoint and args.resume:
        parser.error("--init-checkpoint and --resume are mutually exclusive")
    if (
        args.grad_acc < 1
        or args.logging_steps < 1
        or args.max_length < 1
        or args.max_pixels < 1024
    ):
        parser.error("Invalid positive runtime settings")
    manifest = validate_manifest(args.data_dir, args.stage)
    config = manifest["config"]
    stage = config["stages"][args.stage]
    save_steps = args.save_steps if args.save_steps is not None else stage["save_steps"]
    if args.stage == "s3" and (stage["aux"] or not stage["open_rates"]):
        raise ValueError(
            "Final UAV-PUR S3 requires mixed choice/open data and no auxiliary heads"
        )
    if (
        args.stop_after_epoch is not None
        and not 1 <= args.stop_after_epoch <= stage["epochs"]
    ):
        parser.error("--stop-after-epoch must be between 1 and the planned epoch count")
    if not config.get("train_mergers") or config.get("expected_deepstack_mergers") != 3:
        raise ValueError(
            "Final UAV-PUR requires trainable main and three DeepStack mergers"
        )
    if not 0 < config.get("merger_lr_ratio", 0) <= 1:
        raise ValueError("Invalid merger_lr_ratio")
    rows = [json.loads(l) for l in (args.data_dir / "train.jsonl").open()]
    if len(rows) != manifest["audit"]["records_per_epoch"]:
        raise ValueError("Manifest count mismatch")
    if args.preflight:
        missing = sorted(
            {p for r in rows for p in r["images"] if not Path(p).is_file()}
        )
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing images: {missing[:10]}")
        print(json.dumps(manifest["audit"], ensure_ascii=False, indent=2))
        if args.token_check:
            processor = AutoProcessor.from_pretrained(
                args.model, max_pixels=args.max_pixels
            )
            collate = Collator(processor, args.max_length, args.max_pixels)
            counts = {}
            longest = 0
            for row in rows:
                key = row["task"] + "/" + row["view"]
                if args.token_check > 0 and counts.get(key, 0) >= args.token_check:
                    continue
                batch = collate([row])
                counts[key] = counts.get(key, 0) + 1
                longest = max(longest, batch["input_ids"].shape[1])
            print("Token checks:", counts, "longest=", longest)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA; use --preflight to audit without training")
    rank, local, world = (
        int(os.environ.get(k, default))
        for k, default in [("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1)]
    )
    torch.cuda.set_device(local)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported on this GPU")
    if world > 1:
        dist.init_process_group("nccl")
    output = args.output_dir.resolve()
    if output == args.data_dir.resolve() or output.is_relative_to(
        args.data_dir.resolve()
    ):
        raise ValueError("Training output must not be inside prepared data")
    if (output / "final").exists():
        raise ValueError("Stage already completed; use a new output directory")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError(
            "Output is not empty; resume explicitly or use a new directory"
        )
    if args.stage != "s1" and not (args.init_checkpoint or args.resume):
        raise ValueError("Later stages require the previous stage checkpoint")
    source = args.resume or args.init_checkpoint
    if source:
        if not (source / "COMPLETE").is_file():
            raise ValueError("Checkpoint is incomplete or not from this staged trainer")
        cursor = json.loads((source / "training_cursor.json").read_text())
        expected = (
            args.stage if args.resume else {"s2": "s1", "s3": "s2"}.get(args.stage)
        )
        if cursor["identity"]["stage"] != expected:
            raise ValueError(f"Wrong source stage, expected {expected}")
        if not args.resume and not cursor["stage_finished"]:
            raise ValueError("Next stage requires a completed previous stage")
    random.seed(config["seed"] + rank)
    torch.manual_seed(
        config["seed"]
    )  # identical initialization of auxiliary heads across ranks
    model = build_model(args.model, config, args.stage, source).cuda(local)
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    collate = Collator(processor, args.max_length, args.max_pixels)
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = optimizer_groups(model, config, stage)
    optimizer = torch.optim.AdamW(groups)
    per_epoch = math.ceil(len(rows) / (world * args.grad_acc))
    total = per_epoch * stage["epochs"]
    warmup = math.ceil(total * 0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, total, warmup)
    )
    identity = {
        "stage": args.stage,
        "data_sha256": manifest["output_sha256"],
        "config": config,
        "model": str(Path(args.model).resolve()),
        "world_size": world,
        "grad_acc": args.grad_acc,
        "batch_size": 1,
        "max_pixels": args.max_pixels,
        "max_length": args.max_length,
        "total_steps": total,
        "trainable_names": [n for n, p in parameters],
        "code_sha256": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
    }
    identity["base_config_sha256"] = sha(Path(args.model) / "config.json")
    identity["base_weight_files"] = {
        p.name: [p.stat().st_size, p.stat().st_mtime_ns]
        for p in sorted(Path(args.model).glob("*.safetensors"))
    }
    identity["package_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "safetensors", "scipy")
    }
    epoch_start, offset_start, global_step = 0, 0, 0
    resume_rng = None
    if args.resume:
        saved_identity = dict(cursor["identity"])
        if saved_identity != identity:
            raise ValueError(
                "Exact resume requires identical data/config/code/model/GPU count/batch"
            )
        if cursor["stage_finished"]:
            raise ValueError("Resume stage is already finished")
        state = load_tree(args.resume / "training_state")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        resume_rng = state["rng_by_rank"][rank]
        epoch_start, offset_start, global_step = (
            cursor["next_epoch"],
            cursor["next_offset"],
            cursor["global_step"],
        )
        del state
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_manifest.json").write_text(
            json.dumps(
                {
                    "identity": identity,
                    "init_checkpoint": str(source) if source else None,
                    "trainable_parameters": {n: list(p.shape) for n, p in parameters},
                    "trainable_count": sum(p.numel() for _, p in parameters),
                    "optimizer_groups": [
                        {
                            "name": g["group_name"],
                            "peak_lr": g["initial_lr"],
                            "numel": sum(p.numel() for p in g["params"]),
                        }
                        for g in optimizer.param_groups
                    ],
                    "sampling": "Exact once per prepared record each epoch; final distributed slots carry zero weight.",
                },
                indent=2,
            )
        )
        print(
            f'{args.stage}: {len(rows)} rows x {stage["epochs"]} epochs, {per_epoch} optimizer steps/epoch, {total} total',
            flush=True,
        )
    if world > 1:
        dist.barrier()
        # Different ranks/microbatches can use different heads; non-reentrant checkpointing.
        train_model = DDP(
            model, device_ids=[local], find_unused_parameters=stage["aux"]
        )
    else:
        train_model = model
    torch.manual_seed(config["seed"] + rank)
    if resume_rng:
        torch.set_rng_state(resume_rng["torch"])
        torch.cuda.set_rng_state(resume_rng["cuda"])
        random.setstate(resume_rng["python"])
    train_model.train()
    dummy = next(r for r in rows if r["view"] == "choice")
    metric_sum = torch.zeros(10, device=local)
    started = time.monotonic()
    for epoch in range(epoch_start, args.stop_after_epoch or stage["epochs"]):
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        order = torch.randperm(len(rows), generator=generator).tolist()
        for offset in range(
            offset_start if epoch == epoch_start else 0,
            len(rows),
            world * args.grad_acc,
        ):
            assignments, n = window_assignments(order, offset, world, args.grad_acc)
            optimizer.zero_grad(set_to_none=True)
            for slots in assignments:
                index = slots[rank]
                row = rows[index] if index is not None else dummy
                batch = collate([row])
                batch = {
                    k: (
                        v.cuda(local, non_blocking=True)
                        if isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in batch.items()
                }
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = train_model(**batch)
                # DDP averages rank gradients; scale by world/n for exact sample mean.
                loss = result["loss"] * (world / n if index is not None else 0)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at {row["question_id"]}')
                loss.backward()
                if index is not None:
                    metric_sum[:9] += result["metrics"]
                    metric_sum[9] += result["loss"].detach()
            norm = torch.nn.utils.clip_grad_norm_(
                [p for _, p in parameters], 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_end = offset + world * args.grad_acc >= len(rows)
            if global_step % args.logging_steps == 0 or epoch_end:
                if world > 1:
                    dist.all_reduce(metric_sum)
                values = metric_sum.cpu().tolist()
                log = {
                    "stage": args.stage,
                    "step": global_step,
                    "epoch": epoch
                    + min(1, (offset + world * args.grad_acc) / len(rows)),
                    "loss": values[9] / max(1, values[1]),
                    "ce": values[0] / max(1, values[1]),
                    "box_l1": values[2] / max(1, values[4]),
                    "box_giou_loss": values[3] / max(1, values[4]),
                    "box_examples": values[4],
                    "count_mae": values[5] / max(1, values[6]),
                    "count_examples": values[6],
                    "od_set_loss": values[7] / max(1, values[8]),
                    "od_examples": values[8],
                    "next_lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": float(norm),
                    "next_lrs": {
                        g["group_name"]: g["lr"] for g in optimizer.param_groups
                    },
                    "elapsed_seconds": time.monotonic() - started,
                }
                if rank == 0:
                    print(json.dumps(log), flush=True)
                    with (output / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps(log) + "\n")
                metric_sum.zero_()
            cursor = {
                "global_step": global_step,
                "next_epoch": epoch + 1 if epoch_end else epoch,
                "next_offset": 0 if epoch_end else offset + world * args.grad_acc,
                "stage_finished": epoch_end and epoch + 1 == stage["epochs"],
            }
            if global_step % save_steps == 0 or epoch_end:
                save_checkpoint(
                    output / f"checkpoint-{global_step}",
                    model,
                    processor,
                    optimizer,
                    scheduler,
                    cursor,
                    identity,
                    rank,
                    world,
                )
    if cursor["stage_finished"]:
        save_checkpoint(
            output / "final",
            model,
            processor,
            optimizer,
            scheduler,
            cursor,
            identity,
            rank,
            world,
        )
    elif rank == 0:
        print(
            "Stopped at an epoch checkpoint with the original LR schedule preserved.",
            flush=True,
        )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
