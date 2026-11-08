# UAV-PUR Training

The training pipeline supports Qwen3-VL-4B and Qwen3-VL-8B through three sequential stages. Configuration files are in `train/configs/`; commands below run from the repository root.

## Architecture and Objectives

The vision encoder and original language-model weights are frozen. Trainable components are language-layer LoRA, the main visual merger, all three DeepStack mergers, and the active auxiliary heads. LoRA targets the attention `q/k/v/o` and MLP `gate/up/down` projections, with rank 32, alpha 64 and dropout 0.05. The original LM head supplies answer-token CE; its weights remain frozen.

Auxiliary heads use prompt hidden states, without access to answer-token states:

| Head | Structure | Auxiliary loss |
| --- | --- | --- |
| Single box | LayerNorm, Linear(512), GELU, Linear(4), ordered normalized coordinates | `0.1 * (mean L1 + 1 - GIoU)` |
| Count | LayerNorm, Linear(512), GELU, Linear(1), Softplus | `0.01 * MAE` |
| Detection | 256-wide projected prompt memory, 128 learned queries, two eight-head Transformer decoder layers, box/objectness branches | `0.1 * (5 * L1 + 2 * (1 - GIoU) + BCE)` with Hungarian matching |

Detection L1 sums the four coordinate errors for each matched box. Unmatched detection queries have BCE weight 0.1. CE averages supervised answer tokens per sample, then averages over samples; prompt tokens are masked. Auxiliary losses apply to their corresponding open-ended records. MVR-VG answers indicating an absent target use CE only.

## Stages

| Stage | Epochs | Training data | Auxiliary heads |
| --- | --- | --- | --- |
| S1 | 1 | Perception tasks: OC, MIG, OR and a small OD pool | Box, count, detection |
| S2 | 1 | All tasks, including six grounding/counting open-ended settings and the small OD pool | Box, count, detection |
| S3 | 1 | Ten choice tasks and six grounding/counting open-ended settings | None; token CE only |

`train/configs/uav_pur.json` defines task sampling, learning rates, loss weights and save intervals. Sampling is deterministic under the configured seed. The default OD budget is 2,500 records, with the same selected OD pool used in S1 and S2. S3 excludes OD. All supplied training data is available for training; no validation split is created.

## Data Preparation

Install the project using the root [installation instructions](../README.md#installation). Training annotations and images are separate from the evaluation benchmark.

Edit `train/configs/train_sources.example.json` to specify the training files for each task/subtask. The preparer accepts the public schema (`images`, `template.box`, typed `ground_truth`) and task-specific source annotations. Relative annotation and image paths resolve under `--data-root`.

Validate inputs, then write the stage datasets:

```bash
uav-pur-prepare --data-root /path/to/Drone-PUR-Train --sources train/configs/train_sources.example.json --config train/configs/uav_pur.json --output train/data/stages --dry-run
uav-pur-prepare --data-root /path/to/Drone-PUR-Train --sources train/configs/train_sources.example.json --config train/configs/uav_pur.json --output train/data/stages
```

Use clean reference images for MIG and MVR-VG. The preparer creates red-box derivatives in the prepared output. Set `--template-input prepared` if the supplied reference images already contain their template boxes. `--choice-jsonl FILE` optionally imports an existing prepared choice corpus.

OD selection uses source/count strata and excludes annotations exceeding the 128-query capacity. `--benchmark-root PATH` additionally excludes OD records sharing resolved image paths with the benchmark.

Each stage contains `train.jsonl` and `manifest.json`, with the configuration, record counts and source hashes. Use a new empty output directory when preparing a different configuration.

## Default Parameters

| Parameter | Default |
| --- | --- |
| GPUs / per-GPU microbatch / accumulation | 8 / 1 / 4 |
| Global batch | 32 |
| Epochs | 1 per stage |
| LoRA learning rate | S1/S2: `2e-5`; S3: `1e-5` |
| Merger learning rate | 0.1 times the stage learning rate |
| Auxiliary-head learning rate | `1e-4` |
| Optimizer / weight decay | AdamW / 0.01 on non-bias matrices |
| Schedule | 5% warmup, cosine decay within each stage |
| Gradient clipping | 1.0 |
| Precision / checkpointing | BF16 / non-reentrant gradient checkpointing |
| Maximum image area / sequence length | 1,003,520 pixels per image / 8,192 tokens |
| Checkpoint interval | S1/S2: 1,000 steps; S3: 500 steps |

## Launch and Resume

Train all three stages on eight GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_uav_pur.sh /path/to/Qwen3-VL-4B-Instruct train/data/stages train/runs/uav-pur-4b
```

For 8B, supply the corresponding base model and a separate run directory. `NPROC_PER_NODE`, `GRAD_ACC`, `MAX_PIXELS` and `MAX_LENGTH` are environment-variable overrides. GPU memory requirements depend on model size, image area and sequence length.

S2 loads S1 weights; S3 loads S2 weights. Each stage initializes its own optimizer and schedule. Stage outputs include periodic checkpoints, an epoch checkpoint and `final/`. Checkpoints marked `COMPLETE` contain adapter/merger weights, active auxiliary heads and resumable training state.

Resume a stage from a complete checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --standalone --nproc_per_node=8 -m drone_pur.training.train_stage --stage s3 --model /path/to/Qwen3-VL-4B-Instruct --data-dir train/data/stages/s3 --output-dir train/runs/uav-pur-4b/s3 --resume train/runs/uav-pur-4b/s3/checkpoint-4000 --grad-acc 4 --save-steps 200
```

Resume restores optimizer/scheduler state, per-rank RNG and the data cursor. Keep the code, configuration, source data, base model, GPU count and accumulation unchanged for exact resume. The save interval may be changed independently. Checkpoints include optimizer state; reserve disk space for the desired retention period.

Run an input/token preflight without allocating model weights:

```bash
uav-pur-train --stage s1 --model /path/to/Qwen3-VL-4B-Instruct --data-dir train/data/stages/s1 --output-dir train/runs/check --preflight --token-check 2
```

`uav-pur-train` is the worker entry point. Use the shell launcher or `torch.distributed.run` for multi-GPU training. Prepared manifests retain the configuration used to create them; regenerate stage data after changing sampling or epoch settings.
