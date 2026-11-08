# Scripts

Run these scripts from the repository root after installing the package.

| Script | Purpose |
| --- | --- |
| `train_uav_pur.sh` | Launch all three UAV-PUR training stages; defaults to eight GPUs |
| `evaluate_choice.sh` | Forward arguments to the unified multiple-choice evaluator |
| `evaluate_open.sh` | Forward arguments to the unified open-ended evaluator |
| `merge_choice_parts.py` | Validate disjoint local-Qwen choice partitions and recompute their combined metrics |
| `audit_release.py` | Check source files for credentials, host-specific paths and bundled artifacts |
| `github-actions.yml` | Optional GitHub Actions workflow template |
| `checks/` | Regression tests for prompts, metrics, preparation, training, API transport and packaging |

See the [training guide](../train/README.md) and [evaluation guide](../test/README.md) for complete commands.

## Parallel Choice Evaluation

Use separate terminals and output directories to evaluate disjoint task groups with the same local model. Each process loads one model copy on its visible GPU. Prepare the reference-box cache once before starting either process.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_choice.sh --data-root /path/to/Drone-PUR-Benchmark --prepared-dir /path/to/prepared-images --model /path/to/UAV-PUR-8B --tasks CJ PSJ CR MMR --max-pixels 1003520 --device-map cuda:0 --output test/outputs/uav-pur-8b/part1
```

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/evaluate_choice.sh --data-root /path/to/Drone-PUR-Benchmark --prepared-dir /path/to/prepared-images --model /path/to/UAV-PUR-8B --tasks OC OR MIG VG SI MVR --max-pixels 1003520 --device-map cuda:0 --output test/outputs/uav-pur-8b/part2
```

After both finish, merge the full ten-task results:

```bash
python scripts/merge_choice_parts.py --model /path/to/UAV-PUR-8B --data-root /path/to/Drone-PUR-Benchmark --parts test/outputs/uav-pur-8b/part1 test/outputs/uav-pur-8b/part2 --output test/outputs/uav-pur-8b/merged --max-pixels 1003520 --device-map cuda:0
```

Match inference overrides between workers and the merger, including image area, dtype, attention implementation, device map, output length and text-only mode. The merger checks full sample coverage, disjoint task groups, model/code/data identities and runtime versions, then re-scores individual predictions. MMR/MVR and overall accuracy are sample-weighted; partition accuracies are not averaged. `--allow-incomplete` exports progress without marking missing answers incorrect or reporting partial metrics as final.

## Standalone Model Export

Export a completed adapter together with its trained visual mergers as a standalone Transformers model:

```bash
python -m drone_pur.training.export --base-model /path/to/Qwen3-VL-8B-Instruct --adapter /path/to/checkpoint --output /path/to/UAV-PUR-8B --repo-id USER/UAV-PUR-8B --base-id Qwen/Qwen3-VL-8B-Instruct --model-card /path/to/model-card.md --license-file /path/to/LICENSE --notice-file /path/to/NOTICE --report /path/to/internal/export-report.json --verify-image /path/to/image1.jpg --verify-image /path/to/image2.jpg --verify-device cuda:0 --require-complete
```

The exporter checks all adapter tensors, verifies every merged LoRA matrix against its expected weight update, preserves the main and three DeepStack mergers, and reloads the sharded model without an adapter. Image/text smoke checks use a 65,536-pixel verification cap without changing the saved processor settings. BF16 rounding diagnostics record relative logit RMSE, next-token KL divergence and greedy-token agreement; serialization must reproduce the merged model's logits and greedy tokens exactly. The internal report is stored outside the upload folder. Use a new output directory for each export.

Verify the exported files against that report before upload:

```bash
python -m drone_pur.training.export --check-release /path/to/UAV-PUR-8B --report /path/to/internal/export-report.json
```

Run regression tests and the code-release audit:

```bash
python -m unittest discover -s scripts/checks -v
python scripts/audit_release.py
```

Tensor tests require the training dependencies. API transport tests use mock responses and do not send paid requests. To enable GitHub Actions, place the workflow template under `.github/workflows/`.
