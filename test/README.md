# Evaluation

The unified evaluator supports multiple-choice and open-ended prediction on the [Drone-PUR Benchmark](https://huggingface.co/datasets/Henard224/Drone-PUR-Benchmark). Use the benchmark's full task annotations and relative image paths.

## Data and Inputs

Keep `annotations/`, `images/` and `metadata/` under the benchmark root. Annotation files contain `results`; records include `question_id`, ordered `images`, `question`, options where applicable, and task-specific targets.

Prepare reference-box images before evaluating MIG or MVR-VG:

```bash
drone-pur-templates --data-root /path/to/Drone-PUR-Benchmark
```

Image order is task-specific:

- MIG: four frames in temporal order, with a reference box only on frame 1.
- MVR-VG: the reference box is on view 1; predict its target in view 2.
- MVR reasoning: view 1 followed by view 2.
- MMR: RGB followed by TIR; grounding coordinates refer to RGB.

Original images remain unchanged. `--prepared-dir` selects the template-image cache. Task prompt wording, examples and option order are defined in `test/prompts/`. Open prompts omit options. Reference answers and auxiliary annotations are used for scoring, not added to model inputs.

## Models

The root `requirements.txt` provides the reference Qwen3-VL environment. Qwen3.5 requires an environment implementing `Qwen3_5ForConditionalGeneration` with compatible PyTorch/PEFT versions. In a model-specific environment, register this package with `pip install -e .`.

| Backend | Interface | Usage |
| --- | --- | --- |
| `qwen` | Dense Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5 | Supply a model path or ID and its compatible Transformers environment |
| `qwen` | Released UAV-PUR-4B/8B complete models | Supply `Henard224/UAV-PUR-4B`, `Henard224/UAV-PUR-8B` or its downloaded directory; no `--base-model` |
| `qwen` with `--base-model` | Qwen PEFT training checkpoints | Supply the adapter as `--model` and the corresponding original base |
| `internvl` | InternVL remote-code multi-image `.chat()` | Install upstream dependencies and enable `--trust-remote-code` |
| `minicpm4` | MiniCPM-V-4.0 remote-code chat | Install upstream dependencies and enable `--trust-remote-code` |
| `minicpm45` | MiniCPM-V-4.5 remote-code chat | Install upstream dependencies and enable `--trust-remote-code` |
| `llava-ov` | Remote-code LLaVA-OneVision processor/generate | Use a checkpoint exposing this interface and enable `--trust-remote-code` |
| `api` | Vision-capable OpenAI-compatible `chat/completions` | Supply an endpoint and its exact model ID |

Local Qwen inference is greedy (`do_sample=False`, one beam), with Qwen3.5 thinking disabled. The default attention implementation is SDPA; `flash_attention_2` requires a compatible FlashAttention installation. Task-specific output limits are set by default; `--max-new-tokens` overrides them.

`--max-pixels` configures the Qwen processor's maximum image area. Leaving it unset uses the processor default. InternVL uses up to six grid tiles plus a thumbnail, 448-pixel tiles and ImageNet normalization. MiniCPM receives ordered image objects. LLaVA uses its processor with a 1008-pixel long-edge cap for MIG/MVR inputs. Results retain image sizes, token counts and processor grids when available.

The [UAV-PUR-4B](https://huggingface.co/Henard224/UAV-PUR-4B) and [UAV-PUR-8B](https://huggingface.co/Henard224/UAV-PUR-8B) releases are complete merged Transformers models, including their processors. For an unmerged training checkpoint, `--base-model` instead loads LoRA and saved main/DeepStack mergers through PEFT. Neither format requires auxiliary heads at inference.

## Commands

Evaluate all ten choice tasks:

```bash
CUDA_VISIBLE_DEVICES=0 drone-pur-eval --data-root /path/to/Drone-PUR-Benchmark --model Qwen/Qwen3-VL-4B-Instruct --mode choice --tasks all --max-pixels 1003520 --output test/outputs/qwen-choice
```

Evaluate all seven open-ended settings:

```bash
CUDA_VISIBLE_DEVICES=0 drone-pur-eval --data-root /path/to/Drone-PUR-Benchmark --model Henard224/UAV-PUR-4B --mode open --tasks all --max-pixels 1003520 --output test/outputs/uav-pur-open
```

Useful options:

| Option | Behavior |
| --- | --- |
| `--tasks MMR MVR` | Select task groups; individual identifiers such as `MMR-VG` are also accepted |
| `--sample-every 5` | Select indices 0, 5, 10, ... independently within each subtask |
| `--max-samples N` | Uniformly cap the selected set at N samples per subtask |
| `--sample-ids FILE` | Use an ID list or a task-to-ID-list mapping |
| `--text-only` | Send the complete text prompt without images |
| `--check-only` | Validate data and prompts without loading a model or sending requests |
| `--retry-invalid` | Explicitly retry completed outputs that did not parse |

Use identical selected IDs across models for paired comparisons. Rerun the same command and output directory to resume. Model, prompt, image and subset fingerprints protect cache consistency; concurrent writes to one output directory are blocked.

## API Evaluation

The API backend accepts an endpoint through the command line and reads credentials from an environment variable:

```bash
read -rsp 'API key: ' DRONE_PUR_API_KEY; echo; export DRONE_PUR_API_KEY
drone-pur-eval --backend api --model MODEL_ID --api-base-url https://HOST/v1 --data-root /path/to/Drone-PUR-Benchmark --mode choice --tasks all --output test/outputs/api-model
```

The endpoint must accept ordered image inputs through OpenAI-compatible chat completions. Original image bytes are sent by default; `--api-max-side` enables aspect-preserving resizing. `--api-params` accepts documented model parameters, such as `'{"enable_thinking":false}'` when supported. Only final-answer content is parsed.

Response model/IDs, UTC timestamps, usage and finish reasons are retained. API keys are not written to manifests. The endpoint is represented by a hash for cache identity. Keep credentials in environment variables and keep evaluation outputs separate from the source release.

Timeouts, rate limits and transient HTTP failures are retried. Persistent authentication/model errors stop after the configured error streak. Resume retries infrastructure failures while retaining completed predictions.

## Results and Metrics

| Output | Contents |
| --- | --- |
| `manifest.json` | Input/model/code fingerprints, selected IDs, runtime versions and UTC start time |
| `results.json` | Raw responses, predictions, statuses and inference metadata |
| `predictions.json` | Completed predictions for offline scoring |
| `summary.json` | Subtask, task and overall metrics with coverage |
| `summary.csv` | Flat statistics for tables |

Scoring rules:

- **Choice:** option matching; CJ uses Yes/No. Invalid completed outputs count as incorrect.
- **Grounding:** strict `IoU > 0.5`. An absent MVR-VG target requires `not exist`.
- **Counting:** MAE and MRA, where MRA averages accuracy at strict relative-error thresholds 5%, 10%, ..., 50%. For GT=0, only a zero prediction is correct. Invalid outputs contribute zero MRA. `mae_valid` uses parsed predictions; full MAE is undefined if any completed prediction is invalid.
- **OD:** pooled micro precision/recall/F1, with maximum-cardinality one-to-one matching at strict `IoU > 0.5`. Invalid completed outputs act as empty predictions.
- **Aggregation:** MMR/MVR subtasks and the ten-task overall choice result are weighted by sample counts.

`n_valid`, `n_invalid` and `n_missing` are reported separately. Missing/infrastructure failures remain distinct from completed invalid answers. Incomplete runs report `completed_metrics`; final `metrics` become available when all selected samples are complete. `valid_metrics` provides parsed-only diagnostics. Percentages use a 0-100 scale; MAE is in objects.

Parsers use task-specific extraction and syntactic normalization without GT-based correction. OD parsing accepts valid empty detections, normalizes 0-1000 coordinates, orders corners and clamps small out-of-range values. Raw responses remain available alongside parsed outputs.

Offline scoring of full-set predictions:

```bash
drone-pur-score --data-root /path/to/Drone-PUR-Benchmark --mode choice --tasks all --predictions test/outputs/qwen-choice/predictions.json --output test/outputs/qwen-choice/rescored.json
```

For sampled inference, use its saved `summary.json`. The full-set scorer treats samples outside a subset prediction file as missing.
