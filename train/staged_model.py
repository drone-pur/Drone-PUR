"""Qwen3-VL LM + training-only parallel box/count/set heads.

Language LoRA, full main/DeepStack mergers and active auxiliary heads train.
Merger weights are stored in PEFT modules_to_save for standard inference.
"""

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .detection import DetectionHead, generalized_iou, set_loss


@contextmanager
def ddp_adapter_load_compat():
    """PEFT 0.19 / Transformers 4.57: avoid TP-only imports for plain DDP.

    Scoped to adapter loading; never change the installed packages or suppress
    TP handling for models with a mesh, layer TP annotation, or DTensor weights.
    Newer compatible Transformers use the original implementation untouched.
    """
    from peft.utils import save_and_load
    from transformers.integrations import tensor_parallel

    original = getattr(save_and_load, "_maybe_shard_state_dict_for_tp", None)
    if original is None or hasattr(tensor_parallel, "EmbeddingParallel"):
        yield
        return

    def guarded(model, state_dict, adapter_name):
        for module in model.modules():
            if (
                any(
                    getattr(module, name, None) is not None
                    for name in (
                        "_hf_device_mesh",
                        "_device_mesh",
                        "device_mesh",
                        "_hf_tp_plan",
                    )
                )
                or (getattr(module, "_tp_size", 1) or 1) > 1
            ):
                return original(model, state_dict, adapter_name)
        if any(hasattr(p, "device_mesh") for p in model.parameters()):
            return original(model, state_dict, adapter_name)
        # Plain replicated DDP weights require no tensor-parallel sharding.
        return None

    save_and_load._maybe_shard_state_dict_for_tp = guarded
    try:
        yield
    finally:
        save_and_load._maybe_shard_state_dict_for_tp = original


def ordered_box(raw):
    values = raw.sigmoid()
    center = values[..., :2]
    half = values[..., 2:] * torch.minimum(center, 1 - center)
    return torch.cat((center - half, center + half), dim=-1)


def merger_module_names(base, config):
    """Wrap each callable merger, never the ModuleList container itself."""
    visual = base.model.visual
    count = len(visual.deepstack_merger_list)
    expected = config.get("expected_deepstack_mergers", 3)
    if count != expected:
        raise ValueError(f"Expected {expected} DeepStack mergers, found {count}")
    return ["model.visual.merger"] + [
        f"model.visual.deepstack_merger_list.{i}" for i in range(count)
    ]


def is_merger_parameter(name):
    return ".visual.merger." in name or ".visual.deepstack_merger_list." in name


def architecture_signature(value):
    volatile = {
        "transformers_version",
        "torch_dtype",
        "dtype",
        "architectures",
        "use_cache",
        "return_dict",
        "output_hidden_states",
        "output_attentions",
    }
    if isinstance(value, dict):
        return {
            str(k): architecture_signature(v)
            for k, v in value.items()
            if not str(k).startswith("_") and k not in volatile
        }
    if isinstance(value, (list, tuple)):
        return [architecture_signature(v) for v in value]
    return value


def optimizer_groups(model, config, stage):
    """Separate LoRA/merger/head LR, with identical decay rules to control."""
    rates = {
        "lora": stage["lr"],
        "merger": stage["lr"] * config["merger_lr_ratio"],
        "head": config["head_lr"],
    }
    buckets = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        kind = (
            "head"
            if not name.startswith("backbone.")
            else "merger" if is_merger_parameter(name) else "lora"
        )
        decay = p.ndim > 1 and not name.endswith("bias")
        buckets.setdefault((kind, decay), []).append(p)
    return [
        {
            "params": ps,
            "lr": rates[kind],
            "weight_decay": 0.01 if decay else 0,
            "group_name": kind + ("_decay" if decay else "_no_decay"),
        }
        for (kind, decay), ps in buckets.items()
    ]


def answer_ce(hidden, labels, lm_head, chunk_size=128):
    """Mean over answer tokens per sample, then mean over samples.

    Project only supervised next-token positions, checkpointing vocab projection
    to avoid retaining huge [sequence, vocabulary] logits for OD answers.
    """
    per_sample = []
    for h, label in zip(hidden, labels):
        positions = torch.where(label[1:] != -100)[0]
        if not len(positions):
            raise ValueError("No supervised answer tokens")
        values, targets = h[positions], label[positions + 1]
        total = h.sum() * 0
        for start in range(0, len(positions), chunk_size):

            def ce(x, y):
                return F.cross_entropy(lm_head(x).float(), y, reduction="sum")

            total = total + checkpoint(
                ce,
                values[start : start + chunk_size],
                targets[start : start + chunk_size],
                use_reentrant=False,
            )
        per_sample.append(total / len(positions))
    return torch.stack(per_sample)


class StagedModel(nn.Module):
    def __init__(self, backbone, config, stage):
        super().__init__()
        self.backbone = backbone
        self.settings = config
        self.stage = stage
        if stage == "s3" and not config["stages"][stage]["aux"]:
            return
        dim = backbone.config.text_config.hidden_size
        width = config["head_width"]

        def mlp(out):
            return nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, width),
                nn.GELU(),
                nn.Linear(width, out),
            )

        self.box_head = mlp(4)
        self.count_head = nn.Sequential(mlp(1), nn.Softplus())
        self.od_head = DetectionHead(
            dim, config["num_queries"], config["od_width"], config["od_layers"]
        )
        self.stage = stage
        active = config["stages"][stage]["aux"]
        self.box_head.requires_grad_(active)
        self.count_head.requires_grad_(active)
        self.od_head.requires_grad_(active and stage in {"s1", "s2"})

    def forward(self, labels, prompt_lengths, aux_kinds, targets, **inputs):
        # LoRA modules are installed in the original model; calling its multimodal
        # core avoids allocating prompt vocab logits but does NOT detach LoRA.
        full = self.backbone.get_base_model()
        hidden = full.model(
            **inputs, use_cache=False, return_dict=True
        ).last_hidden_state
        ce = answer_ce(hidden, labels, full.lm_head)
        aux = ce * 0
        # Metrics are sums over applicable examples, plus counts for proper logging.
        metrics = torch.zeros(9, device=hidden.device)
        metrics[0], metrics[1] = ce.detach().sum(), len(ce)
        if self.settings["stages"][self.stage]["aux"]:
            weights = self.settings["loss_weights"]
            for i, kind in enumerate(aux_kinds):
                end = int(prompt_lengths[i])
                condition = hidden[i, end - 1].float()
                if kind == "box":
                    predicted = ordered_box(self.box_head(condition)).float()
                    gt = torch.as_tensor(
                        targets[i], device=hidden.device, dtype=torch.float32
                    )
                    l1 = (predicted - gt).abs().mean()
                    giou = 1 - generalized_iou(predicted[None], gt[None])[0, 0]
                    aux[i] = weights["box"] * (l1 + giou)
                    metrics[2] += l1.detach()
                    metrics[3] += giou.detach()
                    metrics[4] += 1
                elif kind == "count":
                    predicted = self.count_head(condition).float().squeeze()
                    mae = (predicted - float(targets[i])).abs()
                    aux[i] = weights["count"] * mae
                    metrics[5] += mae.detach()
                    metrics[6] += 1
                elif kind == "od":
                    if self.stage not in {"s1", "s2"}:
                        raise ValueError("OD auxiliary loss requires s1 or s2")
                    # Slicing, not just padding masking: no answer position enters OD memory.
                    memory = hidden[i : i + 1, :end].float()
                    mask = torch.ones((1, end), dtype=torch.bool, device=hidden.device)
                    boxes, scores = self.od_head(memory, mask)
                    gt = torch.as_tensor(targets[i], device=hidden.device).reshape(
                        -1, 4
                    )
                    loss, _ = set_loss(boxes, scores, [gt])
                    aux[i] = weights["od"] * loss
                    metrics[7] += loss.detach()
                    metrics[8] += 1
                elif kind != "none":
                    raise ValueError(f"Unknown auxiliary kind: {kind}")
        return {"loss": (ce + aux).mean(), "metrics": metrics.detach()}

    def save_weights(self, path):
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path, safe_serialization=True)
        weights = {
            n: v.detach().cpu().contiguous().clone()
            for n, v in self.state_dict().items()
            if not n.startswith("backbone.")
        }
        if weights:
            save_file(weights, str(path / "auxiliary_heads.safetensors"))
        (path / "aux_config.json").write_text(json.dumps(self.settings, indent=2))


def build_model(model_path, config, stage, init=None, dtype=torch.bfloat16):
    from transformers import Qwen3VLForConditionalGeneration
    from peft import LoraConfig, PeftModel, get_peft_model

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="sdpa"
    )
    base.config.use_cache = False
    mergers = merger_module_names(base, config) if config.get("train_mergers") else []
    if init:
        old = json.loads((Path(init) / "aux_config.json").read_text())
        if old != config:
            import copy

            cursor = json.loads((Path(init) / "training_cursor.json").read_text())
            allowed = copy.deepcopy(old)
            allowed["stages"]["s3"] = config["stages"]["s3"]
            if not (
                stage == "s3"
                and not config["stages"][stage]["aux"]
                and cursor["identity"]["stage"] == "s2"
                and cursor["stage_finished"]
                and allowed == config
            ):
                raise ValueError(
                    "Only S3 settings may change when branching from completed S2"
                )
        adapter = json.loads((Path(init) / "adapter_config.json").read_text())
        if set(adapter.get("modules_to_save") or []) != set(mergers):
            raise ValueError("Checkpoint merger modules_to_save mismatch")
        source_config = Path(init) / "base_config.json"
        if source_config.is_file():
            original = architecture_signature(json.loads(source_config.read_text()))
            current = architecture_signature(base.config.to_dict())
            if original != current:
                raise ValueError("Checkpoint base architecture differs from --model")
        elif (
            Path(adapter["base_model_name_or_path"]).resolve()
            != Path(model_path).resolve()
        ):
            raise ValueError(
                "Legacy checkpoint base moved: export base_config.json before migrating it"
            )
        with ddp_adapter_load_compat():
            base = PeftModel.from_pretrained(base, init, is_trainable=True)
    else:
        targets = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
        base = get_peft_model(
            base,
            LoraConfig(
                r=32,
                lora_alpha=64,
                lora_dropout=0.05,
                target_modules=targets,
                modules_to_save=mergers or None,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    # The wrapper original_module stays frozen; only the active saved copy trains.
    for n, p in base.named_parameters():
        if p.requires_grad:
            language_lora = "lora_" in n and ".language_model.layers." in n
            saved_merger = (
                bool(mergers)
                and is_merger_parameter(n)
                and ".modules_to_save.default." in n
            )
            if not (language_lora or saved_merger):
                raise ValueError(f"Unexpected trainable backbone parameter: {n}")
            p.data = p.data.float()
    for name in mergers:
        module = base.get_base_model().get_submodule(name)
        if not all(
            p.requires_grad for p in module.modules_to_save["default"].parameters()
        ):
            raise ValueError(f"Merger is not fully trainable: {name}")
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    result = StagedModel(base, config, stage)
    if init and config["stages"][stage]["aux"]:
        from safetensors.torch import load_file

        state = load_file(str(Path(init) / "auxiliary_heads.safetensors"))
        expected = {n for n in result.state_dict() if not n.startswith("backbone.")}
        if set(state) != expected:
            raise ValueError("Auxiliary checkpoint keys mismatch")
        result.load_state_dict(state, strict=False)
    return result
