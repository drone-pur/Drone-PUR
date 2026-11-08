"""Export a Qwen3-VL adapter and saved visual mergers as a standalone model."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import shutil

from ..data import read_json, sha256, write_json
from .staged_model import ddp_adapter_load_compat


MERGERS = ["model.visual.merger"] + [
    f"model.visual.deepstack_merger_list.{i}" for i in range(3)
]
TEXT_FILES = {
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "chat_template.jinja",
    "chat_template.json",
    "README.md",
    "LICENSE",
    "NOTICE",
    "model.safetensors.index.json",
}


def portable_metadata(value, repo_id):
    if isinstance(value, dict):
        return {
            key: (
                repo_id
                if key in {"_name_or_path", "name_or_path"}
                else portable_metadata(item, repo_id)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [portable_metadata(item, repo_id) for item in value]
    return value


def tensor_hash(tensor):
    import torch

    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def merger_hashes(model, wrapped=False):
    result = {}
    for name in MERGERS:
        module = model.get_submodule(name)
        if wrapped:
            module = module.modules_to_save["default"]
        for key, value in module.state_dict().items():
            result[name + "." + key] = tensor_hash(value)
    return result


def expected_lora_hashes(model):
    import torch
    from peft.tuners.lora.layer import LoraLayer

    result = {}
    with torch.no_grad():
        for name, layer in model.named_modules():
            if not isinstance(layer, LoraLayer):
                continue
            if set(layer.lora_A) != {"default"} or layer.use_dora.get("default", False):
                raise ValueError(
                    "Standalone export expects a single standard LoRA adapter"
                )
            a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
            delta = (b @ a) * layer.scaling["default"]
            if layer.fan_in_fan_out:
                delta = delta.T
            weight = layer.get_base_layer().weight.detach().clone()
            weight.add_(delta.to(weight.dtype))
            result[name + ".weight"] = tensor_hash(weight)
    if not result:
        raise ValueError("No LoRA layers were found")
    return result


def check_merged_lora(model, expected):
    state = model.state_dict()
    for name, digest in expected.items():
        if name not in state or tensor_hash(state[name]) != digest:
            raise ValueError(
                f"Merged LoRA weight does not equal its expected update: {name}"
            )


def check_loaded_adapter(model, source):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors import safe_open

    loaded = get_peft_model_state_dict(model)
    with safe_open(str(source), framework="pt", device="cpu") as saved:
        if set(loaded) != set(saved.keys()):
            raise ValueError("Loaded adapter keys differ from the source checkpoint")
        for name, value in loaded.items():
            expected = saved.get_tensor(name).to(dtype=value.dtype, device=value.device)
            if not torch.equal(value, expected):
                raise ValueError(f"Adapter tensor was not loaded exactly: {name}")
    return len(loaded)


def validation_inputs(processor, image_paths):
    import copy
    from PIL import Image
    from ..backends import set_max_pixels

    processor = copy.deepcopy(processor)
    set_max_pixels(processor, 65536)

    cases = [("text", [], "Reply with the word OK.")]
    if image_paths:
        cases.append(
            (
                "single_image",
                image_paths[:1],
                "Describe this image in one short sentence.",
            )
        )
    if len(image_paths) > 1:
        cases.append(
            ("two_images", image_paths[:2], "Describe each image briefly in order.")
        )
    result = {}
    for name, paths, prompt in cases:
        images = []
        try:
            for path in paths:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
            content = [{"type": "image"} for _ in images] + [
                {"type": "text", "text": prompt}
            ]
            text = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
            if images:
                kwargs["images"] = images
            result[name] = dict(processor(**kwargs))
        finally:
            for image in images:
                image.close()
    return result


def run_smoke(model, inputs, device, tokens):
    import torch

    model.to(device).eval()
    result = {}
    with torch.inference_mode():
        for name, case in inputs.items():
            case = {key: value.to(device) for key, value in case.items()}
            scores = model(**case, use_cache=False).logits[:, -1].float().cpu()
            generated = model.generate(
                **case,
                max_new_tokens=tokens,
                do_sample=False,
                num_beams=1,
            )[:, case["input_ids"].shape[1] :].cpu()
            result[name] = {"logits": scores, "tokens": generated}
    model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def compare_smoke(reference, candidate, exact=False):
    import torch

    result = {}
    for name, before in reference.items():
        after = candidate[name]
        delta = (before["logits"] - after["logits"]).abs()
        lhs, rhs = before["logits"].double(), after["logits"].double()
        relative_rmse = (
            (lhs - rhs).square().mean().sqrt()
            / lhs.square().mean().sqrt().clamp_min(1.0)
        ).item()
        kl = (
            torch.nn.functional.kl_div(
                rhs.log_softmax(-1),
                lhs.log_softmax(-1),
                log_target=True,
                reduction="batchmean",
            )
            .clamp_min(0)
            .item()
        )
        finite = bool(torch.isfinite(lhs).all() and torch.isfinite(rhs).all())
        equal = torch.equal(before["logits"], after["logits"])
        result[name] = {
            "finite_logits": finite,
            "exact_logits_equal": equal,
            "check_kind": "serialization_exact" if exact else "rounding_diagnostic",
            "max_abs_logit_difference": delta.max().item(),
            "mean_abs_logit_difference": delta.mean().item(),
            "relative_logit_rmse": relative_rmse,
            "next_token_kl_nats": kl,
            "top1_equal": before["logits"].argmax(-1).tolist()
            == after["logits"].argmax(-1).tolist(),
            "greedy_tokens_equal": torch.equal(before["tokens"], after["tokens"]),
            "reference_tokens": before["tokens"].tolist(),
            "candidate_tokens": after["tokens"].tolist(),
        }
        if (
            not finite
            or exact
            and (not equal or not result[name]["greedy_tokens_equal"])
        ):
            raise ValueError(f"Model verification failed for {name}: {result[name]}")
    return result


def validate_release(folder):
    from safetensors import safe_open

    folder = Path(folder)
    for required in (
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "README.md",
        "LICENSE",
        "NOTICE",
    ):
        if not (folder / required).is_file():
            raise FileNotFoundError(folder / required)
    if read_json(folder / "config.json")["model_type"] != "qwen3_vl":
        raise ValueError("Expected a standalone Qwen3-VL configuration")
    keys, locations = set(), {}
    files = list(folder.iterdir())
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Unexpected release entry: {path.name}")
        if re.fullmatch(r"model(?:-\d{5}-of-\d{5})?\.safetensors", path.name):
            with safe_open(str(path), framework="pt", device="cpu") as weights:
                for key in weights.keys():
                    if key in keys or any(
                        part in key
                        for part in (
                            "lora_",
                            "modules_to_save",
                            "original_module",
                            "auxiliary_heads",
                        )
                    ):
                        raise ValueError(f"Non-standalone or duplicate tensor: {key}")
                    keys.add(key)
                    locations[key] = path.name
        elif path.name not in TEXT_FILES:
            raise ValueError(f"Unexpected release file: {path.name}")
        else:
            text = path.read_text(encoding="utf-8")
            if re.search(r"sk-[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{25,}", text):
                raise ValueError(f"Possible credential in {path.name}")
            if path.name not in {
                "tokenizer.json",
                "vocab.json",
                "merges.txt",
            } and re.search(r"/(?:root|mnt|home)/(?:[^\s\"']+/)*", text):
                raise ValueError(f"Machine-local path in {path.name}")
    if not keys:
        raise ValueError("No complete model tensors were exported")
    for module in MERGERS:
        if not any(key.startswith(module + ".") for key in keys):
            raise ValueError(f"Missing visual merger: {module}")
    index = folder / "model.safetensors.index.json"
    if index.is_file() and read_json(index)["weight_map"] != locations:
        raise ValueError("Shard index does not match the serialized weights")
    if (
        not index.is_file()
        and len([p for p in files if p.suffix == ".safetensors"]) != 1
    ):
        raise ValueError("Multiple model shards require an index")
    return {
        "tensor_count": len(keys),
        "total_bytes": sum(p.stat().st_size for p in files),
    }


def check_release_report(folder, report_path):
    report = read_json(report_path)
    if (
        not report.get("passed")
        or Path(report["output"]).resolve() != Path(folder).resolve()
    ):
        raise ValueError("No successful verification report for this export")
    expected = report["files_sha256"]
    actual = {
        path.name: sha256(path) for path in Path(folder).iterdir() if path.is_file()
    }
    if actual != expected:
        raise ValueError(
            "Release files changed after verification; revalidate before upload"
        )
    return validate_release(folder)


def export_model(args):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch.set_num_threads(args.cpu_threads)

    output, report_path = args.output.resolve(), args.report.resolve()
    adapter, base = args.adapter.resolve(), args.base_model.resolve()
    pending = output.with_name(output.name + ".incomplete")
    if output.exists() or pending.exists():
        raise FileExistsError(
            "Use a new output directory; exports never overwrite existing files"
        )
    if report_path.is_relative_to(output) or report_path.is_relative_to(pending):
        raise ValueError("Keep internal verification reports outside the upload folder")
    for source in (base, adapter):
        if output.is_relative_to(source) or pending.is_relative_to(source):
            raise ValueError("Export output must be separate from source checkpoints")
    if args.require_complete and not (adapter / "COMPLETE").is_file():
        raise ValueError("Source checkpoint is not marked COMPLETE")
    settings = read_json(adapter / "adapter_config.json")
    if set(settings.get("modules_to_save") or []) != set(MERGERS):
        raise ValueError(
            "Adapter must save the main merger and all three DeepStack mergers"
        )
    model_file = adapter / "adapter_model.safetensors"
    source_hashes = {
        name: sha256(adapter / name)
        for name in ("adapter_config.json", "adapter_model.safetensors")
    }
    print("Loading base model and adapter on CPU", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base,
        torch_dtype=getattr(torch, args.dtype),
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    with ddp_adapter_load_compat():
        model = PeftModel.from_pretrained(
            model, adapter, is_trainable=False, local_files_only=True
        ).eval()
    loaded_count = check_loaded_adapter(model, model_file)
    expected_mergers = merger_hashes(model.get_base_model(), wrapped=True)
    processor = AutoProcessor.from_pretrained(adapter, local_files_only=True)
    inputs = validation_inputs(processor, args.verify_image)
    print("Checking adapter inference", flush=True)
    reference = run_smoke(model, inputs, args.verify_device, args.verify_tokens)
    print("Merging LoRA and retaining trained visual mergers", flush=True)
    expected_lora = expected_lora_hashes(model.get_base_model())
    model = model.merge_and_unload(safe_merge=True).eval()
    check_merged_lora(model, expected_lora)
    if merger_hashes(model) != expected_mergers:
        raise ValueError("Trained merger weights changed during merge")
    merged = run_smoke(model, inputs, args.verify_device, args.verify_tokens)
    merge_check = compare_smoke(reference, merged)
    pending.mkdir(parents=True)
    model.config._name_or_path = args.repo_id
    model.config.use_cache = True
    model.save_pretrained(
        pending, safe_serialization=True, max_shard_size=args.max_shard_size
    )
    processor.save_pretrained(pending)
    for path in pending.glob("*.json"):
        if path.name not in {
            "tokenizer.json",
            "vocab.json",
            "model.safetensors.index.json",
        }:
            write_json(path, portable_metadata(read_json(path), args.repo_id))
    for source, name in (
        (args.model_card, "README.md"),
        (args.license_file, "LICENSE"),
        (args.notice_file, "NOTICE"),
    ):
        shutil.copyfile(source, pending / name)
    structure = validate_release(pending)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Reloading the standalone model without PEFT", flush=True)
    model, loading = Qwen3VLForConditionalGeneration.from_pretrained(
        pending,
        torch_dtype=getattr(torch, args.dtype),
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Standalone reload was not clean: {loading}")
    if merger_hashes(model) != expected_mergers:
        raise ValueError("Serialized merger weights differ from the checkpoint")
    check_merged_lora(model, expected_lora)
    restored_processor = AutoProcessor.from_pretrained(pending, local_files_only=True)
    restored_inputs = validation_inputs(restored_processor, args.verify_image)
    for name, case in inputs.items():
        if set(case) != set(restored_inputs[name]) or any(
            not torch.equal(value, restored_inputs[name][key])
            for key, value in case.items()
        ):
            raise ValueError("Processor inputs changed after serialization")
    restored = run_smoke(model, restored_inputs, args.verify_device, args.verify_tokens)
    reload_check = compare_smoke(merged, restored, exact=True)
    for name, expected in source_hashes.items():
        if sha256(adapter / name) != expected:
            raise ValueError("Source checkpoint changed while exporting")
    report = {
        "passed": True,
        "repo_id": args.repo_id,
        "base_id": args.base_id,
        "output": str(output),
        "base_model": str(base),
        "adapter": str(adapter),
        "source_sha256": source_hashes,
        "dtype": args.dtype,
        "adapter_tensor_count": loaded_count,
        "merger_tensor_count": len(expected_mergers),
        "verified_lora_layer_count": len(expected_lora),
        "verification_images": [str(p.resolve()) for p in args.verify_image],
        "merge_check": merge_check,
        "reload_check": reload_check,
        "structure": structure,
        "files_sha256": {p.name: sha256(p) for p in pending.iterdir() if p.is_file()},
    }
    del model
    pending.rename(output)
    write_json(report_path, report)
    print(
        json.dumps(
            {"output": str(output), "report": str(report_path), **structure}, indent=2
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--base-id")
    parser.add_argument("--model-card", type=Path)
    parser.add_argument("--license-file", type=Path)
    parser.add_argument("--notice-file", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--verify-image", type=Path, action="append", default=[])
    parser.add_argument("--verify-device", default="cpu")
    parser.add_argument("--verify-tokens", type=int, default=16)
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--check-release", type=Path)
    args = parser.parse_args()
    if args.check_release:
        print(
            json.dumps(check_release_report(args.check_release, args.report), indent=2)
        )
        return
    for name in (
        "base_model",
        "adapter",
        "output",
        "repo_id",
        "base_id",
        "model_card",
        "license_file",
        "notice_file",
    ):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required for export")
    export_model(args)


if __name__ == "__main__":
    main()
