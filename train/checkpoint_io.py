"""Full optimizer/RNG checkpoint without pickle; safe tensors + typed JSON tree."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def save_tree(path, value):
    path = Path(path)
    tensors = {}

    def encode(x):
        if isinstance(x, torch.Tensor):
            key = f"t{len(tensors)}"
            tensors[key] = x.detach().cpu().contiguous().clone()
            return {"type": "tensor", "key": key}
        if isinstance(x, dict):
            return {
                "type": "dict",
                "items": [[encode(k), encode(v)] for k, v in x.items()],
            }
        if isinstance(x, (tuple, list)):
            return {
                "type": "tuple" if isinstance(x, tuple) else "list",
                "items": [encode(v) for v in x],
            }
        if x is None or isinstance(x, (int, float, str, bool)):
            return {"type": "value", "value": x}
        raise TypeError(f"Unsupported checkpoint type: {type(x)}")

    tree = encode(value)
    save_file(tensors, str(path.with_suffix(".safetensors")))
    path.with_suffix(".json").write_text(json.dumps(tree))


def load_tree(path):
    path = Path(path)
    tensors = load_file(str(path.with_suffix(".safetensors")))

    def decode(x):
        kind = x["type"]
        if kind == "tensor":
            return tensors[x["key"]]
        if kind == "dict":
            return {decode(k): decode(v) for k, v in x["items"]}
        if kind in ("tuple", "list"):
            values = [decode(v) for v in x["items"]]
            return tuple(values) if kind == "tuple" else values
        if kind == "value":
            return x["value"]
        raise ValueError(f"Unknown checkpoint node: {kind}")

    return decode(json.loads(path.with_suffix(".json").read_text()))
