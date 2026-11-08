"""Task registry, canonical-to-prompt adapter and deterministic subsets."""

from importlib import import_module
import json
import math

from .data import ANNOTATION_FILES, OPEN_TASKS

CHOICE_FUNCTIONS = {
    "VG": "build_choice_prompt",
    "OC": "build_choice_prompt",
    "OR": "build_or_prompt",
    "MIG": "build_choice_prompt",
    "CJ": "build_cj_prompt",
    "PSJ": "build_prompt",
    "SI": "build_prompt",
    "CR": "build_prompt",
    "MMR-OC": "build_dual_choice_prompt",
    "MMR-VG": "build_dual_choice_prompt",
    "MMR-R": "build_dual_choice_prompt",
    "MVR-VG": "build_choice_prompt",
    "MVR-R": "build_mvr_reasoning_prompt",
}
OPEN_FUNCTIONS = {
    "VG": "build_direct_prompt",
    "OC": "build_direct_count_prompt",
    "MIG": "build_prompt",
    "MMR-OC": "build_direct_count_prompt",
    "MMR-VG": "build_dual_direct_prompt",
    "MVR-VG": "build_direct_prompt",
    "OD": "build_direct_json_prompt",
}
CHOICE_TOKENS = {
    "OC": 20,
    "OR": 20,
    "PSJ": 32,
    "MIG": 32,
    "MVR-VG": 32,
    "MVR-R": 32,
    "SI": 32,
    "CR": 32,
    "CJ": 20,
    "MMR-OC": 20,
}
OPEN_TOKENS = {
    "OC": 30,
    "MMR-OC": 20,
    "VG": 128,
    "MMR-VG": 30,
    "MIG": 50,
    "MVR-VG": 64,
    "OD": 2048,
}
TRAIN_KEYS = {
    "OC": "oc",
    "MIG": "mig",
    "OR": "or",
    "VG": "vg",
    "PSJ": "psj",
    "CJ": "cj",
    "SI": "si",
    "CR": "cr",
    "MMR-OC": "mmr_oc",
    "MMR-VG": "mmr_vg",
    "MMR-R": "dmjr",
    "MVR-VG": "mvr_vg",
    "MVR-R": "mvr",
}


def select_tasks(names, mode):
    allowed = (
        [t for t in ANNOTATION_FILES if t != "OD"]
        if mode == "choice"
        else [t for t in ANNOTATION_FILES if t in OPEN_TASKS]
    )
    output = []
    for name in names:
        name = name.upper().replace("_", "-")
        expanded = (
            allowed
            if name == "ALL"
            else (
                [t for t in allowed if t.startswith(name + "-")]
                if name in {"MMR", "MVR"}
                else [name]
            )
        )
        if not expanded or any(t not in allowed for t in expanded):
            raise ValueError(
                f"Unsupported {mode} task: {name}. Available: {', '.join(allowed)}"
            )
        output.extend(t for t in expanded if t not in output)
    return output


def prompt_module(task, mode):
    return import_module(
        ".prompts." + mode + "_" + task.lower().replace("-", "_"), __package__
    )


def prompt_sample(row):
    # Public annotations store the template once; this adapter is input-only.
    sample = dict(row)
    if row.get("template"):
        sample["template_coordinates"] = [json.dumps(row["template"]["box"])]
    return sample


def build_prompt(row, task, mode):
    fn = (CHOICE_FUNCTIONS if mode == "choice" else OPEN_FUNCTIONS)[task]
    return getattr(prompt_module(task, mode), fn)(prompt_sample(row))


def parse_response(text, task, mode):
    module = prompt_module(task, mode)
    if mode == "choice":
        return module.parse_answer(text)[0]
    if task in {"OC", "MMR-OC"}:
        return module.parse_count(text)[0]
    if task == "OD":
        # Preserve the historical decoder, but do not accept malformed arrays
        # as a successfully parsed empty prediction in the public protocol.
        try:
            payload = module.extract_json_payload(text)
            values = payload.get("boxes") if isinstance(payload, dict) else payload
            if not isinstance(values, list):
                return None
            boxes = [module.normalize_box(value) for value in values]
            return boxes if all(box is not None for box in boxes) else None
        except (ValueError, TypeError, OverflowError):
            return None
    if task == "MVR-VG" and module.parse_not_exist(text):
        return "not exist"
    try:
        return module.parse_coordinates(text)
    except (ValueError, TypeError):
        return None


def token_limit(task, mode):
    return CHOICE_TOKENS.get(task, 50) if mode == "choice" else OPEN_TOKENS[task]


def uniform_subset(rows, every=1, max_samples=None):
    if every < 1 or max_samples is not None and max_samples < 1:
        raise ValueError("Sampling arguments must be positive")
    chosen = rows[::every]
    if max_samples is not None and len(chosen) > max_samples:
        chosen = [
            chosen[math.floor(i * len(chosen) / max_samples)]
            for i in range(max_samples)
        ]
    return chosen
