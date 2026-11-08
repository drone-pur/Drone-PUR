"""Preserved task prompt and response parser; model loading is separate."""

from __future__ import annotations
import ast
import json
import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

OPTION_KEYS = ["A", "B", "C", "D", "E"]


def normalize_options(options) -> Dict[str, str]:
    if isinstance(options, dict):
        return {key: str(options.get(key, "")) for key in OPTION_KEYS}
    if isinstance(options, list):
        return {
            key: str(options[idx]) if idx < len(options) else ""
            for idx, key in enumerate(OPTION_KEYS)
        }
    return {key: "" for key in OPTION_KEYS}


def choose_text(sample: Dict, field: str, use_zh: bool = False) -> str:
    if use_zh:
        return str(sample.get(f"{field }_zh") or sample.get(field) or "")
    return str(sample.get(field) or sample.get(f"{field }_zh") or "")


def choose_options(sample: Dict, use_zh: bool = False) -> Dict[str, str]:
    if use_zh and sample.get("options_zh"):
        return normalize_options(sample.get("options_zh"))
    return normalize_options(sample.get("options") or sample.get("options_zh") or {})


def build_mvr_reasoning_prompt(sample: Dict, use_zh: bool = False) -> str:
    question = choose_text(sample, "question", use_zh=use_zh)
    options = choose_options(sample, use_zh=use_zh)

    return f"""
This is a multi-view reasoning multiple-choice question task for UAV imagery. 
You will be provided with two images of the same scene captured simultaneously from different UAV viewpoints, Image-1 and Image-2 (in the order they are input).
The two images are time-aligned but their field of view, scale, viewing angle, visible sides of objects, and crop boundaries may differ. 

Task requirements:
- Answer the question by jointly reasoning over visual clues from both Image-1 and Image-2.
- A single view may be insufficient or misleading; use cross-view evidence to resolve ambiguity.
- Pay attention to view-specific evidence: some clues may be visible only in one image, clearer in one image, occluded, cropped, or seen from a different angle in the other.
- When using spatial relations, using image directions such as top, bottom, left, right, and center.

Select the best answer from A/B/C/D/E.

Output requirements: Only output one option letter (A/B/C/D/E). Do not output explanations or any other text. Example output: A

Question: {question }

Options:
A. {options ["A"]}
B. {options ["B"]}
C. {options ["C"]}
D. {options ["D"]}
E. {options ["E"]}

Answer:
""".strip()


def normalize_response_text(response: str) -> str:
    text = str(response or "").strip()
    full_to_half = str.maketrans(
        {
            "Ａ": "A",
            "Ｂ": "B",
            "Ｃ": "C",
            "Ｄ": "D",
            "Ｅ": "E",
            "ａ": "A",
            "ｂ": "B",
            "ｃ": "C",
            "ｄ": "D",
            "ｅ": "E",
        }
    )
    return text.translate(full_to_half)


def parse_answer(response: str) -> Tuple[Optional[str], str, List[str]]:
    text = normalize_response_text(response)
    if not text:
        return None, "invalid_answer_format", []
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
        if not text:
            return None, "invalid_answer_format", []

    upper = text.upper()
    explicit_patterns = [
        r"(?:答案|答|选择|选项|正确答案|ANSWER|Answer|OPTION|Option)\s*(?:是|为|:|：|=)?\s*[\(\[]?([A-E])[\)\]\.。:：、,，\s]?",
        r"^\s*[\(\[]?([A-E])[\)\]\.。:：、,，\s]*$",
        r"^\s*([A-E])\s*(?:\.|。|:|：|、|,|，|\)|）)",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1), "tested", [match.group(1)]

    candidates = re.findall(r"(?<![A-Z])([A-E])(?![A-Z])", upper)
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    if len(unique) == 1:
        return unique[0], "tested", unique
    if len(unique) > 1:
        return None, "ambiguous_answer_format", unique
    return None, "invalid_answer_format", []
