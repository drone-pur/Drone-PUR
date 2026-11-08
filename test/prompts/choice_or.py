"""Preserved task prompt and response parser; model loading is separate."""

from __future__ import annotations
import ast
import json
import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple


def build_or_prompt(sample: Dict) -> str:

    options = sample["options"]
    prompt = f"""
This is an object recognition task from a UAV-view image.
Carefully inspect the image and identify the target referred to by the question, usually through its normalized bbox coordinates [x1, y1, x2, y2]. x1, y1 are the normalized coordinates of the top-left corner of the target bounding box (range 0-1).x2, y2 are the normalized coordinates of the bottom-right corner of the target bounding box (range 0-1)
Choose the option that best describes the target's visible compound attributes, such as type, structure, color pattern, markings, carried items, or clear image-relative position.
Output only one option letter: A, B, C, or D. Do not output explanations or extra text.
When describing spatial relationships, all directional terms (up, down, left, right) are defined relative to the image itself, meaning the top of the image represents the "up" direction.

Question: {sample ['question']}

Options:
A. {options ['A']}
B. {options ['B']}
C. {options ['C']}
D. {options ['D']}

Answer:
"""
    return prompt.strip()


def normalize_response_text(response: str) -> str:
    text = str(response or "").strip()
    full_to_half = str.maketrans(
        {
            "Ａ": "A",
            "Ｂ": "B",
            "Ｃ": "C",
            "Ｄ": "D",
            "ａ": "A",
            "ｂ": "B",
            "ｃ": "C",
            "ｄ": "D",
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
        r"(?:答案|答|选择|选项|正确答案|Answer|ANSWER|Option|OPTION)\s*(?:是|为|:|：|=)?\s*[\(\[]?([A-D])[\)\]\.。:：、,，\s]?",
        r"^\s*[\(\[]?([A-D])[\)\]\.。:：、,，\s]*$",
        r"^\s*([A-D])\s*(?:\.|。|:|：|、|,|，|\)|）)",
    ]
    for pat in explicit_patterns:
        m = re.search(pat, upper)
        if m:
            return m.group(1), "tested", [m.group(1)]

    candidates = re.findall(r"(?<![A-Z])([A-D])(?![A-Z])", upper)
    unique = []
    for c in candidates:
        if c not in unique:
            unique.append(c)

    if len(unique) == 1:
        return unique[0], "tested", unique
    if len(unique) > 1:
        return None, "ambiguous_answer_format", unique
    return None, "invalid_answer_format", []
