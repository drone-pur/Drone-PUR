"""Preserved task prompt and response parser; model loading is separate."""

from __future__ import annotations
import ast
import json
import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple


def build_direct_count_prompt(sample: Dict) -> str:

    prompt = f"""
This is an object counting task from a UAV-view image.
Carefully inspect the image and count the target objects described in the question.
Output only one integer, such as 17. Do not output explanations, option letters, units, punctuation, or extra text.

Question: {sample ['question']}

Count:
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


def parse_count(response: str) -> Tuple[Optional[int], str, List[int]]:

    text = normalize_response_text(response)
    if not text:
        return None, "invalid_answer_format", []

    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
        if not text:
            return None, "invalid_answer_format", []

    text = text.replace(",", "")
    explicit_patterns = [
        r"^\s*(\d+)\s*$",
        r"(?:count|answer|total|number|人数|数量|答案)\s*(?:is|为|是|:|：|=)?\s*(\d+)",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1)), "tested", [int(match.group(1))]

    numbers = [int(x) for x in re.findall(r"(?<![\d.])\d+(?![\d.])", text)]
    unique = []
    for number in numbers:
        if number not in unique:
            unique.append(number)

    if len(unique) == 1:
        return unique[0], "tested", unique
    if len(unique) > 1:
        return None, "ambiguous_answer_format", unique
    return None, "invalid_answer_format", []
