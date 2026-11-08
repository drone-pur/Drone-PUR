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
This is a dual-modal joint counting task from UAV images.
You will receive two spatially aligned images of the same scene:
Image 1 is RGB visible light, and Image 2 is TIR thermal infrared.
If they are shown as one side-by-side composite image, the left half is RGB and the right half is TIR.

Task: directly answer the counting question by combining both modalities.
Count the target objects specified by the question. Use RGB and TIR together, including targets visible in only one modality.
Do not double-count the same physical object if it appears in both RGB and TIR.
If the question specifies a class or region, count only objects that match that class or region.

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
            "Ｅ": "E",
            "ａ": "A",
            "ｂ": "B",
            "ｃ": "C",
            "ｄ": "D",
            "ｅ": "E",
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
