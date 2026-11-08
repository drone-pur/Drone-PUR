"""Preserved task prompt and response parser; model loading is separate."""

from __future__ import annotations
import ast
import json
import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple


def build_choice_prompt(sample: Dict, include_template_bbox: bool = True) -> str:
    options = sample["options"]
    template_box = sample["template_coordinates"][0]
    query_frame = sample.get("query_frame", 3)
    template_bbox_note = ""
    if include_template_bbox:
        template_bbox_note = f"""
The template target in Image 1 is marked by this normalized bounding box:
{template_box }
"""

    prompt = f"""
This is a UAV multi-image grounding task. You will receive four temporally ordered images from the same video sequence: Image 1 is the template frame, and Images 2-4 are later frames. The target object and the UAV camera viewpoint may both move over time.

Your task is to identify the same target object marked by the template bounding box in Image 1, then answer where that target is located in the query image specified by the question. Do not choose a visually similar distractor; track the same physical object across the image sequence.
{template_bbox_note }

Bounding box format: [x1, y1, x2, y2]. x1 and y1 are the normalized coordinates of the top-left corner. x2 and y2 are the normalized coordinates of the bottom-right corner. All values are in the range [0, 1].

The options below are candidate bounding boxes in Image {query_frame }. Select the option that best matches the same target object from Image 1.

Question: {sample ["question"]}

Options:
A. {options ["A"]}
B. {options ["B"]}
C. {options ["C"]}
D. {options ["D"]}

Output only one option letter: A, B, C, or D. Do not output explanations, coordinates, or any other text.

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
        r"(?:答案|答|选择|选项|正确答案|ANSWER|Answer|OPTION|Option)\s*(?:是|为|:|：|=)?\s*[\(\[]?([A-D])[\)\]\.。:：、,，\s]?",
        r"^\s*[\(\[]?([A-D])[\)\]\.。:：、,，\s]*$",
        r"^\s*([A-D])\s*(?:\.|。|:|：|、|,|，|\)|）)",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1), "tested", [match.group(1)]

    candidates = re.findall(r"(?<![A-Z])([A-D])(?![A-Z])", upper)
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)

    if len(unique) == 1:
        return unique[0], "tested", unique
    if len(unique) > 1:
        return None, "ambiguous_answer_format", unique
    return None, "invalid_answer_format", []
