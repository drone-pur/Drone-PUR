"""Preserved task prompt and response parser; model loading is separate."""

from __future__ import annotations
import ast
import json
import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

ABSENT_ANSWER = "not exist"


def build_choice_prompt(sample: Dict, include_template_bbox: bool = True) -> str:
    options = sample["options"]
    template_bbox_note = ""
    if include_template_bbox:
        template_bbox_note = f"""
The query target in Image-1 is marked by this normalized bounding box:
{sample ["template_coordinates"][0 ]}
"""

    prompt = f"""
This is a UAV multi-view target matching multiple-choice task. You will receive two images captured at the same time from different UAV viewpoints.

Image-1 is the query image. The query target is the object inside the red box in Image-1.
Image-2 is the candidate image from another viewpoint of the same scene. Your task is to identify where the same physical target from Image-1 appears in Image-2, and answer with exactly one option letter.

The same target may look different across views: its position, scale, orientation, visible side, and surrounding context can change. Use both object appearance and the global scene layout to match the same physical object.

Important absence rule:
- The query target from Image-1 may actually not exist as a visible answer in Image-2.
- Some options may contain "{ABSENT_ANSWER }".
- Choose the "{ABSENT_ANSWER }" option only when the query target is not present in Image-2, or when it is present in the scene but is essentially impossible to identify because another object fully occludes it, with about 95% or more of its own area occluded.
- If the target is partially visible, very small, truncated by the image boundary, viewed from another angle, or difficult but still identifiable, do not choose "{ABSENT_ANSWER }"; choose the bbox option that best covers that same target in Image-2.
- If "{ABSENT_ANSWER }" appears as a distractor while the target is visible in Image-2, choose the correct bbox option instead.
{template_bbox_note }

Bounding box options use normalized Image-2 coordinates in the format [x1, y1, x2, y2]. x1 and y1 are the top-left coordinates; x2 and y2 are the bottom-right coordinates; all values are in [0, 1].

Question: {sample ["question"]}

Options:
A. {options ["A"]}
B. {options ["B"]}
C. {options ["C"]}
D. {options ["D"]}
E. {options ["E"]}

Output only one option letter: A, B, C, D, or E. Do not output explanations, coordinates, or any other text.

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
