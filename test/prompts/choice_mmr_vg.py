"""Preserved task prompt and response parser; model loading is separate."""

from __future__ import annotations
import ast
import json
import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple


def build_dual_choice_prompt(sample: Dict) -> str:

    options = sample["options"]
    prompt = f"""
This is a dual-modal joint grounding task. You will receive two spatially aligned images:
The first is an RGB visible light image, and the second is a TIR thermal infrared image. Both images are captured at the same time point and are roughly aligned in spatial position.
Task requirements: Combine information from both modalities to select the bounding box that matches the target object described in the question from the options.
Note that some targets cannot be uniquely identified by a single modality alone; you need to comprehensively utilize information from both modalities to analyze the target's position.

Direction note: All directional terms (up, down, left, right) refer to the image frame itself, i.e., the top of the image represents "up".
Bounding box format: [x1, y1, x2, y2], where x1, y1 are the normalized coordinates of the top-left corner, and x2, y2 are the normalized coordinates of the bottom-right corner (range 0-1).
Output requirements: Only output the option letter (A/B/C/D/E) corresponding to the correct bounding box. Do not output any other content. Example output: A

Question: {sample ['question']}

Options:
A. {options ['A']}
B. {options ['B']}
C. {options ['C']}
D. {options ['D']}
E. {options .get ('E','')}

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
        r"(?:答案|答|选择|选项|正确答案|Answer|ANSWER|Option|OPTION)\s*(?:是|为|:|：|=)?\s*[\(\[]?([A-E])[\)\]\.。:：、,，\s]?",
        r"^\s*[\(\[]?([A-E])[\)\]\.。:：、,，\s]*$",
        r"^\s*([A-E])\s*(?:\.|。|:|：|、|,|，|\)|）)",
    ]
    for pat in explicit_patterns:
        m = re.search(pat, upper)
        if m:
            return m.group(1), "tested", [m.group(1)]

    candidates = re.findall(r"(?<![A-Z])([A-E])(?![A-Z])", upper)
    unique = []
    for c in candidates:
        if c not in unique:
            unique.append(c)

    if len(unique) == 1:
        return unique[0], "tested", unique
    if len(unique) > 1:
        return None, "ambiguous_answer_format", unique
    return None, "invalid_answer_format", []
