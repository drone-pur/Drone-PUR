"""Preserved task prompt and response parser; model loading is separate."""

from __future__ import annotations
import ast
import json
import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple


def build_dual_choice_prompt(sample: Dict) -> str:

    question = sample.get("question") or sample.get("question_zh", "")
    options = sample.get("options") or sample.get("options_zh", {})
    prompt = f"""
This is a dual-modal joint reasoning task. You will receive two images:
1. The first image is an RGB visible-light image.
2. The second image is a TIR thermal infrared image.

The two images are captured at the same time, so their scene state is temporally aligned. However, their spatial alignment may be imperfect: viewpoints, scale, field of view, and local object positions may not match exactly. Do not assume pixel-level correspondence. Use cross-modal semantic, thermal, structural, and contextual clues to reason jointly.

Task requirements:
Analyze both images comprehensively. Use RGB cues and TIR cues. Select the most appropriate answer from the options.

Output requirements: Only output one option letter (A/B/C/D/E). Do not output any other content. Example output: A

Question: {question }

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
