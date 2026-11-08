"""Preserved task prompt and response parser; model loading is separate."""

from __future__ import annotations
import ast
import json
import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple


def build_prompt(sample: Dict) -> str:
    opts = sample["options"]
    return (
        "Please analyze the image and answer the following reasoning question. "
        "Please select the most appropriate answer from the given options. "
        "Please strictly output only one option letter: A, B, C, D, or E, and do not provide any explanation.\n\n"
        f"Question: {sample ['question']}\n\n"
        f"Options:\nA. {opts ['A']}\nB. {opts ['B']}\nC. {opts ['C']}\n"
        f"D. {opts ['D']}\nE. {opts ['E']}\n\nAnswer:"
    )


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

    think_end_pos = upper.find("</THINK>")
    if think_end_pos != -1:
        text_after_think = upper[think_end_pos + len("</THINK>") :]
        for pat in explicit_patterns:
            m = re.search(pat, text_after_think)
            if m:
                return m.group(1), "tested", [m.group(1)]

        candidates_after = re.findall(r"(?<![A-Z])([A-E])(?![A-Z])", text_after_think)
        unique_after = []
        for c in candidates_after:
            if c not in unique_after:
                unique_after.append(c)
        if len(unique_after) == 1:
            return unique_after[0], "tested", unique_after
        if len(unique_after) > 1:
            return None, "ambiguous_answer_format", unique_after
        return None, "invalid_answer_format", []

    think_start_pos = upper.find("<THINK>")
    if think_start_pos != -1:
        return None, "invalid_answer_format", []

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
