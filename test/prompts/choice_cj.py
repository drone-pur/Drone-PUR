"""Preserved task prompt and response parser; model loading is separate."""

from __future__ import annotations
import ast
import json
import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple


def build_cj_prompt(sample: Dict) -> str:

    scope = str(sample.get("scope_description", "")).strip()
    scope_block = f"\nJudgment scope: {scope }\n" if scope else "\n"
    prompt = f"""
This is a conditional judgment task from a UAV-view image.
Carefully inspect the image and decide whether the condition stated in the question is true.
If the condition is true, answer Yes. If the condition is false, answer No.
Output only one word: Yes or No. Do not output explanations or extra text.
{scope_block }
Question: {sample ['question']}

Answer:
"""
    return prompt.strip()


def normalize_response_text(response: str) -> str:
    text = str(response or "").strip()
    full_to_half = str.maketrans(
        {
            "Ｙ": "Y",
            "ｅ": "e",
            "ｓ": "s",
            "Ｎ": "N",
            "ｏ": "o",
            "ｙ": "y",
            "Ｅ": "E",
            "Ｓ": "S",
            "ｎ": "n",
            "Ｏ": "O",
        }
    )
    return text.translate(full_to_half)


def normalize_yes_no(value: str) -> Optional[str]:
    text = normalize_response_text(value)
    if not text:
        return None
    compact = re.sub(r"[\s\.。:：,，;；!！?？\"'`]+", "", text).lower()
    yes_values = {"yes", "y", "true", "是", "对", "正确", "成立", "有", "存在"}
    no_values = {
        "no",
        "n",
        "false",
        "否",
        "不",
        "错",
        "错误",
        "不成立",
        "没有",
        "不存在",
    }
    if compact in yes_values:
        return "Yes"
    if compact in no_values:
        return "No"
    return None


def parse_answer(response: str) -> Tuple[Optional[str], str, List[str]]:

    text = normalize_response_text(response)
    if not text:
        return None, "invalid_answer_format", []

    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
        if not text:
            return None, "invalid_answer_format", []

    explicit_patterns = [
        r"(?:答案|答|回答|结论|Answer|ANSWER|Result|RESULT)\s*(?:是|为|:|：|=)?\s*[\(\[]?\s*(yes|no|true|false|是|否|成立|不成立|正确|错误)\s*[\)\]\.。:：、,，\s]?",
        r"^\s*[\(\[]?\s*(yes|no|true|false|是|否|成立|不成立|正确|错误)\s*[\)\]\.。:：、,，\s]*$",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            normalized = normalize_yes_no(match.group(1))
            if normalized:
                return normalized, "tested", [normalized]

    candidates = []
    token_patterns = [
        (r"(?<![A-Za-z])yes(?![A-Za-z])", "Yes"),
        (r"(?<![A-Za-z])no(?![A-Za-z])", "No"),
        (r"(?<![A-Za-z])true(?![A-Za-z])", "Yes"),
        (r"(?<![A-Za-z])false(?![A-Za-z])", "No"),
        (r"不成立", "No"),
        (r"(?<!不)成立", "Yes"),
        (r"错误", "No"),
        (r"(?<!不)正确", "Yes"),
        (r"否", "No"),
        (r"是", "Yes"),
    ]
    for pattern, value in token_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            candidates.append(value)

    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)

    if len(unique) == 1:
        return unique[0], "tested", unique
    if len(unique) > 1:
        return None, "ambiguous_answer_format", unique
    return None, "invalid_answer_format", []
