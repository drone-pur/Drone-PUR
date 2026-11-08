"""Preserved task prompt and response parser; model loading is separate."""

from __future__ import annotations
import ast
import json
import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

TASK_PROMPTS = {
    "1. Vehicle Heading": 'This is a vehicle heading sub-problem of spatial judgment. Please determine the current heading direction of the target vehicle in the image. The directions used in the options: they are based on the absolute directions of the image. For example, the top of the image corresponds to "up," and the left area of the image corresponds to "left."',
    "2. Person Heading": 'This is a person heading sub-problem of spatial judgment. Please determine the current heading direction of the target person in the image. The directions used in the options: they are based on the absolute directions of the image. For example, the top of the image corresponds to "up," and the left area of the image corresponds to "left."',
    "3. Travel Direction": "This is a vehicle intersection travel direction sub-problem of spatial judgment. Please determine the 'intended' or 'current' travel direction of the target vehicle at the intersection.",
    "4. Relative Position": "This is a relative position judgment sub-problem of spatial judgment. Please determine the relative position of the target object (object B) with respect to the reference object (object A) based on the reference object's (object A) own heading and orientation. Note that the directional terms used for this sub-problem are object-centric rather than image-centric. For example, if Object C is facing downward, then Object D—despite appearing further to the left in the image—is considered to be on the right side of Object C.",
    "5. Height Comparison": "This is a relative height comparison sub-problem of spatial judgment. Please determine which object is located at a higher vertical position (elevation) in the scene.",
}


MAIN_PROMPT = "This is a spatial judgment question. Please choose the correct option based on the question. The bounding box [x1, y1, x2, y2] in the Question, where (x1, y1) and (x2, y2) denote the coordinates of the top-left and bottom-right corners, respectively. Output only the letter corresponding to the correct option, e.g., output: A; e.g., output: D. Do not output any explanatory text or reasoning process. You only need to output one letter."


def build_prompt(sample: Dict) -> str:

    options = sample["options"]
    sampled_question_type = sample.get("sampled_question_type", "")
    task_prompt = TASK_PROMPTS.get(sampled_question_type, "")

    prompt = f"{MAIN_PROMPT }\n\n{task_prompt }\n\nQuestion: {sample ['question']}\n\nOptions:\n"
    option_keys = sorted(options.keys())
    for key in option_keys:
        prompt += f"{key }. {options [key ]}\n"
    prompt += "\nAnswer:"
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
