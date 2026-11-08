"""Preserved task prompt and response parser; model loading is separate."""

from __future__ import annotations
import ast
import json
import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

ABSENT_ANSWER = "not exist"


def parse_coordinates(response: str) -> Optional[List[float]]:
    text = str(response or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    patterns = [
        r"\[\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]\]",
        r"\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]",
        r"x1\s*=\s*([\d.]+),?\s*y1\s*=\s*([\d.]+),?\s*x2\s*=\s*([\d.]+),?\s*y2\s*=\s*([\d.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            coords = [float(match.group(i)) for i in range(1, 5)]
        except ValueError:
            continue
        if all(0 <= value <= 1 for value in coords):
            x1, y1, x2, y2 = coords
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        if any(value > 1 for value in coords) and all(
            0 <= value <= 1000 for value in coords
        ):
            coords = [value / 1000.0 for value in coords]
            x1, y1, x2, y2 = coords
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return None


def parse_not_exist(response: str) -> bool:
    text = str(response or "").strip().lower()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    normalized = re.sub(r"[^a-z]+", " ", text).strip()
    absent_phrases = {
        "not exist",
        "does not exist",
        "not visible",
        "invisible",
        "absent",
        "no target",
        "not present",
    }
    return normalized in absent_phrases or normalized.startswith("not exist")


def build_direct_prompt(sample: Dict, include_template_bbox: bool = True) -> str:
    bbox_note = ""
    if include_template_bbox:
        bbox_note = f"""
The query target in Image-1 is marked by this normalized bounding box:
{sample ["template_coordinates"][0 ]}
"""

    prompt = f"""
This is a UAV multi-view target matching task. You will receive two images captured at the same time from different UAV viewpoints.

Image-1 is the query image. The target object is marked with a red box in Image-1.
Image-2 is another viewpoint of the same scene. Your task is to locate the same physical target in Image-2.

Important notes:
1. Match the same physical object across viewpoints; do not choose a nearby object or a merely similar-looking distractor.
2. The viewpoint, orientation, scale, and apparent position may change significantly between Image-1 and Image-2.
3. Use scene layout, object appearance, road/building context, and relative positions to infer the cross-view match.
4. If the target from Image-1 is not visible in Image-2, output exactly: {ABSENT_ANSWER }
{bbox_note }

Output format requirements:
- If the target is visible in Image-2, output ONLY one normalized bbox in this exact format: [x1, y1, x2, y2]
- x1, y1 are the normalized top-left coordinates of the target bbox in Image-2.
- x2, y2 are the normalized bottom-right coordinates of the target bbox in Image-2.
- All four numbers must be in [0, 1]. Never output pixel coordinates such as [234, 107, 398, 445].
- If the target is not visible in Image-2, output ONLY: {ABSENT_ANSWER }
- Do not output explanations, option letters, JSON objects, or any extra text.

Correct visible examples:
[0.1231, 0.4526, 0.7849, 0.9031]
[0.0341, 0.3640, 0.1223, 0.5634]

Correct invisible example:
{ABSENT_ANSWER }

Question: {sample ["question"]}

Output:
"""
    return prompt.strip()
