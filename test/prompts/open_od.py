"""Preserved task prompt and response parser; model loading is separate."""

from __future__ import annotations
import ast
import json
import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple


def normalize_box(raw_box) -> Optional[List[float]]:
    if isinstance(raw_box, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", raw_box)
        if len(nums) != 4:
            return None
        coords = [float(x) for x in nums]
    elif isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
        try:
            coords = [float(x) for x in raw_box]
        except (TypeError, ValueError):
            return None
    else:
        return None

    if any(c > 1.0 for c in coords) and all(0.0 <= c <= 1000.0 for c in coords):
        coords = [c / 1000.0 for c in coords]

    if not all(-0.05 <= c <= 1.05 for c in coords):
        return None

    x1, y1, x2, y2 = coords
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))

    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def parse_boxes_any(value) -> List[List[float]]:
    boxes = []
    if isinstance(value, dict):
        value = value.get("boxes", [])

    if isinstance(value, (list, tuple)):
        for item in value:
            box = normalize_box(item)
            if box:
                boxes.append(box)
    else:
        box = normalize_box(value)
        if box:
            boxes.append(box)
    return boxes


def strip_thinking(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def extract_json_payload(text: str):
    text = strip_thinking(text)
    if not text:
        raise ValueError("empty response")

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        return json.loads(text[obj_start : obj_end + 1])

    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start >= 0 and arr_end > arr_start:
        return json.loads(text[arr_start : arr_end + 1])

    raise ValueError("no json object or array found")


def parse_predicted_boxes(
    response: str,
) -> Tuple[List[List[float]], str, Optional[str]]:
    try:
        payload = extract_json_payload(response)
        boxes = parse_boxes_any(payload)
        if isinstance(payload, dict) and "boxes" not in payload:
            return (
                boxes,
                "invalid_answer_format",
                "JSON object does not contain key 'boxes'",
            )
        return boxes, "tested", None
    except Exception as exc:

        text = strip_thinking(response)
        candidates = re.findall(
            r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
            r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
            text,
        )
        boxes = []
        for item in candidates:
            box = normalize_box([float(x) for x in item])
            if box:
                boxes.append(box)
        if boxes:
            return (
                boxes,
                "invalid_answer_format",
                f"strict JSON parse failed; regex recovered boxes: {exc }",
            )
        return [], "invalid_answer_format", str(exc)


def build_direct_json_prompt(sample: Dict) -> str:
    return f"""
You are solving an object detection task for a UAV image.

Task: find every object that matches the question and output all bounding boxes.

Coordinate rules:
1. Use normalized coordinates in [0, 1].
2. Each box format is [x1, y1, x2, y2].
3. x1, y1 are the top-left corner; x2, y2 are the bottom-right corner.
4. Do not use pixel coordinates.
5. Include every matching visible object. Do not include objects that do not match the question.

Output rules:
1. Output ONLY valid JSON.
2. The JSON schema must be exactly: {{"boxes": [[x1, y1, x2, y2], ...]}}
3. Do not output class names, confidence scores, markdown, or explanations.
4. If there is no matching object, output: {{"boxes": []}}

Question: {sample ["question"]}
""".strip()
