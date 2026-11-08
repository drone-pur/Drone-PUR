"""Preserved task prompt and response parser; model loading is separate."""

from __future__ import annotations
import ast
import json
import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple


def build_prompt(sample: Dict) -> str:

    template_coords = sample["template_coordinates"][0]
    question = sample["question"]
    query_frame = sample.get("query_frame", 3)

    prompt = f"""
This is a UAV multi-image grounding task. You will receive four temporally ordered images from the same video sequence:
Image 1 is the template frame, and Images 2-4 are later frames. The target object and the UAV camera viewpoint may both move over time.
The four input images are Image 1, Image 2, Image 3, and Image 4 in exactly that order. Image {query_frame } is the query frame for this question.

Your task has two steps:
1. In Image 1, identify the target object marked by the red/template bounding box.
2. Track that same physical object through the later frames and output its bounding box in the query image.

Important:
- The red box is drawn only on Image 1 to identify the physical target. Use the object inside it as the visual reference; the red box position is not a prediction.
- Locate that same physical target again in Image {query_frame }. The query image has no red box, so find the object itself and use Image {query_frame } as the coordinate frame.
- The answer must be the target's location in Image {query_frame }, not its location in Image 1.
- Do NOT copy or repeat the template bbox from Image 1 as the final answer.
- Do NOT output the location of the red box itself.
- Do NOT choose a visually similar distractor; locate the same tracked object.
- If the object has moved, the output coordinates should reflect its new position in Image {query_frame }.

This template bbox is NOT the answer. Use it only to identify which object to track.

The query image is Image {query_frame }.

Bounding box format: [x1, y1, x2, y2]
- x1 and y1 are the normalized coordinates of the top-left corner.
- x2 and y2 are the normalized coordinates of the bottom-right corner.
- All four values must be between 0.0 and 1.0.
- Never output pixel coordinates such as [234, 107, 398, 445].

=== OUTPUT FORMAT REQUIREMENTS ===
1. Output exactly one coordinate list for the target in the query image.
2. The output must contain exactly four normalized numbers.
3. Use this exact format: [x1, y1, x2, y2]
4. Output only the coordinate list. Do not output explanations, labels, JSON, markdown, or any other text.

Output format example (illustrative numbers only):
[0.1111, 0.2222, 0.3333, 0.4444]
These example numbers do not describe the target in any input image. Determine the actual coordinates in Image {query_frame }; do not copy the example.

Original question: {question }
Interpreted task: output the normalized bounding box of the same target object in Image {query_frame }.
Before answering, check that your box encloses the object in Image {query_frame }, using coordinates relative to that image. Do not use the red box position from Image 1.

Output:
"""

    return prompt.strip()


def parse_coordinates(response: str) -> Optional[List[float]]:

    patterns = [
        r"\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]",
        r"(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)",
        r"x1=([\d.]+),?\s*y1=([\d.]+),?\s*x2=([\d.]+),?\s*y2=([\d.]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            try:
                coords = [float(match.group(i)) for i in range(1, 5)]
                if all(0 <= c <= 1 for c in coords):
                    x1, y1, x2, y2 = coords
                    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            except ValueError:
                continue

    return None
