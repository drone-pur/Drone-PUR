"""Preserved task prompt and response parser; model loading is separate."""

from __future__ import annotations
import ast
import json
import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple


def build_dual_direct_prompt(sample: Dict) -> str:

    prompt = f"""
This is a dual-modal joint grounding task. You will receive two spatially aligned images:
The first is an RGB visible light image, and the second is a TIR thermal infrared image. Both images are captured at the same time point and are roughly aligned in spatial position.
Task requirements: Combine information from both modalities to locate the target object described in the question. Note that some targets cannot be uniquely identified by a single modality alone; you need to comprehensively utilize information from both modalities.
Unless specified otherwise as relative to the target's orientation, all directional terms (up, down, left, right) refer to the image frame itself—i.e., the top of the image represents "up."

=== OUTPUT FORMAT REQUIREMENTS ===
1. Output must be in strict list format containing exactly four normalized coordinate numbers
2. Format: [x1, y1, x2, y2]
3. x1, y1 are the normalized coordinates of the top-left corner of the target bounding box (range 0-1)
4. x2, y2 are the normalized coordinates of the bottom-right corner of the target bounding box (range 0-1)
5. All coordinate values must be between 0.0 and 1.0
6. Output ONLY the coordinate list, without any additional text or explanation
7. NEVER use actual pixel values in the image as the output answer. For example, [234,107,398,445] is incorrect. The all four coordinate values must be normalized between 0 and 1.

=== CORRECT OUTPUT EXAMPLES ===
Example 1: [0.1231, 0.4526, 0.7849, 0.9031]
Example 2: [0.6489, 0.2453, 0.6873, 0.3011]
Example 3: [0.0341, 0.3640, 0.1223, 0.5634]

=== INCORRECT OUTPUT EXAMPLES (DO NOT DO THIS) ===
Incorrect 1: The answer is [0.1231, 0.4526, 0.7849, 0.9031]
Incorrect 2: {{"bbox": [0.123, 0.456, 0.789, 0.901]}}
Incorrect 3: Top-left: 0.123, 0.456 Bottom-right: 0.789, 0.901
Incorrect 4: x1=0.123 y1=0.456 x2=0.789 y2=0.901

Question: {sample ['question']}

Output:
"""
    return prompt.strip()


def parse_coordinates(response: str) -> List[float]:

    patterns = [
        r"\[\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]\]",
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
                elif any(c > 1 for c in coords) and all(c <= 1000 for c in coords):
                    coords = [c / 1000.0 for c in coords]
                    x1, y1, x2, y2 = coords
                    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            except ValueError:
                continue

    return None
