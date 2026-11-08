"""Preserved task prompt and response parser; model loading is separate."""

from __future__ import annotations
import ast
import json
import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

PSJ_TASK_PROMPTS = {
    "1. Vehicle Heading": 'This is a vehicle heading sub-problem of spatial judgment. Please determine the current heading direction of the target vehicle in the image. The directions used in the options: they are based on the absolute directions of the image. For example, the top of the image corresponds to "up," and the left area of the image corresponds to "left."',
    "2. Person Heading": 'This is a person heading sub-problem of spatial judgment. Please determine the current heading direction of the target person in the image. The directions used in the options: they are based on the absolute directions of the image. For example, the top of the image corresponds to "up," and the left area of the image corresponds to "left."',
    "3. Travel Direction": "This is a vehicle intersection travel direction sub-problem of spatial judgment. Please determine the 'intended' or 'current' travel direction of the target vehicle at the intersection.",
    "4. Relative Position": "This is a relative position judgment sub-problem of spatial judgment. Please determine the relative position of the target object (object B) with respect to the reference object (object A) based on the reference object's (object A) own heading and orientation. Note that the directional terms used for this sub-problem are object-centric rather than image-centric. For example, if Object C is facing downward, then Object D-despite appearing further to the left in the image-is considered to be on the right side of Object C.",
    "5. Height Comparison": "This is a relative height comparison sub-problem of spatial judgment. Please determine which object is located at a higher vertical position (elevation) in the scene.",
}


PSJ_MAIN_PROMPT = (
    "This is a spatial judgment question. Please choose the correct option based on the question. "
    "The bounding box [x1, y1, x2, y2] in the Question, where (x1, y1) and (x2, y2) denote "
    "the coordinates of the top-left and bottom-right corners, respectively. Output only the letter "
    "corresponding to the correct option, e.g., output: A; e.g., output: D. Do not output any "
    "explanatory text or reasoning process. You only need to output one letter."
)


def option_keys(options: Dict[str, Any]) -> List[str]:
    return sorted([str(k).strip().upper() for k in options.keys() if str(k).strip()])


def format_options(options: Dict[str, Any], keys: Iterable[str]) -> str:
    return "\n".join(f"{key }. {options [key ]}" for key in keys if key in options)


def build_psj_prompt(sample: Dict[str, Any]) -> str:
    options = sample["options"]
    sampled_question_type = sample.get("sampled_question_type", "")
    task_prompt = PSJ_TASK_PROMPTS.get(sampled_question_type, "")
    keys = option_keys(options)
    return (
        f"{PSJ_MAIN_PROMPT }\n\n"
        f"{task_prompt }\n\n"
        f"Question: {sample ['question']}\n\n"
        f"Options:\n{format_options (options ,keys )}\n\n"
        "Answer:"
    ).strip()


def build_oc_prompt(sample: Dict[str, Any]) -> str:
    options = sample["options"]
    return f"""
This is an object counting task from a UAV-view image.
Carefully inspect the image, count the target objects described in the question, and choose the option whose number best matches your count.
Output only one option letter: A, B, C, or D. Do not output explanations, numbers, or extra text.

Question: {sample ['question']}

Options:
A. {options ['A']}
B. {options ['B']}
C. {options ['C']}
D. {options ['D']}

Answer:
""".strip()


def build_mmr_oc_prompt(sample: Dict[str, Any]) -> str:
    options = sample["options"]
    return f"""
This is a dual-modal joint counting task from UAV images.
You will receive two spatially aligned images of the same scene:
Image 1 is RGB visible light, and Image 2 is TIR thermal infrared.
If they are shown as one side-by-side composite image, the left half is RGB and the right half is TIR.

Task: answer the counting question by combining both modalities.
Count the target objects specified by the question. Use RGB and TIR together, including targets visible in only one modality.
Do not double-count the same physical object if it appears in both RGB and TIR.
If the question specifies a class or region, count only objects that match that class or region.

Output only one option letter: A, B, C, D, or E. Do not output numbers, explanations, or extra text.
Example output: A

Question: {sample ['question']}

Options:
A. {options ['A']}
B. {options ['B']}
C. {options ['C']}
D. {options ['D']}
E. {options .get ('E','')}

Answer:
""".strip()


def build_mmr_vg_prompt(sample: Dict[str, Any]) -> str:
    options = sample["options"]
    return f"""
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
""".strip()


def build_dmjr_prompt(sample: Dict[str, Any]) -> str:
    question = sample.get("question") or sample.get("question_zh", "")
    options = sample.get("options") or sample.get("options_zh", {})
    return f"""
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
""".strip()


def build_cr_prompt(sample: Dict[str, Any]) -> str:
    opts = sample["options"]
    return (
        "Please analyze the image and answer the following reasoning question. "
        "Please select the most appropriate answer from the given options. "
        "Please strictly output only one option letter: A, B, C, D, or E, and do not provide any explanation.\n\n"
        f"Question: {sample ['question']}\n\n"
        f"Options:\nA. {opts ['A']}\nB. {opts ['B']}\nC. {opts ['C']}\n"
        f"D. {opts ['D']}\nE. {opts ['E']}\n\nAnswer:"
    )


def build_cj_prompt(sample: Dict[str, Any]) -> str:
    scope = str(sample.get("scope_description", "")).strip()
    scope_block = f"\nJudgment scope: {scope }\n" if scope else "\n"
    return f"""
This is a conditional judgment task from a UAV-view image.
Carefully inspect the image and decide whether the condition stated in the question is true.
If the condition is true, answer Yes. If the condition is false, answer No.
Output only one word: Yes or No. Do not output explanations or extra text.
{scope_block }
Question: {sample ['question']}

Answer:
""".strip()


def build_or_prompt(sample: Dict[str, Any]) -> str:
    options = sample["options"]
    return f"""
This is an object recognition task from a UAV-view image.
Carefully inspect the image and identify the target referred to by the question, usually through its normalized bbox coordinates [x1, y1, x2, y2]. x1, y1 are the normalized coordinates of the top-left corner of the target bounding box (range 0-1).x2, y2 are the normalized coordinates of the bottom-right corner of the target bounding box (range 0-1)
Choose the option that best describes the target's visible compound attributes, such as type, structure, color pattern, markings, carried items, or clear image-relative position.
Output only one option letter: A, B, C, or D. Do not output explanations or extra text.
When describing spatial relationships, all directional terms (up, down, left, right) are defined relative to the image itself, meaning the top of the image represents the "up" direction.

Question: {sample ['question']}

Options:
A. {options ['A']}
B. {options ['B']}
C. {options ['C']}
D. {options ['D']}

Answer:
""".strip()


def build_vg_prompt(sample: Dict[str, Any]) -> str:
    options = sample["options"]
    return f"""
This is a visual grounding task. Please analyze the image and answer the following question by selecting the most appropriate option. Output only the option letter (A, B, C, or D).
Unless specified otherwise as relative to the target's orientation, all directional terms (up, down, left, right) refer to the image frame itself-i.e., the top of the image represents "up."
The format of the bounding box in Options: [x1, y1, x2, y2]. x1, y1 are the normalized coordinates of the top-left corner of the target bounding box (range 0-1). x2, y2 are the normalized coordinates of the bottom-right corner of the target bounding box (range 0-1).

Question: {sample ['question']}

Options:
A. {options ['A']}
B. {options ['B']}
C. {options ['C']}
D. {options ['D']}

Answer:
""".strip()


def build_mig_prompt(sample: Dict[str, Any]) -> str:
    options = sample["options"]
    template_box = sample["template_coordinates"][0]
    query_frame = sample.get("query_frame", 3)
    return f"""
This is a UAV multi-image grounding task. You will receive four temporally ordered images from the same video sequence: Image 1 is the template frame, and Images 2-4 are later frames. The target object and the UAV camera viewpoint may both move over time.

Your task is to identify the same target object marked by the template bounding box in Image 1, then answer where that target is located in the query image specified by the question. Do not choose a visually similar distractor; track the same physical object across the image sequence.

The template target in Image 1 is marked by this normalized bounding box:
{template_box }

Bounding box format: [x1, y1, x2, y2]. x1 and y1 are the normalized coordinates of the top-left corner. x2 and y2 are the normalized coordinates of the bottom-right corner. All values are in the range [0, 1].

The options below are candidate bounding boxes in Image {query_frame }. Select the option that best matches the same target object from Image 1.

Question: {sample ["question"]}

Options:
A. {options ["A"]}
B. {options ["B"]}
C. {options ["C"]}
D. {options ["D"]}

Output only one option letter: A, B, C, or D. Do not output explanations, coordinates, or any other text.

Answer:
""".strip()


def build_mvr_vg_prompt(sample: Dict[str, Any]) -> str:
    options = sample["options"]
    return f"""
This is a UAV multi-view target matching multiple-choice task. You will receive two images captured at the same time from different UAV viewpoints.

Image-1 is the query image. The query target is the object inside the red box in Image-1.
Image-2 is the candidate image from another viewpoint of the same scene. Your task is to identify where the same physical target from Image-1 appears in Image-2, and answer with exactly one option letter.

The same target may look different across views: its position, scale, orientation, visible side, and surrounding context can change. Use both object appearance and the global scene layout to match the same physical object.

Important absence rule:
- The query target from Image-1 may actually not exist as a visible answer in Image-2.
- Some options may contain "not exist".
- Choose the "not exist" option only when the query target is not present in Image-2, or when it is present in the scene but is essentially impossible to identify because another object fully occludes it, with about 95% or more of its own area occluded.
- If the target is partially visible, very small, truncated by the image boundary, viewed from another angle, or difficult but still identifiable, do not choose "not exist"; choose the bbox option that best covers that same target in Image-2.
- If "not exist" appears as a distractor while the target is visible in Image-2, choose the correct bbox option instead.

The query target in Image-1 is marked by this normalized bounding box:
{sample ["template_coordinates"][0 ]}

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
""".strip()


def normalize_options(options: Any) -> Dict[str, str]:
    if isinstance(options, dict):
        return {key: str(options.get(key, "")) for key in ["A", "B", "C", "D", "E"]}
    if isinstance(options, list):
        return {
            key: str(options[idx]) if idx < len(options) else ""
            for idx, key in enumerate(["A", "B", "C", "D", "E"])
        }
    return {key: "" for key in ["A", "B", "C", "D", "E"]}


def build_mvr_prompt(sample: Dict[str, Any]) -> str:
    question = str(sample.get("question") or sample.get("question_zh") or "")
    options = normalize_options(sample.get("options") or sample.get("options_zh") or {})
    return f"""
This is a multi-view reasoning multiple-choice question task for UAV imagery. 
You will be provided with two images of the same scene captured simultaneously from different UAV viewpoints, Image-1 and Image-2 (in the order they are input).
The two images are time-aligned but their field of view, scale, viewing angle, visible sides of objects, and crop boundaries may differ. 

Task requirements:
- Answer the question by jointly reasoning over visual clues from both Image-1 and Image-2.
- A single view may be insufficient or misleading; use cross-view evidence to resolve ambiguity.
- Pay attention to view-specific evidence: some clues may be visible only in one image, clearer in one image, occluded, cropped, or seen from a different angle in the other.
- When using spatial relations, using image directions such as top, bottom, left, right, and center.

Select the best answer from A/B/C/D/E.

Output requirements: Only output one option letter (A/B/C/D/E). Do not output explanations or any other text. Example output: A

Question: {question }

Options:
A. {options ["A"]}
B. {options ["B"]}
C. {options ["C"]}
D. {options ["D"]}
E. {options ["E"]}

Answer:
""".strip()


def build_prompt(sample: Dict[str, Any], task: str) -> str:
    if task == "psj":
        return build_psj_prompt(sample)
    if task == "oc":
        return build_oc_prompt(sample)
    if task == "mmr_oc":
        return build_mmr_oc_prompt(sample)
    if task == "mmr_vg":
        return build_mmr_vg_prompt(sample)
    if task == "dmjr":
        return build_dmjr_prompt(sample)
    if task in {"cr", "si"}:
        return build_cr_prompt(sample)
    if task == "cj":
        return build_cj_prompt(sample)
    if task == "or":
        return build_or_prompt(sample)
    if task == "vg":
        return build_vg_prompt(sample)
    if task == "mig":
        return build_mig_prompt(sample)
    if task == "mvr_vg":
        return build_mvr_vg_prompt(sample)
    if task == "mvr":
        return build_mvr_prompt(sample)
    raise ValueError(f"Unsupported task: {task }")
