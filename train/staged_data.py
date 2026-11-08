"""Right-padded multimodal batches with exact prompt/answer boundary validation."""

import torch
from PIL import Image


def chat_messages(record, answer):
    prompt = record["messages"][0]["content"]
    pieces = prompt.split("<image>")
    if len(pieces) - 1 != len(record["images"]):
        raise ValueError(f'Image placeholder mismatch: {record["question_id"]}')
    content = []
    for index, text in enumerate(pieces):
        if index:
            content.append({"type": "image"})
        if text:
            content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    if answer:
        messages.append(
            {"role": "assistant", "content": record["messages"][-1]["content"]}
        )
    return messages


class Collator:
    def __init__(self, processor, max_length=8192, max_pixels=1003520):
        self.processor, self.max_length, self.max_pixels = (
            processor,
            max_length,
            max_pixels,
        )
        processor.tokenizer.padding_side = "right"

    def __call__(self, rows):
        images, full, prompts = [], [], []
        for row in rows:
            for path in row["images"]:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    if im.width * im.height > self.max_pixels:
                        scale = (self.max_pixels / (im.width * im.height)) ** 0.5
                        im = im.resize(
                            (
                                max(1, int(im.width * scale)),
                                max(1, int(im.height * scale)),
                            ),
                            Image.Resampling.BICUBIC,
                        )
                    images.append(im)
            full.append(
                self.processor.apply_chat_template(
                    chat_messages(row, True),
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            prompts.append(
                self.processor.apply_chat_template(
                    chat_messages(row, False),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        batch = self.processor(
            text=full, images=images, padding=True, return_tensors="pt"
        )
        prefix = self.processor(
            text=prompts, images=images, padding=True, return_tensors="pt"
        )
        if batch["input_ids"].shape[1] > self.max_length:
            raise ValueError(
                f'No truncation allowed: length={batch["input_ids"].shape[1]}, limit={self.max_length}; IDs={[r["question_id"] for r in rows]}'
            )
        lengths = prefix["attention_mask"].sum(1)
        labels = batch["input_ids"].clone()
        for i, length in enumerate(lengths.tolist()):
            if length < 1 or not torch.equal(
                batch["input_ids"][i, :length], prefix["input_ids"][i, :length]
            ):
                raise ValueError(
                    f'Prompt is not exact token prefix: {rows[i]["question_id"]}'
                )
            labels[i, :length] = -100
        labels[~batch["attention_mask"].bool()] = -100
        if not (labels != -100).any(1).all():
            raise ValueError("Missing supervised answer")
        batch["labels"], batch["prompt_lengths"] = labels, lengths
        batch["aux_kinds"] = [r["aux_kind"] for r in rows]
        batch["targets"] = [
            (
                r.get("target_box")
                if r["aux_kind"] == "box"
                else (
                    r.get("target_count")
                    if r["aux_kind"] == "count"
                    else r.get("target_boxes") if r["aux_kind"] == "od" else None
                )
            )
            for r in rows
        ]
        return dict(batch)
