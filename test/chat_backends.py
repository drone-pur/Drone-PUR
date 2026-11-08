"""Optional remote-code VLM adapters with explicit ordered multi-image input."""


class ChatBackend:
    def __init__(self, args):
        import torch
        from transformers import (
            AutoModel,
            AutoModelForCausalLM,
            AutoProcessor,
            AutoTokenizer,
        )

        if not args.trust_remote_code:
            raise ValueError(
                "This backend executes upstream Python code; review it and pass --trust-remote-code explicitly"
            )
        self.args, self.torch = args, torch
        self.task = None
        revision = getattr(args, "resolved_revision", args.revision)
        self.dtype = getattr(torch, args.dtype)
        loader = AutoModelForCausalLM if args.backend == "minicpm45" else AutoModel
        self.model = loader.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            device_map=args.device_map,
            trust_remote_code=True,
            revision=revision,
        ).eval()
        self.device = self.model.get_input_embeddings().weight.device
        if args.backend == "llava-ov":
            self.processor = AutoProcessor.from_pretrained(
                args.model, trust_remote_code=True, revision=revision
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.model, trust_remote_code=True, revision=revision
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def set_context(self, task, mode):
        self.task = task

    def internvl_image(self, image):
        """Preserve the existing six-tile grid search, thumbnail and ImageNet transform."""
        import numpy as np
        from PIL import Image

        torch, limit = self.torch, self.args.chat_max_tiles
        w, h = image.size
        rows = cols = 1
        best = float("inf")
        for r in range(1, limit + 1):
            for c in range(1, limit + 1):
                if r * c <= limit and abs(c / r - w / max(h, 1)) < best:
                    best, rows, cols = abs(c / r - w / max(h, 1)), r, c
        tile_w, tile_h = max(1, w // cols), max(1, h // rows)
        tiles = [
            image.crop(
                (
                    c * tile_w,
                    r * tile_h,
                    w if c == cols - 1 else (c + 1) * tile_w,
                    h if r == rows - 1 else (r + 1) * tile_h,
                )
            )
            for r in range(rows)
            for c in range(cols)
        ]
        tiles.append(image)
        tensors = []
        mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        for tile in tiles[: limit + 1]:
            resized = tile.resize(
                (self.args.chat_image_size, self.args.chat_image_size),
                Image.Resampling.BICUBIC,
            )
            tensor = torch.from_numpy(
                (np.asarray(resized).astype("float32") / 255).transpose(2, 0, 1)
            )
            tensors.append((tensor - mean) / std)
        return torch.stack(tensors).to(device=self.device, dtype=self.dtype)

    def infer(self, paths, prompt, max_new_tokens):
        from PIL import Image

        torch = self.torch
        images = []
        try:
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            info = {"image_sizes": [list(im.size) for im in images]}
            if self.args.backend == "llava-ov" and self.task in {
                "MIG",
                "MVR-VG",
                "MVR-R",
            }:
                # These tasks historically shared the 1008-pixel LLaVA wrapper.
                for index, image in enumerate(images):
                    if max(image.size) > 1008:
                        scale = 1008 / max(image.size)
                        images[index] = image.resize(
                            (
                                max(1, round(image.width * scale)),
                                max(1, round(image.height * scale)),
                            ),
                            Image.Resampling.BICUBIC,
                        )
                        image.close()
                info["processed_image_sizes"] = [list(im.size) for im in images]
            with torch.inference_mode():
                if self.args.backend == "internvl":
                    pixels = [self.internvl_image(im) for im in images]
                    counts = [len(p) for p in pixels]
                    info["patch_counts"] = counts
                    question = "<image>\n" * len(images) + prompt
                    response = self.model.chat(
                        self.tokenizer,
                        torch.cat(pixels) if pixels else None,
                        question,
                        {
                            "max_new_tokens": max_new_tokens,
                            "do_sample": False,
                            "pad_token_id": self.tokenizer.pad_token_id,
                            "eos_token_id": self.tokenizer.eos_token_id,
                        },
                        **({"num_patches_list": counts} if len(counts) > 1 else {}),
                    )
                elif self.args.backend in {"minicpm4", "minicpm45"}:
                    kwargs = {
                        "msgs": [{"role": "user", "content": [*images, prompt]}],
                        "tokenizer": self.tokenizer,
                        "generation_config": {
                            "max_new_tokens": max_new_tokens,
                            "do_sample": False,
                            "temperature": 0.0,
                        },
                    }
                    if self.args.backend == "minicpm4" and images:
                        kwargs["image"] = images[0] if len(images) == 1 else images
                    response = self.model.chat(**kwargs)
                else:
                    # Keep the original LLaVA text-first content order.
                    content = [{"type": "text", "text": prompt}] + [
                        {"type": "image"} for _ in images
                    ]
                    text = self.processor.apply_chat_template(
                        [{"role": "user", "content": content}],
                        add_generation_prompt=True,
                    )
                    kwargs = {"text": text, "return_tensors": "pt"}
                    if images:
                        kwargs["images"] = images
                    inputs = self.processor(**kwargs).to(self.device)
                    for key, value in inputs.items():
                        if torch.is_floating_point(value):
                            inputs[key] = value.to(self.dtype)
                    ids = self.model.generate(
                        **inputs, max_new_tokens=max_new_tokens, do_sample=False
                    )
                    response = self.processor.batch_decode(
                        ids[:, inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]
            return str(response).strip(), info
        finally:
            for image in images:
                image.close()
