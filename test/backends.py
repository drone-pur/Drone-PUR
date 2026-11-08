"""Independent local Qwen and OpenAI-compatible multimodal API backends."""

import base64
import json
import mimetypes
import os
from pathlib import Path
import time
from urllib.parse import urlsplit


def set_max_pixels(processor, value):
    if value is None:
        return
    image_processor = processor.image_processor
    changed = False
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = value
        changed = True
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict) and "longest_edge" in size:
        size["longest_edge"] = value
        changed = True
    elif hasattr(size, "longest_edge"):
        size.longest_edge = value
        changed = True
    if not changed:
        raise ValueError(
            "This processor cannot apply --max-pixels; omit it to use model defaults"
        )


class QwenBackend:
    def __init__(self, args):
        import torch
        import transformers
        from transformers import AutoConfig, AutoProcessor

        self.args = args
        base = args.base_model or args.model
        revision = getattr(args, "resolved_revision", args.revision)
        if (Path(args.model) / "adapter_config.json").is_file() and not args.base_model:
            raise ValueError(
                "An adapter requires an explicit --base-model; saved server paths are not portable"
            )
        config = AutoConfig.from_pretrained(base, revision=revision)
        classes = {
            "qwen3_vl": "Qwen3VLForConditionalGeneration",
            "qwen3_5": "Qwen3_5ForConditionalGeneration",
            "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
            "qwen2_vl": "Qwen2VLForConditionalGeneration",
        }
        cls = getattr(transformers, classes.get(config.model_type, ""), None)
        if cls is None:
            raise ImportError(
                f"Unsupported architecture {config.model_type!r} in Transformers {transformers.__version__}; use a compatible separate environment"
            )
        self.model = cls.from_pretrained(
            base,
            torch_dtype=getattr(torch, args.dtype),
            device_map=args.device_map,
            revision=revision,
            attn_implementation=args.attn_implementation,
        ).eval()
        if args.base_model:
            from peft import PeftModel
            from .training.staged_model import ddp_adapter_load_compat

            with ddp_adapter_load_compat():
                self.model = PeftModel.from_pretrained(
                    self.model,
                    args.model,
                    is_trainable=False,
                    revision=getattr(args, "adapter_revision", args.revision),
                ).eval()
        processor_path = (
            args.model
            if args.base_model
            and (Path(args.model) / "preprocessor_config.json").is_file()
            else base
        )
        self.processor = AutoProcessor.from_pretrained(
            processor_path, revision=revision
        )
        self.model_type = config.model_type
        set_max_pixels(self.processor, args.max_pixels)

    def infer(self, paths, prompt, max_new_tokens):
        import torch
        from PIL import Image

        images = []
        try:
            for path in paths:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
            content = [{"type": "image"} for _ in images] + [
                {"type": "text", "text": prompt}
            ]
            extra = {"enable_thinking": False} if self.model_type == "qwen3_5" else {}
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                **extra,
            )
            kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
            if images:
                kwargs["images"] = images
            inputs = self.processor(**kwargs)
            info = {
                "image_sizes": [list(im.size) for im in images],
                "input_tokens": inputs["input_ids"].shape[1],
            }
            if "image_grid_thw" in inputs:
                info["image_grid_thw"] = inputs["image_grid_thw"].tolist()
            device = self.model.get_input_embeddings().weight.device
            inputs = inputs.to(device)
            with torch.inference_mode():
                ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
            ids = ids[:, inputs["input_ids"].shape[1] :]
            return (
                self.processor.batch_decode(
                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip(),
                info,
            )
        finally:
            for image in images:
                image.close()


class APIError(RuntimeError):
    pass


def api_url(value):
    parsed = urlsplit(value)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError("--api-base-url must be an HTTP(S) endpoint")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "API endpoint URLs must not contain credentials, queries or fragments"
        )
    return value.rstrip("/")


def api_parameters(value):
    extra = json.loads(value) if value else {}
    if (
        not isinstance(extra, dict)
        or {"model", "messages", "stream", "n"} & extra.keys()
    ):
        raise ValueError(
            "--api-params must be a JSON object without model/messages/stream/n"
        )

    def check(obj):
        if isinstance(obj, dict):
            for key, item in obj.items():
                if any(
                    word in str(key).lower()
                    for word in (
                        "api_key",
                        "apikey",
                        "authorization",
                        "password",
                        "secret",
                        "access_token",
                    )
                ):
                    raise ValueError(
                        "Credentials must be provided through --api-key-env, not --api-params"
                    )
                check(item)
        elif isinstance(obj, list):
            for item in obj:
                check(item)

    check(extra)
    return extra


class APIBackend:
    def __init__(self, args):
        import requests

        self.args = args
        self.key = os.environ.get(args.api_key_env, "").strip()
        if not self.key:
            raise ValueError(
                f"Set {args.api_key_env} in the terminal; API keys are never stored in code or manifests"
            )
        self.url = api_url(args.api_base_url)
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        self.session = requests.Session()
        self.extra = api_parameters(args.api_params)

    def encode_image(self, path):
        from PIL import Image
        import io

        with Image.open(path) as image:
            original = list(image.size)
            if self.args.api_max_side and max(image.size) > self.args.api_max_side:
                image = image.convert("RGB")
                scale = self.args.api_max_side / max(image.size)
                image = image.resize(
                    (
                        max(1, round(image.width * scale)),
                        max(1, round(image.height * scale)),
                    ),
                    Image.Resampling.BICUBIC,
                )
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=95)
                payload, media, sent = buffer.getvalue(), "image/jpeg", list(image.size)
            else:
                payload, media, sent = (
                    Path(path).read_bytes(),
                    mimetypes.guess_type(path)[0],
                    original,
                )
        if media not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
            raise ValueError(
                f"Unsupported API image media type: {media}; convert explicitly before evaluation"
            )
        return "data:" + media + ";base64," + base64.b64encode(payload).decode(), {
            "original": original,
            "sent": sent,
        }

    def infer(self, paths, prompt, max_new_tokens):
        import requests

        content, sizes = [], []
        for path in paths:
            data, size = self.encode_image(path)
            sizes.append(size)
            content.append({"type": "image_url", "image_url": {"url": data}})
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": self.args.model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "temperature": 0,
            "max_tokens": max_new_tokens,
            **self.extra,
        }
        error = "Unknown API error"
        for attempt in range(self.args.api_retries + 1):
            retry = True
            try:
                response = self.session.post(
                    self.url,
                    headers={
                        "Authorization": "Bearer " + self.key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.args.api_timeout,
                )
                if response.status_code >= 400:
                    error = f"HTTP {response.status_code}: {response.text[:800]}"
                    retry = response.status_code in {408, 429, 500, 502, 503, 504}
                    if not retry:
                        raise APIError(error.replace(self.key, "[REDACTED]"))
                else:
                    obj = response.json()
                    message = obj["choices"][0]["message"]
                    text = message.get("content")
                    if isinstance(text, list):
                        text = "".join(
                            item.get("text", "")
                            for item in text
                            if isinstance(item, dict)
                        )
                    # Reasoning text is not a final answer and must not be parsed.
                    if not isinstance(text, str):
                        text = ""
                    return text, {
                        "image_sizes": sizes,
                        "response_model": obj.get("model"),
                        "response_id": obj.get("id"),
                        "usage": obj.get("usage"),
                        "finish_reason": obj["choices"][0].get("finish_reason"),
                    }
            except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            if attempt < self.args.api_retries and retry:
                time.sleep(min(30, 2**attempt))
        raise APIError(error.replace(self.key, "[REDACTED]"))


def create_backend(args):
    if args.backend == "qwen":
        return QwenBackend(args)
    if args.backend == "api":
        return APIBackend(args)
    from .chat_backends import ChatBackend

    return ChatBackend(args)
