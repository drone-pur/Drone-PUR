import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


CONFIG = {
    "head_width": 16,
    "num_queries": 4,
    "od_width": 16,
    "od_layers": 1,
    "train_mergers": True,
    "expected_deepstack_mergers": 3,
    "merger_lr_ratio": 0.1,
    "head_lr": 1e-4,
    "loss_weights": {"box": 0.1, "count": 0.01, "od": 0.1},
    "stages": {stage: {"aux": stage != "s3"} for stage in ("s1", "s2", "s3")},
}


@unittest.skipUnless(
    importlib.util.find_spec("torch"),
    "Install the training dependencies for tensor tests",
)
class TrainingTests(unittest.TestCase):
    def setUp(self):
        import torch

        torch.set_num_threads(1)

    def test_worker_cli_defaults_to_accumulation_four(self):
        import argparse
        from unittest.mock import patch
        from drone_pur.training.train_stage import main

        class ParserInspected(Exception):
            pass

        def inspect_parser(parser):
            self.assertEqual(parser.get_default("grad_acc"), 4)
            raise ParserInspected

        with patch.object(argparse.ArgumentParser, "parse_args", inspect_parser):
            with self.assertRaises(ParserInspected):
                main()

    def test_ce_is_per_sample_answer_only(self):
        import torch
        from drone_pur.training.staged_model import answer_ce

        head = torch.nn.Linear(5, 7)
        hidden = torch.randn(2, 6, 5, requires_grad=True)
        labels = torch.tensor([[-100, -100, 2, 3, -100, -100], [-100, 1, 2, 3, 4, 5]])
        actual = answer_ce(hidden, labels, head, chunk_size=2)
        expected = torch.stack(
            [
                torch.nn.functional.cross_entropy(
                    head(hidden[i, :-1]).float(), labels[i, 1:], ignore_index=-100
                )
                for i in range(2)
            ]
        )
        torch.testing.assert_close(actual, expected)
        actual.mean().backward()
        self.assertIsNotNone(hidden.grad)

    def test_sampler_has_no_duplicate_tail_and_lr_keeps_planned_duration(self):
        from drone_pur.training.train_stage import window_assignments, lr_factor

        seen = []
        for offset in (0, 32):
            assignments, count = window_assignments(list(range(37)), offset, 8, 4)
            real = [v for micro in assignments for v in micro if v is not None]
            self.assertEqual(count, len(real))
            seen.extend(real)
            self.assertAlmostEqual(
                sum((v + 1) * 8 / count for v in real) / 8,
                sum(v + 1 for v in real) / count,
            )
        self.assertEqual(seen, list(range(37)))
        self.assertGreater(lr_factor(100, 200, 10), lr_factor(100, 100, 5))

    def test_default_s3_cosine_schedule_completes_after_one_epoch(self):
        import math
        from drone_pur.training.train_stage import lr_factor

        config = json.loads(
            (
                Path(__file__).resolve().parents[2] / "train/configs/uav_pur.json"
            ).read_text()
        )
        per_epoch = math.ceil(141907 / 32)
        total = per_epoch * config["stages"]["s3"]["epochs"]
        warmup = math.ceil(total * 0.05)
        self.assertEqual(total, per_epoch)
        self.assertEqual(total, 4435)
        self.assertAlmostEqual(lr_factor(total, total, warmup), 0.0)
        self.assertGreater(
            lr_factor(total, total * 2, math.ceil(total * 2 * 0.05)), 0.0
        )

    def test_safe_optimizer_rng_roundtrip(self):
        import torch
        from drone_pur.training.checkpoint_io import save_tree, load_tree

        net = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(net.parameters(), lr=0.01)
        net(torch.ones(1, 2)).sum().backward()
        opt.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state"
            save_tree(
                path,
                {
                    "optimizer": opt.state_dict(),
                    "rng": torch.get_rng_state(),
                    "cursor": (1, None),
                },
            )
            loaded = load_tree(path)
            self.assertEqual(loaded["cursor"], (1, None))
            torch.testing.assert_close(loaded["rng"], torch.get_rng_state())
            other = torch.optim.AdamW(net.parameters(), lr=0.1)
            other.load_state_dict(loaded["optimizer"])
            self.assertEqual(other.param_groups[0]["lr"], 0.01)

    def test_auxiliary_losses_have_gradients_and_no_answer_leakage(self):
        import torch
        from torch import nn
        from drone_pur.training.staged_model import StagedModel

        class Core(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(32, 16)
                self.linear = nn.Linear(16, 16)

            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(
                    last_hidden_state=self.linear(self.embedding(input_ids).cumsum(1))
                )

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    text_config=SimpleNamespace(hidden_size=16)
                )
                self.model, self.lm_head = Core(), nn.Linear(16, 32)

            def get_base_model(self):
                return self

        for stage in ("s1", "s2"):
            model = StagedModel(Backbone(), CONFIG, stage).eval()
            for kind, target, head in [
                ("box", [0.1, 0.1, 0.6, 0.7], model.box_head),
                ("count", 4, model.count_head),
                ("od", [[0.2, 0.3, 0.6, 0.8]], model.od_head),
            ]:
                with self.subTest(stage=stage, head=kind):
                    self.assertTrue(all(p.requires_grad for p in head.parameters()))
                    model.zero_grad()
                    inputs = dict(
                        input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
                        attention_mask=torch.ones(1, 5),
                        labels=torch.tensor([[-100, -100, -100, 4, 5]]),
                        prompt_lengths=torch.tensor([3]),
                        aux_kinds=[kind],
                        targets=[target],
                    )
                    before = model(**inputs)
                    after = copy.deepcopy(inputs)
                    after["input_ids"][0, 3:] = torch.tensor([9, 10])
                    torch.testing.assert_close(
                        before["metrics"][2:], model(**after)["metrics"][2:]
                    )
                    before["loss"].backward()
                    self.assertTrue(
                        any(
                            p.grad is not None and p.grad.abs().sum() > 0
                            for p in head.parameters()
                        )
                    )
        s3 = StagedModel(Backbone(), CONFIG, "s3")
        self.assertFalse(hasattr(s3, "box_head"))
        self.assertFalse(hasattr(s3, "count_head"))
        self.assertFalse(hasattr(s3, "od_head"))

    def test_tiny_qwen_stage_transitions_and_inference_adapter(self):
        import torch
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        from peft import PeftModel
        from drone_pur.training.staged_model import (
            build_model,
            ddp_adapter_load_compat,
            is_merger_parameter,
            optimizer_groups,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = Qwen3VLConfig(
                text_config=dict(
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=4,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=8,
                    vocab_size=40,
                    rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
                ),
                vision_config=dict(
                    depth=3,
                    hidden_size=32,
                    intermediate_size=64,
                    num_heads=4,
                    out_hidden_size=32,
                    patch_size=2,
                    temporal_patch_size=2,
                    spatial_merge_size=2,
                    num_position_embeddings=16,
                    deepstack_visual_indexes=[0, 1, 2],
                ),
                image_token_id=29,
                video_token_id=30,
                vision_start_token_id=27,
                vision_end_token_id=28,
            )
            Qwen3VLForConditionalGeneration(config).save_pretrained(root / "base")
            model = build_model(str(root / "base"), CONFIG, "s1", dtype=torch.float32)
            inputs = dict(
                input_ids=torch.tensor([[1, 27, 29, 28, 3, 4, 5]]),
                attention_mask=torch.ones(1, 7, dtype=torch.long),
                pixel_values=torch.randn(4, 24),
                image_grid_thw=torch.tensor([[1, 2, 2]]),
                labels=torch.tensor([[-100, -100, -100, -100, -100, 4, 5]]),
                prompt_lengths=torch.tensor([5]),
                aux_kinds=["box"],
                targets=[[0.1, 0.1, 0.7, 0.8]],
            )
            model.train()
            model(**inputs)["loss"].backward()
            for module in (
                ".visual.merger.",
                ".visual.deepstack_merger_list.0.",
                ".visual.deepstack_merger_list.1.",
                ".visual.deepstack_merger_list.2.",
            ):
                params = [
                    p
                    for name, p in model.backbone.named_parameters()
                    if module in name and p.requires_grad
                ]
                self.assertTrue(params)
                self.assertTrue(all(p.grad is not None for p in params))
                self.assertTrue(any(p.grad.abs().sum() > 0 for p in params))
            self.assertFalse(
                any(
                    p.requires_grad
                    for name, p in model.backbone.named_parameters()
                    if "lm_head" in name
                )
            )
            groups = optimizer_groups(model, CONFIG, {"lr": 2e-5})
            for group in groups:
                kind = group["group_name"].split("_")[0]
                self.assertAlmostEqual(
                    group["lr"], {"lora": 2e-5, "merger": 2e-6, "head": 1e-4}[kind]
                )
            optimizer = torch.optim.AdamW(groups)
            optimizer.step()
            model.eval()
            model.save_weights(root / "s1")
            (root / "s1/base_config.json").write_text(
                json.dumps(model.backbone.get_base_model().config.to_dict())
            )
            s2 = build_model(
                str(root / "base"), CONFIG, "s2", root / "s1", dtype=torch.float32
            ).eval()
            torch.testing.assert_close(model(**inputs)["loss"], s2(**inputs)["loss"])
            self.assertTrue(all(p.requires_grad for p in s2.od_head.parameters()))
            od_inputs = dict(inputs, aux_kinds=["od"], targets=[[[0.1, 0.1, 0.7, 0.8]]])
            od_output = s2(**od_inputs)
            self.assertEqual(od_output["metrics"][8].item(), 1)
            od_output["loss"].backward()
            self.assertTrue(
                any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in s2.od_head.parameters()
                )
            )
            s2.save_weights(root / "s2")
            (root / "s2/base_config.json").write_text(
                json.dumps(s2.backbone.get_base_model().config.to_dict())
            )
            s3 = build_model(
                str(root / "base"), CONFIG, "s3", root / "s2", dtype=torch.float32
            ).eval()
            self.assertFalse(hasattr(s3, "box_head"))
            s3(**inputs)["loss"].backward()
            for name, param in s3.backbone.named_parameters():
                if param.requires_grad and is_merger_parameter(name):
                    self.assertIsNotNone(param.grad)
            with ddp_adapter_load_compat():
                plain = PeftModel.from_pretrained(
                    Qwen3VLForConditionalGeneration.from_pretrained(root / "base"),
                    root / "s2",
                ).eval()
            lm_inputs = {
                key: val
                for key, val in inputs.items()
                if key
                in {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
            }
            with torch.no_grad():
                torch.testing.assert_close(
                    s2.backbone(**lm_inputs, use_cache=False).logits,
                    plain(**lm_inputs, use_cache=False).logits,
                )


if __name__ == "__main__":
    unittest.main()
