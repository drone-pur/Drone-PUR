import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from drone_pur.data import sha256, write_json

TRAINING_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers", "peft", "safetensors")
)
if TRAINING_AVAILABLE:
    from drone_pur.training.export import (
        MERGERS,
        check_loaded_adapter,
        check_release_report,
        check_merged_lora,
        compare_smoke,
        expected_lora_hashes,
        merger_hashes,
        portable_metadata,
        run_smoke,
        validate_release,
    )


@unittest.skipUnless(
    TRAINING_AVAILABLE, "Install training dependencies for export checks"
)
class ExportTests(unittest.TestCase):
    def test_metadata_uses_public_model_id(self):
        value = {
            "_name_or_path": "/private/base",
            "text_config": {"name_or_path": "/private/tokenizer", "hidden_size": 32},
        }
        result = portable_metadata(value, "example/model")
        self.assertEqual(result["_name_or_path"], "example/model")
        self.assertEqual(result["text_config"]["name_or_path"], "example/model")
        self.assertEqual(result["text_config"]["hidden_size"], 32)
        self.assertEqual(value["_name_or_path"], "/private/base")

    def make_release(self, folder):
        import torch
        from safetensors.torch import save_file

        folder.mkdir()
        for name in ("README.md", "LICENSE", "NOTICE"):
            (folder / name).write_text("Example release\n")
        for name in ("preprocessor_config.json", "tokenizer_config.json"):
            write_json(folder / name, {})
        write_json(folder / "config.json", {"model_type": "qwen3_vl"})
        save_file(
            {name + ".weight": torch.ones(2, 2) for name in MERGERS},
            str(folder / "model.safetensors"),
        )

    def test_release_rejects_training_files_and_changed_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "release"
            self.make_release(folder)
            self.assertEqual(validate_release(folder)["tensor_count"], 4)
            write_json(
                root / "report.json",
                {
                    "passed": True,
                    "output": str(folder),
                    "files_sha256": {p.name: sha256(p) for p in folder.iterdir()},
                },
            )
            check_release_report(folder, root / "report.json")
            (folder / "training_state.json").write_text("{}")
            with self.assertRaises(ValueError):
                validate_release(folder)
            (folder / "training_state.json").unlink()
            (folder / "README.md").write_text("Changed")
            with self.assertRaises(ValueError):
                check_release_report(folder, root / "report.json")

    def test_smoke_tolerances_and_exact_reload(self):
        import torch

        before = {
            "text": {
                "logits": torch.tensor([[1.0, 2.0]]),
                "tokens": torch.tensor([[1]]),
            }
        }
        rounded = {
            "text": {
                "logits": torch.tensor([[1.01, 2.01]]),
                "tokens": torch.tensor([[1]]),
            }
        }
        self.assertTrue(compare_smoke(before, rounded)["text"]["finite_logits"])
        with self.assertRaises(ValueError):
            compare_smoke(before, rounded, exact=True)
        self.assertTrue(
            compare_smoke(before, before, exact=True)["text"]["greedy_tokens_equal"]
        )

    def test_smoke_records_prediction_distribution_changes(self):
        import torch

        before = {
            "text": {
                "logits": torch.tensor([[1000.0, 1001.0]]),
                "tokens": torch.tensor([[1]]),
            }
        }
        shifted = {
            "text": {
                "logits": torch.tensor([[1001.0, 1002.0]]),
                "tokens": torch.tensor([[1]]),
            }
        }
        changed = {
            "text": {
                "logits": torch.tensor([[1001.0, 1000.0]]),
                "tokens": torch.tensor([[0]]),
            }
        }
        self.assertAlmostEqual(
            compare_smoke(before, shifted)["text"]["next_token_kl_nats"], 0.0
        )
        result = compare_smoke(before, changed)["text"]
        self.assertGreater(result["next_token_kl_nats"], 0.1)
        self.assertFalse(result["greedy_tokens_equal"])

    def test_tiny_merged_qwen_preserves_saved_mergers(self):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        from drone_pur.training.staged_model import ddp_adapter_load_compat

        torch.set_num_threads(1)
        torch.manual_seed(42)
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
            bos_token_id=1,
            eos_token_id=39,
            pad_token_id=0,
        )
        inputs = {
            "vision": dict(
                input_ids=torch.tensor([[1, 27, 29, 28, 3]]),
                attention_mask=torch.ones(1, 5, dtype=torch.long),
                pixel_values=torch.randn(4, 24),
                image_grid_thw=torch.tensor([[1, 2, 2]]),
            )
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = Qwen3VLForConditionalGeneration(config).eval()
            base.save_pretrained(root / "base")
            model = get_peft_model(
                base,
                LoraConfig(
                    r=4,
                    lora_alpha=4,
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj"],
                    modules_to_save=MERGERS,
                ),
            )
            with torch.no_grad():
                for name, value in model.named_parameters():
                    if "lora_B" in name:
                        value.normal_(std=0.01)
                    elif "modules_to_save.default" in name:
                        value.add_(0.01)
            model.save_pretrained(root / "adapter")
            with ddp_adapter_load_compat():
                model = PeftModel.from_pretrained(
                    Qwen3VLForConditionalGeneration.from_pretrained(root / "base"),
                    root / "adapter",
                ).eval()
            self.assertGreater(
                check_loaded_adapter(model, root / "adapter/adapter_model.safetensors"),
                24,
            )
            expected = merger_hashes(model.get_base_model(), wrapped=True)
            expected_lora = expected_lora_hashes(model.get_base_model())
            before = run_smoke(model, inputs, "cpu", 2)
            merged = model.merge_and_unload(safe_merge=True).eval()
            check_merged_lora(merged, expected_lora)
            key = next(iter(expected_lora))
            weight = merged.state_dict()[key]
            original = weight.clone()
            with torch.no_grad():
                weight.add_(1.0)
            with self.assertRaises(ValueError):
                check_merged_lora(merged, expected_lora)
            with torch.no_grad():
                weight.copy_(original)
            self.assertEqual(merger_hashes(merged), expected)
            after = run_smoke(merged, inputs, "cpu", 2)
            self.assertTrue(compare_smoke(before, after)["vision"]["finite_logits"])
            self.assertFalse(
                any(
                    "lora_" in key or "modules_to_save" in key
                    for key in merged.state_dict()
                )
            )
            merged.save_pretrained(root / "merged", safe_serialization=True)
            loaded, info = Qwen3VLForConditionalGeneration.from_pretrained(
                root / "merged", output_loading_info=True
            )
            self.assertFalse(info["missing_keys"])
            self.assertFalse(info["unexpected_keys"])
            self.assertEqual(merger_hashes(loaded), expected)
            compare_smoke(after, run_smoke(loaded, inputs, "cpu", 2), exact=True)


if __name__ == "__main__":
    unittest.main()
