from pathlib import Path
import re
import unittest

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[2]


class LayoutTests(unittest.TestCase):
    def test_only_four_source_folders_at_top_level(self):
        folders = {
            path.name
            for path in ROOT.iterdir()
            if path.is_dir()
            and not path.name.startswith(".")
            and not path.name.endswith(".egg-info")
            and path.name not in {"build", "dist", "__pycache__"}
        }
        self.assertEqual(folders, {"train", "test", "generation", "scripts"})

    def test_generation_is_only_a_placeholder(self):
        self.assertLessEqual(
            {path.name for path in (ROOT / "generation").iterdir()}, {".gitkeep"}
        )

    def test_installed_namespace_maps_to_relocated_sources(self):
        import drone_pur
        from drone_pur.training import prompts

        self.assertEqual(Path(drone_pur.__file__).resolve(), ROOT / "test/__init__.py")
        self.assertEqual(Path(prompts.__file__).resolve(), ROOT / "train/prompts.py")

    def test_evaluation_fingerprints_both_source_folders(self):
        from drone_pur.evaluate import identity, parser

        args = parser().parse_args(
            [
                "--model",
                "fake-test-model",
                "--data-root",
                str(ROOT),
                "--output",
                str(ROOT / "outputs/check"),
            ]
        )
        info = identity(args, {"OC": [{"question_id": "example"}]}, {"OC": "example"})
        self.assertIn("test/evaluate.py", info["code_sha256"])
        self.assertIn("train/staged_model.py", info["code_sha256"])
        self.assertIn("train/prompts.py", info["code_sha256"])
        self.assertFalse(
            any(key.startswith("test/training/") for key in info["code_sha256"])
        )

    def test_relative_document_links_resolve(self):
        for path in ROOT.rglob("*.md"):
            if any(part.endswith(".egg-info") for part in path.parts):
                continue
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
                if "://" in target or target.startswith(("mailto:", "#")):
                    continue
                relative = target.split("#", 1)[0]
                with self.subTest(document=str(path.relative_to(ROOT)), target=target):
                    self.assertTrue((path.parent / relative).is_file())

    def test_shared_requirements_cover_training_evaluation_and_checks(self):
        names = {
            canonicalize_name(Requirement(line).name)
            for line in (ROOT / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        self.assertLessEqual(
            {
                "torch",
                "torchvision",
                "transformers",
                "peft",
                "accelerate",
                "safetensors",
                "numpy",
                "scipy",
                "pillow",
                "requests",
                "huggingface-hub",
                "einops",
                "timm",
                "sentencepiece",
                "protobuf",
                "pytest",
            },
            names,
        )
        self.assertFalse((ROOT / "requirements-training.txt").exists())


if __name__ == "__main__":
    unittest.main()
