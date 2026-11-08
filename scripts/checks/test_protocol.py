import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from drone_pur.data import write_json
from drone_pur.evaluate import parser, run
from drone_pur.metrics import (
    box_iou,
    count_metrics,
    evaluate_samples,
    match_boxes,
    score_prediction,
)
from drone_pur.tasks import (
    build_prompt,
    parse_response,
    prompt_sample,
    TRAIN_KEYS,
    uniform_subset,
)
from drone_pur.templates import render_template
from drone_pur.utils import output_lock


class PromptTests(unittest.TestCase):
    def test_preserved_prompt_golden_cases(self):
        from drone_pur.training.prompts import build_prompt as train_prompt

        cases = json.loads(Path(__file__).with_name("prompt_cases.json").read_text())
        for case in cases:
            with self.subTest(task=case["task"], mode=case["mode"]):
                text = (
                    train_prompt(
                        prompt_sample(case["sample"]), TRAIN_KEYS[case["task"]]
                    )
                    if case["mode"] == "training"
                    else build_prompt(case["sample"], case["task"], case["mode"])
                )
                self.assertEqual(
                    hashlib.sha256(text.encode()).hexdigest(), case["sha256"]
                )

    def test_targets_do_not_enter_prompt(self):
        row = {
            "question": "Find the target",
            "options": dict(zip("ABCDE", "12345")),
            "correct_answer": "A",
            "template": {"box": [0.1, 0.2, 0.3, 0.4]},
        }
        before = build_prompt(row, "MVR-VG", "choice")
        row.update(
            ground_truth={"type": "bbox", "box": [0.6, 0.6, 0.7, 0.7]},
            correct_answer="B",
            reasoning_clues="A hidden answer",
            high_box=[0.8, 0.8, 0.9, 0.9],
        )
        self.assertEqual(before, build_prompt(row, "MVR-VG", "choice"))
        text = build_prompt(row, "MVR-VG", "open")
        self.assertNotIn("Options:", text)
        self.assertNotIn("hidden answer", text)

    def test_parsers(self):
        for task in TRAIN_KEYS:
            if task == "CJ":
                self.assertEqual(parse_response("Yes", task, "choice"), "Yes")
            else:
                self.assertEqual(parse_response("B", task, "choice"), "B")
        self.assertEqual(parse_response("17", "OC", "open"), 17)
        self.assertIsNone(parse_response("17 or 18", "OC", "open"))
        self.assertEqual(parse_response("not exist", "MVR-VG", "open"), "not exist")
        self.assertEqual(parse_response('{"boxes":[]}', "OD", "open"), [])
        self.assertIsNone(parse_response('{"boxes":[1,2,3,4]}', "OD", "open"))
        self.assertIsNone(parse_response('{"boxes":[[0,0,0.5,0.5],[]]}', "OD", "open"))


class MetricTests(unittest.TestCase):
    def test_strict_grounding_boundary(self):
        sample = {"ground_truth": {"type": "bbox", "box": [0, 0, 1, 1]}}
        self.assertEqual(box_iou([0, 0, 0.5, 1], [0, 0, 1, 1]), 0.5)
        self.assertFalse(
            score_prediction("VG", sample, [0, 0, 0.5, 1], "open")["correct"]
        )

    def test_mra_boundaries_zero_and_invalid(self):
        self.assertEqual(count_metrics(6, 8)["mra"], 0.5)
        self.assertEqual(count_metrics(0, 0)["mra"], 1)
        self.assertEqual(count_metrics(1, 0)["mra"], 0)
        self.assertFalse(count_metrics(-1, 8)["valid"])
        self.assertFalse(count_metrics(True, 8)["valid"])

    def test_maximum_cardinality_not_greedy(self):
        edges = {(0, 0): 0.9, (0, 1): 0.7, (1, 0): 0.6, (1, 1): 0}
        with patch("drone_pur.metrics.box_iou", side_effect=lambda a, b: edges[a, b]):
            self.assertEqual(match_boxes([0, 1], [0, 1]), {"tp": 2, "fp": 0, "fn": 0})

    def test_weighted_grouping_and_missing(self):
        samples = {
            "MMR-OC": [
                {
                    "question_id": f"c{i}",
                    "options": {"A": 1, "B": 2},
                    "correct_answer": "A",
                }
                for i in range(3)
            ],
            "MMR-R": [
                {"question_id": "r1", "options": {"A": 1}, "correct_answer": "A"}
            ],
        }
        predictions = [
            {"question_id": "c0", "prediction": "A"},
            {"question_id": "c1", "prediction": "B"},
            {"question_id": "c2", "prediction": None},
            {"question_id": "r1", "prediction": "A"},
        ]
        result = evaluate_samples(samples, predictions, "choice")
        self.assertEqual(result["tasks"]["MMR"]["metrics"]["accuracy_percent"], 50)
        self.assertEqual(result["overall"]["n_invalid"], 1)
        incomplete = evaluate_samples(samples, predictions[:-1], "choice")
        self.assertIsNone(incomplete["overall"]["metrics"])
        self.assertEqual(incomplete["overall"]["n_missing"], 1)

    def test_invalid_count_full_mae_is_undefined(self):
        samples = {
            "OC": [
                {"question_id": "1", "ground_truth": {"type": "count", "count": 5}},
                {"question_id": "2", "ground_truth": {"type": "count", "count": 5}},
            ]
        }
        result = evaluate_samples(
            samples,
            [
                {"question_id": "1", "prediction": 4},
                {"question_id": "2", "prediction": None},
            ],
            "open",
        )
        metrics = result["subtasks"]["OC"]["metrics"]
        self.assertIsNone(metrics["mae"])
        self.assertEqual(metrics["mae_valid"], 1)
        self.assertEqual(result["subtasks"]["OC"]["n_invalid"], 1)


class IOTests(unittest.TestCase):
    def test_sampling(self):
        self.assertEqual(uniform_subset(list(range(11)), every=5), [0, 5, 10])
        self.assertEqual(uniform_subset(list(range(11)), max_samples=3), [0, 3, 7])
        with self.assertRaises(ValueError):
            uniform_subset([1], every=0)

    def test_template_keeps_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            Image.new("RGB", (100, 80), "white").save(source)
            before = source.read_bytes()
            render_template(
                source,
                root / "derived/a.jpg",
                {
                    "box": [0.1, 0.2, 0.5, 0.6],
                    "padding_ratio": 0.002,
                    "min_line_width": 4,
                    "line_width_ratio": 0.004,
                    "jpeg_quality": 95,
                },
            )
            self.assertEqual(before, source.read_bytes())
            with Image.open(root / "derived/a.jpg") as im:
                r, g, b = im.getpixel((10, 20))
                self.assertGreater(r, g + 80)
                self.assertGreater(r, b + 80)

    def test_exclusive_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            with output_lock(tmp):
                with self.assertRaises(RuntimeError):
                    with output_lock(tmp):
                        pass


class RunnerTests(unittest.TestCase):
    def fixture(self, root):
        data = root / "benchmark"
        (data / "images").mkdir(parents=True)
        Image.new("RGB", (20, 20), "white").save(data / "images/a.png")
        samples = [
            {
                "question_id": "OC_1",
                "images": [{"path": "images/a.png"}],
                "question": "Count objects",
                "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
                "correct_answer": "A",
                "ground_truth": {"type": "count", "count": 1},
            }
        ]
        write_json(
            data / "annotations/OC/test.json",
            {
                "dataset": "Drone-PUR",
                "split": "test",
                "task": "OC",
                "subtask": None,
                "total_samples": 1,
                "results": samples,
            },
        )
        return parser().parse_args(
            [
                "--model",
                "fake-test-model",
                "--data-root",
                str(data),
                "--output",
                str(root / "output"),
                "--tasks",
                "OC",
            ]
        )

    def test_resume_and_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))

            class Fake:
                def infer(self, paths, prompt, limit):
                    return "A", {"test": True}

            self.assertEqual(run(args, lambda _: Fake()), 0)

            def forbidden(_):
                raise AssertionError("A finished run must not load a model")

            self.assertEqual(run(args, forbidden), 0)
            result = json.loads((args.output / "summary.json").read_text())
            self.assertEqual(result["overall"]["metrics"]["accuracy_percent"], 100)
            args.max_new_tokens = 99
            with self.assertRaisesRegex(ValueError, "identity changed"):
                run(args, forbidden)

    def test_infrastructure_failure_is_resumable_not_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))

            class Broken:
                def infer(self, *args):
                    raise TypeError("temporary backend signature failure")

            self.assertEqual(run(args, lambda _: Broken()), 2)
            result = json.loads((args.output / "summary.json").read_text())
            self.assertEqual(result["overall"]["n_missing"], 1)
            self.assertEqual(result["overall"]["n_invalid"], 0)

            class Fixed:
                def infer(self, *args):
                    return "A", {}

            self.assertEqual(run(args, lambda _: Fixed()), 0)

    def test_check_only_has_no_model_or_output_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.check_only = True
            self.assertEqual(run(args, lambda _: self.fail("Loaded a model")), 0)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
