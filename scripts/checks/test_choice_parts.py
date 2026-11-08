import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from drone_pur.data import read_json, write_json
from drone_pur.evaluate import save_results


SPEC = importlib.util.spec_from_file_location(
    "choice_parts_merger", Path(__file__).resolve().parents[1] / "merge_choice_parts.py"
)
MERGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGER)


class MergeChoicePartsTests(unittest.TestCase):
    def fixtures(self):
        ids = {"MMR-OC": ["a"], "MMR-R": ["b", "c"], "OC": ["d", "e"]}
        expected = {
            "mode": "choice",
            "backend": "qwen",
            "model": {"path": "example-model"},
            "code_sha256": {"test/example.py": "source-hash"},
            "ids": ids,
            "annotations_sha256": {t: "annotation-" + t for t in ids},
            "max_new_tokens": {t: 20 for t in ids},
        }
        samples = {
            task: [
                {
                    "question_id": qid,
                    "options": {"A": "one", "B": "two"},
                    "correct_answer": "A",
                }
                for qid in qids
            ]
            for task, qids in ids.items()
        }
        pairs = []
        predictions = {"a": "A", "b": "B", "c": None, "d": "A", "e": "A"}
        for tasks in (("MMR-OC", "MMR-R"), ("OC",)):
            info = {
                k: copy.deepcopy(v)
                for k, v in expected.items()
                if k not in MERGER.TASK_FIELDS
            }
            for field in ("ids", "annotations_sha256", "max_new_tokens"):
                info[field] = {task: expected[field][task] for task in tasks}
            info["input_fingerprints"] = {
                task: {qid: "fingerprint-" + qid for qid in ids[task]} for task in tasks
            }
            manifest = {
                "identity": info,
                "versions": {"torch": "example-version"},
                "model_name": "example-model",
                "selected_counts": {task: len(ids[task]) for task in tasks},
            }
            rows = [
                {
                    "task": task,
                    "question_id": qid,
                    "prediction": predictions[qid],
                    "completed": True,
                }
                for task in tasks
                for qid in ids[task]
            ]
            pairs.append((manifest, rows))
        return expected, samples, pairs

    def test_weighted_merge_and_invalid_answers(self):
        expected, samples, pairs = self.fixtures()
        ident, rows, _, _ = MERGER.collect_parts(expected, pairs)
        self.assertEqual(set(ident["input_fingerprints"]), set(samples))
        with tempfile.TemporaryDirectory() as tmp:
            save_results(
                SimpleNamespace(output=Path(tmp), mode="choice"), samples, rows
            )
            report = read_json(Path(tmp) / "summary.json")
            self.assertEqual(report["overall"]["metrics"]["accuracy_percent"], 60.0)
            self.assertEqual(report["overall"]["n_completed"], 5)
            self.assertEqual(report["overall"]["n_invalid"], 1)
            self.assertAlmostEqual(
                report["tasks"]["MMR"]["metrics"]["accuracy_percent"], 100 / 3
            )

    def test_different_models_and_duplicate_tasks_rejected(self):
        expected, _, pairs = self.fixtures()
        altered = copy.deepcopy(pairs)
        altered[1][0]["identity"]["model"]["path"] = "another-model"
        with self.assertRaisesRegex(ValueError, "identical model"):
            MERGER.collect_parts(expected, altered)
        with self.assertRaisesRegex(ValueError, "multiple partitions"):
            MERGER.collect_parts(expected, [*pairs, pairs[0]])

    def test_subsets_annotations_and_missing_tasks_rejected(self):
        expected, _, pairs = self.fixtures()
        for field, replacement in (
            ("ids", ["d"]),
            ("annotations_sha256", "changed"),
            ("max_new_tokens", 999),
        ):
            altered = copy.deepcopy(pairs)
            altered[1][0]["identity"][field]["OC"] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                MERGER.collect_parts(expected, altered)
        with self.assertRaisesRegex(ValueError, "cover all choice tasks"):
            MERGER.collect_parts(expected, pairs[:1])

    def test_duplicate_rows_rejected_and_incomplete_not_zero_filled(self):
        expected, samples, pairs = self.fixtures()
        altered = copy.deepcopy(pairs)
        altered[0][1].append(altered[0][1][0])
        with self.assertRaisesRegex(ValueError, "Duplicate result"):
            MERGER.collect_parts(expected, altered)
        partial = copy.deepcopy(pairs)
        partial[1][1].pop()
        _, rows, _, _ = MERGER.collect_parts(expected, partial)
        with tempfile.TemporaryDirectory() as tmp:
            save_results(
                SimpleNamespace(output=Path(tmp), mode="choice"), samples, rows
            )
            report = read_json(Path(tmp) / "summary.json")
            self.assertEqual(report["overall"]["n_missing"], 1)
            self.assertFalse(report["overall"]["complete"])
            self.assertIsNone(report["overall"]["metrics"])
            self.assertEqual(
                report["overall"]["completed_metrics"]["accuracy_percent"], 50.0
            )

    def test_cli_writes_complete_outputs_and_requires_finished_parts(self):
        expected, samples, pairs = self.fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = [root / "part1", root / "part2"]
            for path, (manifest, rows) in zip(parts, pairs):
                write_json(path / "manifest.json", manifest)
                write_json(path / "results.json", rows)
            command = [
                "merge",
                "--model",
                str(root / "model"),
                "--data-root",
                str(root / "benchmark"),
                "--max-pixels",
                "1003520",
                "--device-map",
                "cuda:0",
                "--parts",
                *map(str, parts),
                "--output",
                str(root / "merged"),
            ]
            with (
                patch.object(MERGER, "load_samples", return_value=(samples, {})),
                patch.object(MERGER, "identity", return_value=expected) as identify,
            ):
                with patch("sys.argv", command):
                    MERGER.main()
                summary = read_json(root / "merged/summary.json")
                self.assertTrue(summary["overall"]["complete"])
                self.assertTrue((root / "merged/summary.csv").is_file())
                self.assertEqual(len(read_json(root / "merged/results.json")), 5)
                actual_args = identify.call_args.args[0]
                self.assertEqual(actual_args.max_pixels, 1003520)
                self.assertEqual(actual_args.device_map, "cuda:0")
                write_json(parts[1] / "results.json", pairs[1][1][:-1])
                incomplete = command[:-1] + [str(root / "incomplete")]
                with (
                    patch("sys.argv", incomplete),
                    self.assertRaises(SystemExit) as caught,
                ):
                    MERGER.main()
                self.assertEqual(caught.exception.code, 2)
                self.assertFalse((root / "incomplete").exists())


if __name__ == "__main__":
    unittest.main()
