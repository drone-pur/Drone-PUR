import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from drone_pur.data import write_json
from drone_pur.tasks import TRAIN_KEYS

from drone_pur.training.prepare import (
    build_mixtures,
    open_view,
    sample_passes,
    select_od,
    main as prepare_main,
)


class PreparationTests(unittest.TestCase):
    def test_default_stages_each_train_one_epoch(self):
        root = Path(__file__).resolve().parents[2]
        config = json.loads((root / "train/configs/uav_pur.json").read_text())
        self.assertEqual(list(config["stages"]), ["s1", "s2", "s3"])
        for stage in config["stages"].values():
            self.assertEqual(stage["epochs"], 1)
        launcher = (root / "scripts/train_uav_pur.sh").read_text()
        self.assertNotIn("--stop-after-epoch", launcher)

    def test_full_preparation_cli_and_read_only_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "train"
            (data / "images").mkdir(parents=True)
            Image.new("RGB", (32, 24), "white").save(data / "images/a.png")
            config = json.loads(
                (
                    Path(__file__).resolve().parents[2] / "train/configs/uav_pur.json"
                ).read_text()
            )
            config["od_samples"] = 1
            write_json(root / "config.json", config)
            sources = {}
            for leaf in [*TRAIN_KEYS, "OD"]:
                row = {
                    "question_id": leaf + "_train_1",
                    "question": "Inspect the target",
                    "images": [{"path": "images/a.png"}],
                    "options": dict(zip("ABCDE", ["1", "2", "3", "4", "5"])),
                    "correct_answer": "A",
                }
                if leaf in {"VG", "MIG", "MMR-VG", "MVR-VG"}:
                    row["options"] = {key: "[0,0.1,0.4,0.5]" for key in "ABCDE"}
                if leaf in {"MIG", "MVR-VG"}:
                    row["template"] = {"box": [0.1, 0.2, 0.3, 0.4]}
                    row["images"] *= 4 if leaf == "MIG" else 2
                elif leaf.startswith(("MMR-", "MVR-")):
                    row["images"] *= 2
                if leaf == "CJ":
                    row["correct_answer"] = "Yes"
                if leaf == "OD":
                    row["ground_truth"] = {"type": "boxes", "boxes": []}
                sources[leaf] = leaf + ".json"
                write_json(data / sources[leaf], {"split": "train", "results": [row]})
            write_json(root / "sources.json", sources)
            argv = [
                "prepare",
                "--data-root",
                str(data),
                "--sources",
                str(root / "sources.json"),
                "--config",
                str(root / "config.json"),
                "--output",
                str(root / "prepared"),
            ]
            with patch("sys.argv", argv + ["--dry-run"]):
                prepare_main()
            self.assertFalse((root / "prepared").exists())
            with patch("sys.argv", argv):
                prepare_main()
            od_ids = {}
            for stage in ("s1", "s2", "s3"):
                manifest = json.loads(
                    (root / "prepared" / stage / "manifest.json").read_text()
                )
                rows = [
                    json.loads(line)
                    for line in (root / "prepared" / stage / "train.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(len(rows), manifest["audit"]["records_per_epoch"])
                self.assertEqual(manifest["stage"], stage)
                self.assertEqual(manifest["config"]["stages"][stage]["epochs"], 1)
                self.assertEqual(manifest["audit"]["epochs"], 1)
                od_ids[stage] = {r["question_id"] for r in rows if r["task"] == "od"}
                expected_od = 1 if stage in {"s1", "s2"} else 0
                self.assertEqual(len(od_ids[stage]), expected_od)
                self.assertEqual(manifest["audit"]["od_selected"], expected_od)
                self.assertEqual(
                    manifest["audit"]["task_view_counts"].get("od/open", 0), expected_od
                )
            self.assertEqual(od_ids["s1"], od_ids["s2"])

    def test_sampling_is_reproducible(self):
        rows = list(range(20))
        a = sample_passes(rows, 1.5, random.Random(42))
        b = sample_passes(rows, 1.5, random.Random(42))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 30)
        self.assertEqual(set(a[:20]), set(rows))

    def test_open_input_does_not_depend_on_target_presence(self):
        record = {
            "task": "mvr_vg",
            "question_id": "x",
            "correct_answer": "E",
            "images": ["a", "b"],
            "messages": [],
        }
        raw = {
            "question": "Find the target",
            "correct_answer": "E",
            "options": {"E": "not exist"},
            "template_coordinates": ["[0.1,0.2,0.3,0.4]"],
        }
        absent = open_view(record, raw)
        visible_raw = copy.deepcopy(raw)
        visible_raw["options"]["E"] = "[0.2,0.3,0.4,0.5]"
        visible = open_view(record, visible_raw)
        self.assertEqual(absent["messages"][0], visible["messages"][0])
        self.assertEqual(absent["aux_kind"], "none")
        self.assertEqual(visible["aux_kind"], "box")
        self.assertNotIn("Options:", visible["messages"][0]["content"])

    def test_s3_mixed_counts_and_no_od(self):
        config = json.loads(
            (
                Path(__file__).resolve().parents[2] / "train/configs/uav_pur.json"
            ).read_text()
        )
        pools = {
            key: [
                {"task": key, "question_id": f"{key}-{i}", "view": "choice"}
                for i in range(8)
            ]
            for key in config["stages"]["s3"]["choice_rates"]
        }
        opens = {
            key: [dict(row, view="open") for row in pools[key]]
            for key in config["stages"]["s3"]["open_rates"]
        }
        od = [{"task": "od", "question_id": "od-1", "view": "open"}]
        stages = build_mixtures(config, pools, opens, od)
        self.assertEqual(len(stages["s3"]), 13 * 8 + 6 * 2)
        self.assertFalse(any(row["task"] == "od" for row in stages["s3"]))
        for name in ("s1", "s2"):
            self.assertEqual([row for row in stages[name] if row["task"] == "od"], od)
        without_od = build_mixtures(config, pools, opens, [])
        for name, rows in stages.items():
            selected = sorted(
                (r["task"], r["question_id"], r["view"])
                for r in rows
                if r["task"] != "od"
            )
            expected = sorted(
                (r["task"], r["question_id"], r["view"]) for r in without_od[name]
            )
            self.assertEqual(selected, expected)

    def test_od_selection_is_without_replacement(self):
        rows = [
            {
                "question_id": str(i),
                "images": [f"images/VisDrone/{i}.jpg"],
                "target_boxes": [],
            }
            for i in range(10)
        ]
        selected = select_od(rows, 5, 42)
        self.assertEqual(len({r["question_id"] for r in selected}), 5)
        self.assertEqual(selected, select_od(rows, 5, 42))


if __name__ == "__main__":
    unittest.main()
