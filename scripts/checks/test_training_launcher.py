import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TrainingLauncherTests(unittest.TestCase):
    def capture_commands(self, overrides=None):
        launcher = Path(__file__).resolve().parents[2] / "scripts/train_uav_pur.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_python = root / "python"
            fake_python.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['DRONE_PUR_CAPTURE'], 'a') as handle:\n"
                "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            for name in ("NPROC_PER_NODE", "GRAD_ACC"):
                env.pop(name, None)
            env.update(
                PATH=str(root) + os.pathsep + env["PATH"],
                DRONE_PUR_CAPTURE=str(root / "commands.jsonl"),
            )
            env.update(overrides or {})
            subprocess.run(
                ["bash", str(launcher), "example base", "example data", "example run"],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            return [
                json.loads(line)
                for line in (root / "commands.jsonl").read_text().splitlines()
            ]

    def test_all_stages_default_to_eight_gpus_and_batch_32(self):
        commands = self.capture_commands()
        self.assertEqual(len(commands), 3)
        for stage, command in zip(("s1", "s2", "s3"), commands):
            self.assertIn("--nproc_per_node=8", command)
            self.assertEqual(command[command.index("--grad-acc") + 1], "4")
            self.assertEqual(command[command.index("--stage") + 1], stage)
            self.assertEqual(command[command.index("--model") + 1], "example base")
            self.assertEqual(8 * 1 * int(command[command.index("--grad-acc") + 1]), 32)

    def test_explicit_smaller_layout_override_is_preserved(self):
        commands = self.capture_commands({"NPROC_PER_NODE": "4", "GRAD_ACC": "8"})
        self.assertEqual(len(commands), 3)
        for command in commands:
            self.assertIn("--nproc_per_node=4", command)
            self.assertEqual(command[command.index("--grad-acc") + 1], "8")


if __name__ == "__main__":
    unittest.main()
