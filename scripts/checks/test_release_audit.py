from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


AUDIT = Path(__file__).resolve().parents[1] / "audit_release.py"


class ReleaseAuditTests(unittest.TestCase):
    def audit(self, root):
        return subprocess.run(
            [sys.executable, str(AUDIT), "--root", str(root)],
            capture_output=True,
            text=True,
        )

    def test_documentation_png_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "scripts/assets"
            assets.mkdir(parents=True)
            Image.new("RGB", (8, 8), "white").save(assets / "figure.png")
            result = self.audit(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_malformed_png_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "scripts/assets"
            assets.mkdir(parents=True)
            (assets / "figure.png").write_bytes(b"not a PNG")
            result = self.audit(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid documentation PNG", result.stdout)

    def test_binary_outside_documentation_assets_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (8, 8), "white").save(root / "sample.png")
            result = self.audit(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unexpected binary file", result.stdout)

    def test_credentials_and_weights_remain_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "example.txt").write_text("sk-" + "a" * 30)
            (root / "weights.safetensors").write_bytes(b"weights")
            result = self.audit(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("possible credential", result.stdout)
            self.assertIn("weights/data/archive", result.stdout)


if __name__ == "__main__":
    unittest.main()
