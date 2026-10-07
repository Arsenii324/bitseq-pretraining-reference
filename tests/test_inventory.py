import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class InventoryContract(unittest.TestCase):
    def verifier(self):
        path = Path(__file__).resolve().parents[1] / "verify.py"
        self.assertTrue(path.exists(), "Missing package verifier")
        spec = importlib.util.spec_from_file_location("package_verifier", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.verify_inventory

    def fixture(self, root):
        raw = b"actual exported bytes\n"
        (root / "model.pt").write_bytes(raw)
        record = {
            "schema": "bitseq-export-inventory-v1",
            "files": {
                "model.pt": {
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                }
            },
        }
        (root / "inventory.json").write_text(json.dumps(record))

    def test_valid_bytes_and_git_work_are_allowed(self):
        check = self.verifier()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            (root / ".git").mkdir()
            (root / ".git/config").write_text("local git state")
            (root / "work").mkdir()
            (root / "work/new.json").write_text("new working result")
            self.assertEqual(check(root), 1)

    def test_mutated_weight_is_rejected(self):
        check = self.verifier()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            (root / "model.pt").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                check(root)

    def test_missing_and_unregistered_files_are_rejected(self):
        check = self.verifier()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            (root / "unexpected.txt").write_text("unregistered")
            with self.assertRaisesRegex(ValueError, "file set"):
                check(root)
            (root / "unexpected.txt").unlink()
            (root / "model.pt").unlink()
            with self.assertRaisesRegex(ValueError, "file set"):
                check(root)

    def test_unsafe_manifest_path_is_rejected(self):
        check = self.verifier()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            data = json.loads((root / "inventory.json").read_text())
            data["files"]["../outside"] = data["files"].pop("model.pt")
            (root / "inventory.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "unsafe"):
                check(root)

    def test_symlink_is_rejected(self):
        check = self.verifier()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            (root / "other").write_text("outside inventory")
            (root / "model.pt").unlink()
            (root / "model.pt").symlink_to(root / "other")
            with self.assertRaisesRegex(ValueError, "symlink"):
                check(root)

    def test_fractional_clock_cannot_be_truncated_into_expected_integer(self):
        path = Path(__file__).resolve().parents[1] / "verify.py"
        spec = importlib.util.spec_from_file_location("package_clock", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(
            hasattr(module, "verify_clock"), "Missing strict Adam clock validation"
        )
        self.assertEqual(module.verify_clock([16500.0] * 30, 16500), 16500)
        for values in ([16500.5] * 30, [float("nan")] * 30, [12500.0] * 30):
            with self.assertRaisesRegex(ValueError, "clock"):
                module.verify_clock(values, 16500)


if __name__ == "__main__":
    unittest.main()
