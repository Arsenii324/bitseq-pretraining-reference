"""Stdlib-only gates: preserve the real loop; reject wrong lineage/selection.

Run directly with unittest while scientific device work owns the machine.
These gates cannot establish neural replay or calibration.
"""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "checks/reproduce_base.py"


class IndependentBaseStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("independent_base", SCRIPT)
        cls.recipe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.recipe)

    def test_hard_initializer_provenance_requires_fixed_seed_and_full_budget(self):
        # Relabelling a seed100 initializer or early-stop hard run is a bug.
        config = dict(seed=101, device="mps", steps=6000, eval_every=6000,
                      batch=256, lr=6e-4, clip=1.0, weight_decay=0.0,
                      target_alpha=12, stop_on_calibration=False,
                      schema="BS-pretrain-v1.3")
        self.recipe.validate_hard_config(config, 101)
        for field, value in [("seed", 100), ("steps", 5999),
                             ("stop_on_calibration", True), ("batch", 8)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.recipe.validate_hard_config(dict(config, **{field: value}), 101)

    def test_parent_contract_rejects_wrong_seed_kind_counters_and_source(self):
        # Accepting an independent-looking filename with wrong parent is a bug.
        meta = dict(seed=101, kind="hard", stage="hard", recipe_hash="a" * 64,
                    source_hash="b" * 64, step=6000, optimizer_steps=6000,
                    total_training_steps=6000, samples=1536000)
        kwargs = dict(seed=101, stage="hard", recipe_hash="a" * 64,
                      source_hash="b" * 64)
        self.recipe.validate_parent_metadata(meta, **kwargs)
        for field, bad in [("seed", 100), ("kind", "soft"), ("stage", "soft-high"),
                           ("samples", 1664000), ("step", 5999),
                           ("total_training_steps", 12500), ("source_hash", "c" * 64)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.recipe.validate_parent_metadata(dict(meta, **{field: bad}), **kwargs)

    def test_only_fixed_final_checkpoint_can_be_accepted(self):
        # Earlier calibrated checkpoints or final failures must never select a base.
        result = self.recipe.final_status(500, 12500, 3200000, dict(calibrated=True))
        self.assertEqual(result, "calibrated")
        self.assertEqual(self.recipe.final_status(500, 12500, 3200000,
                                                  dict(calibrated=False)), "calibration_failed")
        for values in [(300, 12300, 3148800), (500, 12500, 1664000)]:
            with self.assertRaises(ValueError):
                self.recipe.final_status(*values, dict(calibrated=True))

    def test_finalization_replaces_stage_status_without_duplicate_keyword_crash(self):
        # The real stage receipt already has status; final publication must merge it.
        receipt = dict(status="complete", step=500, total_training_steps=12500,
                       samples=3200000, checkpoint="ckpt_000500.pt",
                       evaluation=dict(calibrated=False))
        result = self.recipe.finalize_result(receipt)
        self.assertEqual(result["status"], "calibration_failed")
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(result["checkpoint"], "ckpt_000500.pt")
        accepted = self.recipe.finalize_result(dict(receipt, evaluation=dict(calibrated=True)))
        self.assertEqual(accepted["status"], "calibrated")

    def test_storage_cap_counts_shared_checkpoint_inode_once(self):
        # latest.pt/milestone.pt may be hardlinks to the same real checkpoint.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "checkpoint.pt"
            first.write_bytes(b"0123456789")
            os.link(first, root / "latest.pt")
            second = root / "source.zip"
            second.write_bytes(b"abcd")
            self.assertEqual(self.recipe.storage_bytes(root), 14)


if __name__ == "__main__":
    unittest.main()
