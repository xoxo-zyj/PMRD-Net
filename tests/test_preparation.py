"""Regression checks for moving the archived splits across machines."""
import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

from tools.prepare_splits import portable_path, resolve_path, prepare


class SplitPreparationTests(unittest.TestCase):
    def test_reject_absolute_and_parent_paths(self):
        for value in ("/server/data.png", "C:\\data\\image.png", "../image.png", "x/../image.png", "\\\\server\\share\\image.png"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                portable_path(value)

    def test_mapping_respects_path_boundary_and_longest_prefix(self):
        root = Path(tempfile.gettempdir()) / "example-data"
        mapping = {"train": "fallback", "train/ITS": "RESIDE/ITS"}
        self.assertEqual(Path(resolve_path("train/ITS/a.png", root, mapping)), root / "RESIDE/ITS/a.png")
        self.assertEqual(Path(resolve_path("training/a.png", root, mapping)), root / "training/a.png")

    def make_manifest(self, source, split, suffix):
        target = source / "ihaze" / f"split_manifest_{split}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        fields = ["split", "hazy_path", "gt_path", "group_id", "match_type"]
        row = {"split": split, "hazy_path": f"I-HAZY/hazy/{suffix}.png",
               "gt_path": f"I-HAZY/GT/{suffix}.png", "group_id": f"I-HAZY/GT/{suffix}.png", "match_type": "original"}
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
        return row

    def test_missing_images_fail_before_output_is_written(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_manifest(root / "source", "train", "a")
            with self.assertRaises(FileNotFoundError):
                prepare(root / "source", root / "local", root / "images", check_files=True)
            self.assertFalse((root / "local").exists())

    def test_preserves_membership_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = self.make_manifest(root / "source", "train", "a")
            self.make_manifest(root / "source", "val", "b")
            for suffix in ("a", "b"):
                for category in ("hazy", "GT"):
                    image = root / "images/I-HAZY" / category / f"{suffix}.png"
                    image.parent.mkdir(parents=True, exist_ok=True)
                    image.write_bytes(b"existence-check-only")
            with contextlib.redirect_stdout(io.StringIO()):
                rows = prepare(root / "source", root / "local", root / "images", check_files=True)
            actual = rows[0][2][0]
            self.assertEqual(actual["split"], expected["split"])
            self.assertEqual(actual["match_type"], expected["match_type"])
            self.assertEqual(Path(actual["hazy_path"]).name, "a.png")
            self.assertTrue(Path(actual["gt_path"]).is_file())

    def test_cross_split_scene_reuse_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_manifest(root / "source", "train", "a")
            self.make_manifest(root / "source", "val", "a")
            with self.assertRaisesRegex(ValueError, "Cross-split overlap"):
                prepare(root / "source", root / "local", root / "images")


if __name__ == "__main__":
    unittest.main()
