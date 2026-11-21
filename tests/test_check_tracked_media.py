import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location("media_guard", Path(__file__).resolve().parents[1] / "scripts/check_tracked_media.py")
media_guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media_guard)


class MediaGuardTests(unittest.TestCase):
    def test_index_blocks_disguised_images_and_large_blobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / "readme.txt").write_text("safe", encoding="utf-8")
            subprocess.run(["git", "add", "readme.txt"], cwd=root, check=True)
            self.assertEqual(media_guard.check_index(root), 1)
            (root / "image.txt").write_bytes(b"\x89PNG\r\n\x1a\nsynthetic")
            subprocess.run(["git", "add", "image.txt"], cwd=root, check=True)
            with self.assertRaisesRegex(ValueError, "Unapproved media"):
                media_guard.check_index(root)
            subprocess.run(["git", "rm", "--cached", "-q", "image.txt"], cwd=root, check=True)
            (root / "large.txt").write_bytes(b"x" * (media_guard.MAX_BYTES + 1))
            subprocess.run(["git", "add", "large.txt"], cwd=root, check=True)
            with self.assertRaisesRegex(ValueError, "exceeds 5 MiB"):
                media_guard.check_index(root)

    def test_approved_figure_name_does_not_approve_new_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            figure = root / "reports/assets/evaluation.png"; figure.parent.mkdir(parents=True)
            figure.write_bytes(b"not the approved figure")
            subprocess.run(["git", "add", "reports/assets/evaluation.png"], cwd=root, check=True)
            with self.assertRaisesRegex(ValueError, "requires review"):
                media_guard.check_index(root)


if __name__ == "__main__":
    unittest.main()
