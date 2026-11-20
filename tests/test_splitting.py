import unittest

from glenda.splitting import assert_disjoint, connected_groups


class GroupTests(unittest.TestCase):
    def test_transitive_case_video_and_duplicate_links(self):
        rows = [
            dict(path="a", case_id="1", video_id="10", sha256="a"),
            dict(path="b", case_id="1", video_id="11", sha256="b"),
            dict(path="c", case_id="", video_id="11", sha256="c"),
            dict(path="d", case_id="", video_id="12", sha256="c"),
            dict(path="e", case_id="", video_id="13", sha256="e"),
        ]
        groups = connected_groups(rows)
        self.assertEqual(len(set(groups[:4])), 1)
        self.assertNotEqual(groups[0], groups[4])
        assert_disjoint(rows, ["train"] * 4 + ["test"])
        with self.assertRaises(ValueError):
            assert_disjoint(rows, ["train", "train", "test", "train", "test"])

    def test_missing_assignments_rejected(self):
        with self.assertRaises(ValueError):
            assert_disjoint([{"path": "a"}], [])

    def test_candidate_links_and_missing_path(self):
        rows = [{"path": "a"}, {"path": "b"}]
        with self.assertRaises(ValueError):
            assert_disjoint(rows, ["train", "test"], [("a", "b")])
        with self.assertRaises(ValueError):
            connected_groups(rows, [("a", "missing")])


if __name__ == "__main__":
    unittest.main()
