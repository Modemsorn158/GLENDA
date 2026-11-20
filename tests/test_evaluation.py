import unittest

from glenda.evaluation import cluster_bootstrap, metrics


class EvaluationTests(unittest.TestCase):
    def test_confusion_and_metric_definitions(self):
        result = metrics([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9], 0.5)
        self.assertEqual([result[k] for k in ("tn", "fp", "fn", "tp")], [1, 1, 1, 1])
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["specificity"], 0.5)
        self.assertEqual(result["f1"], 0.5)
        self.assertIsNone(metrics([0, 1], [0.1, 0.2], 0.5)["precision"])

    def test_cluster_bootstrap_is_seeded(self):
        args = ([0, 1, 0, 1], [0.1, 0.9, 0.8, 0.2], ["a", "a", "b", "b"], 0.5)
        self.assertEqual(cluster_bootstrap(*args, samples=20), cluster_bootstrap(*args, samples=20))
        with self.assertRaises(ValueError):
            cluster_bootstrap([0, 1], [0.1, 0.9], ["a", "a"], 0.5)


if __name__ == "__main__":
    unittest.main()
