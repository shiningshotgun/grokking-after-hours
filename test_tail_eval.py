import unittest

from train import tail_eval_due


class TailEvalDueTests(unittest.TestCase):
    def test_off_by_default(self):
        self.assertFalse(tail_eval_due(0, None, 10))
        self.assertFalse(tail_eval_due(100000, None, 10))

    def test_boundary_and_cadence(self):
        self.assertFalse(tail_eval_due(98949, 98950, 10))
        self.assertTrue(tail_eval_due(98950, 98950, 10))
        self.assertFalse(tail_eval_due(98955, 98950, 10))
        self.assertTrue(tail_eval_due(98960, 98950, 10))
        self.assertTrue(tail_eval_due(100000, 98950, 10))
        self.assertFalse(tail_eval_due(100001, 98950, 10))

    def test_every_one(self):
        for gs in (5, 6, 7):
            self.assertTrue(tail_eval_due(gs, 5, 1))


if __name__ == "__main__":
    unittest.main()
