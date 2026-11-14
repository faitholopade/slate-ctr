import tempfile
import unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from slate_ctr import choice_probabilities, correct_event_mass, correct_choices, weekend_correction
from slate_ctr.model import LinearChoiceModel
from slate_ctr.demo import generate, auc


class ProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.base = np.array([[0.1, 0.2, 0.3], [0.04, 0.02, 0.01]])

    def test_outside_option_and_extreme_utilities(self):
        assert_allclose(choice_probabilities([[0, 0]]), [[1/3, 1/3, 1/3]])
        p = choice_probabilities([[1000, -1000], [-1000, -1000]])
        assert_allclose(p, [[1, 0, 0], [0, 0, 1]])

    def test_event_shift_preserves_choice_and_changes_mass(self):
        p = correct_event_mass(self.base, [0.7, -0.4])
        assert_allclose(p / p.sum(1, keepdims=True), self.base / self.base.sum(1, keepdims=True))
        self.assertGreater(p[0].sum(), self.base[0].sum())
        self.assertLess(p[1].sum(), self.base[1].sum())
        assert_allclose(correct_event_mass(self.base, [0, 0]), self.base)

    def test_choice_shift_preserves_mass_and_zero_support(self):
        base = np.array([[0, 0.2, 0.3], [0, 0, 0]])
        p = correct_choices(base, [[1000, 1, -1], [1, 2, 3]])
        assert_allclose(p.sum(1), base.sum(1))
        self.assertEqual(p[0, 0], 0)
        assert_array_equal(p[1], [0, 0, 0])
        assert_array_equal(correct_choices(base, np.ones_like(base), 0), base)

    def test_mass_boundaries(self):
        assert_allclose(correct_event_mass([[0, 0], [0.3, 0.7]], [1000, -1000]), [[0, 0], [0.3, 0.7]])

    def test_candidate_permutation_and_bounded_weekend_delta(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(2, 3, 4))
        mean, std, weight = np.zeros(4), np.ones(4), np.arange(4) * 0.1
        p, delta = weekend_correction(self.base, x, [0, 1], mean, std, weight)
        perm = [2, 0, 1]
        q, _ = weekend_correction(self.base[:, perm], x[:, perm], [0, 1], mean, std, weight)
        assert_allclose(q, p[:, perm], atol=1e-15)
        self.assertTrue((abs(delta) <= 0.2).all())
        assert_allclose(p.sum(1), self.base.sum(1))

    def test_independent_v142_formula(self):
        rng = np.random.default_rng(142)
        base = choice_probabilities(rng.normal(size=(20, 10)))[:, :10]
        x = rng.normal(size=(20, 10, 3))
        weight = np.array([0.2, -0.1, 0.3])
        weekend = np.arange(20) % 2
        centered = x - x.mean(1, keepdims=True)
        delta = 0.2 * np.tanh((centered @ weight) * (weekend - 2/7)[:, None] / 0.2)
        weights = base * np.exp(0.5 * delta)
        expected = weights / weights.sum(1, keepdims=True) * base.sum(1, keepdims=True)
        actual, _ = weekend_correction(base, x, weekend, np.zeros(3), np.ones(3), weight)
        assert_allclose(actual, expected, atol=1e-15)

    def test_reject_invalid_inputs(self):
        for base in ([[float('nan'), 0]], [[0.7, 0.4]], [[-0.1, 0.3]], []):
            with self.assertRaises(ValueError):
                correct_choices(base, [[0, 0]])
        with self.assertRaises(ValueError):
            correct_event_mass(self.base, [0])
        with self.assertRaises(ValueError):
            weekend_correction(self.base, np.ones((2, 3, 1)), [0, 2], [0], [1], [1])
        with self.assertRaises(ValueError):
            weekend_correction(self.base, np.ones((2, 3, 1)), [0, 1], [0], [0], [1])


class ModelTests(unittest.TestCase):
    def test_fit_generalizes_and_roundtrips(self):
        x, y = generate(1200, 5)
        test, target = generate(600, 6)
        model = LinearChoiceModel.fit(x, y, steps=250)
        p = model.predict(test)
        loss = -np.log(p[np.arange(len(target)), target]).mean()
        self.assertLess(loss, np.log(11) - 0.3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            model.save(path)
            assert_array_equal(LinearChoiceModel.load(path).predict(test), p)

    def test_bad_targets_and_models(self):
        x, y = generate(5, 8)
        with self.assertRaises(ValueError):
            LinearChoiceModel.fit(x, y + 11)
        with self.assertRaises(ValueError):
            LinearChoiceModel.fit(x, y.astype(float))
        with self.assertRaises(ValueError):
            LinearChoiceModel(np.array([1])).predict(x)

    def test_auc_ties_and_order(self):
        self.assertEqual(auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertEqual(auc([0, 1, 0, 1], [0, 1, 0, 1]), 1.0)
        self.assertEqual(auc([0, 1], [1, 0]), 0.0)
        with self.assertRaises(ValueError):
            auc([1, 1], [0.1, 0.3])


if __name__ == '__main__':
    unittest.main()
