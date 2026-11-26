"""Checks for the exact best-of-N estimator against brute force and known edge cases."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bon import bon_kl, bon_weights, expected_best  # noqa: E402


@pytest.mark.parametrize("M,N", [(1, 1), (5, 1), (5, 3), (5, 5), (256, 1), (256, 16), (256, 256)])
def test_weights_are_a_distribution(M, N):
    w = bon_weights(M, N)
    assert w.shape == (M,)
    assert np.all(w >= 0)
    assert np.isclose(w.sum(), 1.0)


def test_n1_is_the_pool_mean():
    rng = np.random.default_rng(0)
    sel, val = rng.normal(size=50), rng.normal(size=50)
    assert np.isclose(expected_best(sel, val, 1), val.mean())


def test_n_equals_m_picks_the_top_scored():
    rng = np.random.default_rng(1)
    sel, val = rng.normal(size=40), rng.normal(size=40)
    assert np.isclose(expected_best(sel, val, 40), val[np.argmax(sel)])


@pytest.mark.parametrize("N", [1, 2, 3, 4, 6])
def test_matches_brute_force_over_all_subsets(N):
    rng = np.random.default_rng(2)
    M = 6
    sel, val = rng.normal(size=M), rng.normal(size=M)
    subsets = list(itertools.combinations(range(M), N))
    brute = np.mean([val[max(s, key=lambda i: sel[i])] for s in subsets])
    assert np.isclose(expected_best(sel, val, N), brute)


def test_selecting_on_the_value_itself_never_decreases_with_n():
    rng = np.random.default_rng(3)
    val = rng.normal(size=128)
    curve = [expected_best(val, val, n) for n in (1, 2, 4, 8, 16, 32, 64, 128)]
    assert all(a <= b + 1e-12 for a, b in zip(curve, curve[1:]))


def test_invalid_n_raises():
    with pytest.raises(ValueError):
        bon_weights(5, 0)
    with pytest.raises(ValueError):
        bon_weights(5, 6)


def test_kl_is_zero_at_n1_and_grows():
    assert bon_kl(1) == 0
    assert bon_kl(2) < bon_kl(16) < bon_kl(256)
