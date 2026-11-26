"""Exact best-of-N from one pool of M candidates.

Pick N candidates at random from the pool and keep the one with the highest selection
score. The candidate ranked i (1 = lowest selection score, M = highest) is the winner
exactly when it is in the subset and the other N - 1 come from the i - 1 below it:

    P(rank i wins) = C(i - 1, N - 1) / C(M, N)

So E[value of the winner] = sum_i P(rank i wins) * value_i, with no subset sampling.
N = 1 gives the pool mean; N = M gives the value of the top-scored candidate.
"""
import numpy as np
from scipy.special import gammaln


def log_comb(n, k):
    n, k = np.asarray(n, float), np.asarray(k, float)
    valid = (k >= 0) & (k <= n)
    n, k = np.where(valid, n, 0.0), np.where(valid, k, 0.0)
    out = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return np.where(valid, out, -np.inf)


def bon_weights(M, N):
    """P(rank i wins) for ranks i = 1..M in ascending order of selection score."""
    if not 1 <= N <= M:
        raise ValueError(f"need 1 <= N <= M, got N={N}, M={M}")
    i = np.arange(1, M + 1)
    return np.exp(log_comb(i - 1, N - 1) - log_comb(M, N))


def expected_best(select, value, N):
    """Expected `value` of the candidate with the highest `select` score among N random ones."""
    select, value = np.asarray(select, float), np.asarray(value, float)
    order = np.argsort(select, kind="stable")
    return float(bon_weights(len(select), N) @ value[order])


def bon_kl(N):
    """KL divergence of best-of-N from the base distribution (Gao et al. 2022)."""
    return np.log(N) - (N - 1) / N
