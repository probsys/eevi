#!/usr/bin/env python

import math

import numpy as np

from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from scipy.stats import wishart
from npeet import entropy_estimators

prng = np.random.RandomState(1)

def logmeanexp(x):
    return logsumexp(x) - math.log(len(x))

def merge_sample(ix, iy, vx, vy):
    n = len(ix) + len(iy)
    L = [None] * n
    for i, v in zip(ix, vx):
        L[i] = v
    for i, v in zip(iy, vy):
        L[i] = v
    return L

def is_pos_def(x):
    return np.all(0 < np.linalg.eigvals(x))

def get_mvn(mu, cov, ix):
    assert is_pos_def(cov)
    return multivariate_normal(mu[ix], cov[np.ix_(ix, ix)])

def entropy_theoretical(cov, ix):
    """Exact MVN entropy."""
    assert is_pos_def(cov)
    assert 0 < len(ix)
    n = len(ix)
    det = np.linalg.det(cov[np.ix_(ix, ix)])
    return (n/2.) * (1 + np.log(2*np.pi)) + .5 * np.log(det)

def neg_entropy_theoretical(cov, ix):
    return -entropy_theoretical(cov, ix)

def entropy_unbiased(mu, cov, ix, N):
    dist_x = get_mvn(mu, cov, ix)
    samples_x = dist_x.rvs(N)
    logp_x = dist_x.logpdf(samples_x)
    return -np.mean(logp_x)

def weights_py_prior(mu, cov, iy, y_obs, M):
    """Return list of importance weights using prior as proposal."""
    n = len(mu)
    ixy = range(n)
    # Sample and assess from proposal q(x; y) = p(x).
    ix = [i for i in ixy if i not in iy]
    dist_x = get_mvn(mu, cov, ix)
    samples_x = np.atleast_2d(dist_x.rvs(M, random_state=prng))
    p_x = dist_x.logpdf(samples_x)
    # Compute probabilities under target p(x, y)
    n = len(mu)
    ixy = list(range(n))
    dist_xy = get_mvn(mu, cov, ixy)
    samples_xy = [merge_sample(ix, iy, sample, y_obs) for sample in samples_x]
    p_xy = dist_xy.logpdf(samples_xy)
    # Return list of importance weights.
    p_y_approx = p_xy - p_x
    return np.atleast_1d(p_y_approx)

def weights_py_iwae(mu, cov, iy, y_obs, M, K):
    """Return list of importance weights using k-IWAE as proposal."""
    weights = []
    for j in range(M):
        w_j = weights_py_prior(mu, cov, iy, y_obs, K)
        weights.append(logmeanexp(w_j))
    return weights

def weights_py_inv_prior(mu, cov, iy, y_obs, ix, x_obs):
    """Return single importance weight using posterior sample and targeting the prior."""
    n = len(mu)
    ixy = range(n)
    # No need to sample from the proposal, we are importance sampling
    # the proposal using one exact posterior sample.
    ix = [i for i in ixy if i not in iy]
    dist_x = get_mvn(mu, cov, ix)
    p_x = dist_x.logpdf(x_obs)
    # Compute probability under target p(x, y)
    dist_xy = get_mvn(mu, cov, ixy)
    sample = merge_sample(ix, iy, x_obs, y_obs)
    p_xy = dist_xy.logpdf(sample)
    # Return single importance weight.
    return [p_x - p_xy]

def weights_py_inv_iwae(mu, cov, iy, y_obs, ix, x_obs, M, K):
    """Return single importance weight using posterior sample and target k-IWAE."""
    assert M == 1
    weights = []
    for j in range(M):
        w_exact = weights_py_inv_prior(mu, cov, iy, y_obs, ix, x_obs)[0]
        w_unbiased = weights_py_prior(mu, cov, iy, y_obs, K-1) if K > 1 else []
        w_list = np.append(w_unbiased, -w_exact)
        w_j = -logmeanexp(w_list)
        weights.append(w_j)
    return weights

def entropy_lower_bound(mu, cov, iy, N, M, K):
    dist_y = get_mvn(mu, cov, iy)
    dist_xy = get_mvn(mu, cov, range(len(mu)))
    samples_xy = dist_xy.rvs(N, random_state=prng)
    log_py_approx = []
    for sample in samples_xy:
        y_obs = sample[iy]
        p_y_approx_weights = weights_py_iwae(mu, cov, iy, y_obs, M, K)
        # p_y_approx = np.mean(p_y_approx_weights)
        # weight = math.log(p_y_approx)
        # Compute overall log importance weight,
        # which is log [1/K sum w_k]
        weight = np.mean(p_y_approx_weights)
        log_py_approx.append(weight)
        # Check.
        # p_y = dist_y.pdf(y_obs)
        # print('%1.4f %1.4f' % (p_y, p_y_approx,))
    return np.mean(log_py_approx)

def entropy_upper_bound(mu, cov, iy, N, M, K):
    assert M == 1
    indexes = np.arange(len(mu))
    ix = [i for i in indexes if i not in iy]
    dist_y = get_mvn(mu, cov, iy)
    dist_xy = get_mvn(mu, cov, indexes)
    samples_xy = dist_xy.rvs(N, random_state=prng)
    log_py_inv_approx = []
    for sample in samples_xy:
        y_obs = sample[iy]
        x_obs = sample[ix]
        p_y_inv_approx_weights = weights_py_inv_iwae(mu, cov, iy, y_obs, ix, x_obs, M, K)
        weight = -np.mean(p_y_inv_approx_weights)
        log_py_inv_approx.append(weight)
    return np.mean(log_py_inv_approx)

def experiment_is(mu, cov, iy, N=200, M=1, K=1):
    # mu = np.asarray([0, 0, 0, 0])
    # cov = np.asarray([
    #     [3  , 2 , 1 , 1] , # 0
    #     [2  , 3 , 1 , 2] , # 1
    #     [1  , 1 , 3 , 1] , # 2
    #     [1  , 2 , 1 , 3] , # 3
    # ])
    # iy = [2,3]
    start = time.time()
    H_lo = entropy_lower_bound(mu, cov, iy, N, M, K)
    H_hi = entropy_upper_bound(mu, cov, iy, N, 1, K)
    elapsed = time.time() - start
    print((H_lo, H_hi, elapsed))

import time
def experiment_kraskov(mu, cov, iy, N=200):
    start = time.time()
    dist_y = get_mvn(mu, cov, iy)
    samples = dist_y.rvs(N, random_state=prng)
    H_est = -entropy_estimators.entropy(samples, k=5, base=np.e)
    elapsed = time.time() - start
    return H_est, elapsed

fnames = ['results/cov.dim@4.txt', 'results/cov.dim@20.txt', 'results/cov.dim@50.txt']
for fname in fnames:
    cov = np.loadtxt(fname, delimiter=',')
    dim = len(cov)
    mu = np.zeros(dim)
    dim = len(cov)
    iy = np.arange(dim/2, dtype=int)
    H_ex = neg_entropy_theoretical(cov, iy)
    H_est, elapsed = experiment_kraskov(mu, cov, iy, N=100)
    print(fname, H_ex, H_est, elapsed)
