# Comparison of how long it takes to produce mean and standard deviations of scattering profiles for (a)
# the RadialProfiler class methods, and (b) the fortran.scatter_f functions.  Fortran is the winner by a factor of
# approximately 30 (on Rick's laptop).
#
# Note that the fortran functions are an improvement over the RadialProfiler methods because they provide weighted
# averages.  If the weights are set to the binary mask values, then the fortran output will be equivalent to the
# RadialProfiler output.

import time
import numpy as np
from reborn import fortran
from reborn import detector
npat = int(1e6)
pattern = np.ones(npat, dtype=float)
weights = np.ones(npat, dtype=float)
weights[0:100] = 0
n_bins = int(100)
q = np.linspace(0, n_bins, npat, dtype=float)
q_min = 0
q_max = n_bins

N = 100

# Calculate the time it takes to fetch standard profile statistics (mean and standard deviation) with fortran functions:
t = time.time()
for i in range(N):
    sum_ = np.zeros(n_bins, dtype=np.float64)
    sum2 = np.zeros(n_bins, dtype=np.float64)
    w_sum = np.zeros(n_bins, dtype=np.float64)
    fortran.scatter_f.profile_stats(pattern, q, weights, n_bins, q_min, q_max, sum_, sum2, w_sum)
    meen = np.empty(n_bins, dtype=np.float64)
    std = np.empty(n_bins, dtype=np.float64)
    fortran.scatter_f.profile_stats_avg(sum_, sum2, w_sum, meen, std)
print(f"profile_stats: {(time.time()-t)/N} seconds.")

# The fortran function can be made faster by a factor of 2 or so if the indices are pre-computed, which we do here:
indices = np.empty(npat, dtype=np.int32)
t = time.time()
for i in range(N):
    fortran.scatter_f.profile_indices(q, n_bins, q_min, q_max, indices)
print(f"profile_indices: {(time.time()-t)/N} seconds.")

# Calculate the time it takes to fetch standard profile statistics with pre-indexed fortran functions:
t = time.time()
for i in range(N):
    sum_ = np.zeros(n_bins, dtype=np.float64)
    sum2 = np.zeros(n_bins, dtype=np.float64)
    w_sum = np.zeros(n_bins, dtype=np.float64)
    fortran.scatter_f.profile_stats_indexed(pattern, indices, weights, sum_, sum2, w_sum)
print(f"profile_stats_indexed: {(time.time()-t)/N} seconds.")

# Calculate the time it takes to fetch stanard profile statistics with RadialProfiler methods:
profiler = detector.RadialProfiler(q_mags=q, mask=weights, n_bins=n_bins, q_range=[q_min, q_max])
for i in range(N):
    pmeen = profiler.get_mean_profile(pattern)
    pstd = profiler.get_sdev_profile(pattern)
print(f"RadialProfiler: {(time.time()-t)/N} seconds.")

# Check that the results really are the same:
assert(np.max(np.abs(meen-pmeen)) == 0)
assert(np.max(np.abs(std-pstd)) == 0)
print("Output is identical for integer weights")
