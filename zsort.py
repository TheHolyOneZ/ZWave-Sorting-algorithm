"""
ZWave Sort  —  v1.1
Adaptive sorting algorithm by Etienne (TheHolyOneZ)

PROBE (sample CDF) → SCATTER (place into zones) → PATCH (fix boundary errors)
Average O(n log √n), best O(n), worst O(n log n) with timsort fallback.
"""

import os
import numpy as np
from numba import jit, prange

_SMALL         = 32
_ZONE_TARGET   = 8
_PARALLEL_MIN  = 32_768
_CPU           = os.cpu_count() or 4
_FALLBACK_RATE = 0.04


@jit(nopython=True, fastmath=True, cache=True)
def _insertion(arr, lo, hi):
    for i in range(lo + 1, hi + 1):
        key = arr[i]
        j   = i - 1
        while j >= lo and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


@jit(nopython=True, fastmath=True, cache=True)
def _merge_into(arr, buf, lo, mid, hi):
    buf[lo : hi + 1] = arr[lo : hi + 1]
    i, j, k = lo, mid + 1, lo
    while i <= mid and j <= hi:
        c       = buf[i] <= buf[j]
        arr[k]  = buf[i] if c else buf[j]
        i      += c
        j      += 1 - c
        k      += 1
    if i <= mid:
        arr[k : hi + 1] = buf[i : mid + 1]
    else:
        arr[k : hi + 1] = buf[j : hi + 1]


@jit(nopython=True, fastmath=True, cache=True)
def _timsort(arr):
    n     = len(arr)
    BLOCK = 32
    if n <= BLOCK:
        _insertion(arr, 0, n - 1)
        return
    buf = np.empty_like(arr)
    i = 0
    while i < n:
        _insertion(arr, i, min(i + BLOCK - 1, n - 1))
        i += BLOCK
    size = BLOCK
    while size < n:
        i = 0
        while i < n:
            mid = i + size - 1
            if mid >= n:
                break
            hi = min(i + 2 * size - 1, n - 1)
            _merge_into(arr, buf, i, mid, hi)
            i += 2 * size
        size *= 2


@jit(nopython=True, fastmath=True, cache=True)
def _counting(arr, mn, mx):
    rng    = mx - mn + 1
    counts = np.zeros(rng, dtype=np.int64)
    for v in arr:
        counts[v - mn] += 1
    out = np.empty_like(arr)
    pos = 0
    for i in range(rng):
        for _ in range(counts[i]):
            out[pos] = i + mn
            pos     += 1
    return out


@jit(nopython=True, fastmath=True, cache=True)
def _quick_scan(arr):
    n    = len(arr)
    mn   = arr[0]
    mx   = arr[0]
    asc  = True
    desc = True

    for i in range(1, n):
        v = arr[i]
        if v < mn: mn = v
        if v > mx: mx = v
        if asc  and v < arr[i - 1]: asc  = False
        if desc and v > arr[i - 1]: desc = False

    sz     = min(256, n)
    step   = max(1, n // sz)
    sample = np.empty(sz, dtype=arr.dtype)
    for i in range(sz):
        sample[i] = arr[i * step]
    low_e = len(np.unique(sample)) * 10 < sz

    return mn, mx, asc, desc, low_e


@jit(nopython=True, fastmath=True, cache=True)
def _build_cdf(arr):
    n      = len(arr)
    k      = max(64, int(n ** 0.55))
    k      = min(k, n)
    step   = max(1, n // k)
    sample = np.empty(k, dtype=arr.dtype)
    for i in range(k):
        sample[i] = arr[i * step]
    sample.sort()
    return sample


@jit(nopython=True, fastmath=True, cache=True)
def _predict_zone(v, cdf, K):
    """Interpolation-assisted binary search on the CDF to find a zone index."""
    k  = len(cdf)
    mn = cdf[0]
    mx = cdf[k - 1]

    if v <= mn: return 0
    if v >= mx: return K - 1

    frac  = (v - mn) / (mx - mn)
    guess = int(frac * k)
    if guess >= k: guess = k - 1

    w  = max(4, k >> 5)
    lo = guess - w if guess > w else 0
    hi = (guess + w + 1) if (guess + w + 1) < k else k

    if lo > 0 and cdf[lo] > v:      lo = 0
    if hi < k and cdf[hi - 1] <= v: hi = k

    while lo < hi:
        mid = (lo + hi) >> 1
        if cdf[mid] <= v:
            lo = mid + 1
        else:
            hi = mid

    zone = int(lo * K / k)
    return min(zone, K - 1)


@jit(nopython=True, fastmath=True, cache=True)
def _patch(arr):
    n = len(arr)
    i = 1
    while i < n:
        if arr[i] < arr[i - 1]:
            key = arr[i]
            j   = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        i += 1


@jit(nopython=True, fastmath=True, cache=True)
def _inversion_rate(arr):
    n    = len(arr)
    sz   = min(512, n - 1)
    step = max(1, (n - 1) // sz)
    inv  = 0
    for i in range(sz):
        idx = i * step
        if arr[idx + 1] < arr[idx]:
            inv += 1
    return inv / sz


@jit(nopython=True, fastmath=True, cache=True)
def _zwave(arr, cdf):
    n = len(arr)
    K = max(1, n // _ZONE_TARGET)

    zones  = np.empty(n, dtype=np.int64)
    counts = np.zeros(K,  dtype=np.int64)
    for i in range(n):
        z        = _predict_zone(arr[i], cdf, K)
        zones[i] = z
        counts[z] += 1

    starts = np.zeros(K + 1, dtype=np.int64)
    for i in range(K):
        starts[i + 1] = starts[i] + counts[i]

    out = np.empty_like(arr)
    pos = starts[:K].copy()
    for i in range(n):
        z           = zones[i]
        out[pos[z]] = arr[i]
        pos[z]     += 1

    for i in range(K):
        lo = starts[i]
        hi = starts[i + 1] - 1
        if hi > lo:
            _insertion(out, lo, hi)

    for i in range(n):
        arr[i] = out[i]
    _patch(arr)


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def _sort_chunks_parallel(arr, chunk_size):
    n  = len(arr)
    nc = (n + chunk_size - 1) // chunk_size
    for i in prange(nc):
        lo  = i * chunk_size
        hi  = min(lo + chunk_size - 1, n - 1)
        seg = arr[lo : hi + 1].copy()
        cdf = _build_cdf(seg)
        _zwave(seg, cdf)
        arr[lo : hi + 1] = seg


@jit(nopython=True, fastmath=True, cache=True)
def _merge_chunks(arr, chunk_size):
    n    = len(arr)
    buf  = np.empty_like(arr)
    size = chunk_size
    while size < n:
        i = 0
        while i < n:
            mid = i + size - 1
            if mid >= n:
                break
            hi = min(i + 2 * size - 1, n - 1)
            _merge_into(arr, buf, i, mid, hi)
            i += 2 * size
        size *= 2


def sort(data):
    """
    Sort an integer array using ZWave Sort.

    Parameters
    ----------
    data : list or np.ndarray of integers

    Returns
    -------
    np.ndarray  —  sorted copy, dtype int64
    """
    arr = np.asarray(data, dtype=np.int64)
    n   = len(arr)

    if n <= 1:
        return arr.copy()

    if n <= _SMALL:
        out = arr.copy()
        _insertion(out, 0, n - 1)
        return out

    mn, mx, is_sorted, is_reversed, low_entropy = _quick_scan(arr)

    if is_sorted:   return arr.copy()
    if is_reversed: return arr[::-1].copy()
    if low_entropy: return _counting(arr, mn, mx)

    work = arr.copy()

    if n >= _PARALLEL_MIN:
        chunk_size = max(8_192, n // _CPU)
        _sort_chunks_parallel(work, chunk_size)
        _merge_chunks(work, chunk_size)
    else:
        cdf = _build_cdf(work)
        _zwave(work, cdf)

    if _inversion_rate(work) > _FALLBACK_RATE:
        _timsort(work)

    return work
