"""
ZWave Sort — Benchmarks, correctness tests, and stats report.
Run from the ZSort root: python benchmark/benchmark.py

Produces:
  - benchmark.png    — performance curves per input pattern
  - comparison.png   — bar chart of all algorithms at n=200,000
  - patterns_all.png — grouped bar chart across all patterns
  - stats.txt        — full report for external comparison
"""

import os
import sys
import time
import random
import platform
import datetime

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zsort import sort as zwavesort

try:
    import psutil
    _RAM = f"{psutil.virtual_memory().available / 1e9:.1f} GB available"
except ImportError:
    _RAM = "unknown (psutil not installed)"

try:
    import numba
    _NUMBA = numba.__version__
except ImportError:
    _NUMBA = "not installed"


SIZES        = [500, 2_000, 10_000, 50_000, 200_000, 1_000_000]
PATTERNS     = ["random", "sorted", "reversed", "few_unique", "nearly_sorted", "pipe_organ"]
RUNS         = 3
COMPARE_N    = 200_000
COMPARE_RUNS = 5
VERIFY_N     = 10_000
STATS_FILE   = os.path.join(os.path.dirname(__file__), "stats.txt")


def numpy_quicksort(data):
    return np.sort(np.asarray(data, dtype=np.int64), kind="quicksort")

def numpy_mergesort(data):
    return np.sort(np.asarray(data, dtype=np.int64), kind="mergesort")

def numpy_heapsort(data):
    return np.sort(np.asarray(data, dtype=np.int64), kind="heapsort")

def numpy_stable(data):
    return np.sort(np.asarray(data, dtype=np.int64), kind="stable")

def python_sorted(data):
    return sorted(data)


@jit(nopython=True, fastmath=True, cache=True)
def _qsort(arr, lo, hi):
    while lo < hi:
        pivot = arr[(lo + hi) >> 1]
        i, j = lo, hi
        while i <= j:
            while arr[i] < pivot: i += 1
            while arr[j] > pivot: j -= 1
            if i <= j:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
                j -= 1
        if j - lo < hi - i:
            _qsort(arr, lo, j)
            lo = i
        else:
            _qsort(arr, i, hi)
            hi = j

def numba_quicksort(data):
    arr = np.asarray(data, dtype=np.int64).copy()
    _qsort(arr, 0, len(arr) - 1)
    return arr


@jit(nopython=True, fastmath=True, cache=True)
def _radix_sort(arr):
    n   = len(arr)
    out = np.empty(n, dtype=np.int64)
    tmp = arr.copy()
    for i in range(n):
        tmp[i] = arr[i] ^ np.int64(-9223372036854775808)
    for shift in range(0, 64, 8):
        counts = np.zeros(256, dtype=np.int64)
        for i in range(n):
            counts[(tmp[i] >> shift) & 0xFF] += 1
        prefix = np.zeros(256, dtype=np.int64)
        for i in range(1, 256):
            prefix[i] = prefix[i - 1] + counts[i - 1]
        for i in range(n):
            b = int((tmp[i] >> shift) & 0xFF)
            out[prefix[b]] = tmp[i]
            prefix[b] += 1
        tmp, out = out, tmp
    for i in range(n):
        tmp[i] = tmp[i] ^ np.int64(-9223372036854775808)
    return tmp

def numba_radixsort(data):
    arr = np.asarray(data, dtype=np.int64).copy()
    return _radix_sort(arr)


COMPETITORS = {
    "ZWave Sort"       : zwavesort,
    "numpy quicksort"  : numpy_quicksort,
    "numpy mergesort"  : numpy_mergesort,
    "numpy heapsort"   : numpy_heapsort,
    "numpy stable"     : numpy_stable,
    "Numba quicksort"  : numba_quicksort,
    "Numba radix sort" : numba_radixsort,
    "Python sorted()"  : python_sorted,
}


def make(n, pattern):
    if pattern == "random":
        return random.sample(range(n * 10), n)
    if pattern == "sorted":
        return list(range(n))
    if pattern == "reversed":
        return list(range(n, 0, -1))
    if pattern == "few_unique":
        return [random.randint(0, 9) for _ in range(n)]
    if pattern == "nearly_sorted":
        arr = list(range(n))
        for _ in range(max(1, n // 20)):
            i, j = random.randrange(n), random.randrange(n)
            arr[i], arr[j] = arr[j], arr[i]
        return arr
    if pattern == "pipe_organ":
        half = n // 2
        return list(range(half)) + list(range(half, 0, -1))
    raise ValueError(f"Unknown pattern: {pattern}")


def _time_fn(fn, data, runs):
    best = float("inf")
    for _ in range(runs):
        d  = list(data)
        t0 = time.perf_counter()
        fn(d)
        best = min(best, time.perf_counter() - t0)
    return best


def verify(n=VERIFY_N):
    results = {}
    print(f"Correctness  (n={n:,})")
    print("─" * 36)
    for p in PATTERNS:
        data   = make(n, p)
        result = zwavesort(data)
        ok     = list(result) == sorted(data)
        results[p] = ok
        print(f"  {p:<16}  {'PASS' if ok else '!! FAIL !!'}")
    print()
    return results


def benchmark_zwavesort(sizes=SIZES, patterns=PATTERNS, runs=RUNS):
    results = {p: [] for p in patterns}
    print(f"ZWave Sort performance  (best of {runs} runs)")
    print("─" * 52)
    for p in patterns:
        print(f"  {p}")
        for n in sizes:
            data = make(n, p)
            t    = _time_fn(zwavesort, data, runs)
            results[p].append(t)
            print(f"    n={n:>9,}   {t * 1_000:8.3f} ms   "
                  f"({n / t / 1e6:.2f} M el/s)")
        print()
    return results


def benchmark_comparison(n=COMPARE_N, runs=COMPARE_RUNS):
    data  = make(n, "random")
    times = {}

    print(f"Extended Comparison  n={n:,}  pattern=random  (best of {runs})")
    print("─" * 62)

    for name, fn in COMPETITORS.items():
        t = _time_fn(fn, data, runs)
        times[name] = t

    t_zwv = times["ZWave Sort"]
    for name, t in times.items():
        rel = ("baseline" if name == "ZWave Sort"
               else f"{t / t_zwv:.2f}x slower" if t > t_zwv
               else f"{t_zwv / t:.2f}x faster than ZWave")
        print(f"  {name:<22}  {t * 1_000:8.3f} ms   {t / t_zwv * 100:6.1f}%   {rel}")
    print()
    return times


def benchmark_patterns_all(n=COMPARE_N, runs=3):
    results = {name: {} for name in COMPETITORS}
    print(f"All-pattern comparison  n={n:,}  (best of {runs})")
    print("─" * 62)
    for p in PATTERNS:
        print(f"  Pattern: {p}")
        data = make(n, p)
        for name, fn in COMPETITORS.items():
            t = _time_fn(fn, data, runs)
            results[name][p] = t
            print(f"    {name:<22}  {t * 1_000:8.3f} ms")
        print()
    return results


def plot_zwave(sizes, results, out="benchmark.png"):
    fig, ax = plt.subplots(figsize=(12, 6))
    for pattern, times in results.items():
        ax.plot(sizes, [t * 1_000 for t in times], marker="o", label=pattern)
    ax.set_xlabel("Array size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("ZWave Sort  —  performance by input pattern")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Plot saved → {out}\n")
    plt.show()


def plot_comparison(compare_times, n=COMPARE_N, out="comparison.png"):
    names  = list(compare_times.keys())
    times  = [compare_times[n_] * 1_000 for n_ in names]
    order  = sorted(range(len(times)), key=lambda i: times[i])
    names  = [names[i] for i in order]
    times  = [times[i] for i in order]

    colors = ["#2ecc71" if n == "ZWave Sort" else "#3498db" for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, times, color=colors)
    ax.bar_label(bars, fmt="%.2f ms", padding=4, fontsize=9)
    ax.set_xlabel("Time (ms)  — lower is better")
    ax.set_title(f"Sorting Algorithm Comparison  —  random data, n={n:,}")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Comparison plot saved → {out}\n")
    plt.show()


def plot_patterns_all(pattern_results, n=COMPARE_N, out="patterns_all.png"):
    algs     = list(pattern_results.keys())
    patterns = PATTERNS
    x        = np.arange(len(patterns))
    width    = 0.8 / len(algs)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, alg in enumerate(algs):
        vals = [pattern_results[alg][p] * 1_000 for p in patterns]
        ax.bar(x + i * width, vals, width, label=alg)

    ax.set_xticks(x + width * (len(algs) - 1) / 2)
    ax.set_xticklabels(patterns, rotation=15, ha="right")
    ax.set_ylabel("Time (ms)  — lower is better")
    ax.set_title(f"All Algorithms × All Patterns  —  n={n:,}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Patterns plot saved → {out}\n")
    plt.show()


def _col_w(sizes):
    return max(9, max(len(f"{n:,}") + 2 for n in sizes))


def write_stats(correctness, zwave_results, compare_times, pattern_results,
                sizes=SIZES, patterns=PATTERNS, runs=RUNS,
                compare_n=COMPARE_N, path=STATS_FILE):

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cw  = _col_w(sizes)

    def _header(title):
        return "\n" + "=" * 80 + f"\n  {title}\n" + "=" * 80

    def _divider():
        return "─" * 80

    lines = []

    lines += [
        "=" * 80,
        "  ZWave Sort — Benchmark Report",
        f"  Generated : {now}",
        "=" * 80,
        "",
        "  SYSTEM",
        f"    OS           : {platform.system()} {platform.release()}",
        f"    CPU cores    : {os.cpu_count()}",
        f"    RAM          : {_RAM}",
        f"    Python       : {sys.version.split()[0]}",
        f"    NumPy        : {np.__version__}",
        f"    Numba        : {_NUMBA}",
    ]

    lines += [_header(f"CORRECTNESS  (n = {VERIFY_N:,})")]
    for p, ok in correctness.items():
        lines.append(f"  {p:<20}  {'PASS' if ok else '!! FAIL !!'}")

    lines += [_header(f"ZWAVESORT PERFORMANCE  (ms, best of {runs} runs)")]
    header = f"  {'Pattern':<18}" + "".join(f"{n:>{cw},}" for n in sizes)
    lines.append(header)
    lines.append("  " + _divider()[:len(header) - 2])
    for p in patterns:
        row = f"  {p:<18}"
        for t in zwave_results[p]:
            row += f"{t * 1_000:>{cw}.3f}"
        lines.append(row)

    lines += [_header("THROUGHPUT  (million elements / second)")]
    lines.append(header)
    lines.append("  " + _divider()[:len(header) - 2])
    for p in patterns:
        row = f"  {p:<18}"
        for n, t in zip(sizes, zwave_results[p]):
            row += f"{n / t / 1e6:>{cw}.2f}"
        lines.append(row)

    lines += [_header("ZWAVESORT RAW TIMES  (seconds, for external comparison)")]
    lines.append(header)
    lines.append("  " + _divider()[:len(header) - 2])
    for p in patterns:
        row = f"  {p:<18}"
        for t in zwave_results[p]:
            row += f"{t:>{cw}.6f}"
        lines.append(row)

    t_zwv = compare_times["ZWave Sort"]
    lines += [
        _header(f"EXTENDED COMPARISON  (random data, n = {compare_n:,}, best of {COMPARE_RUNS} runs)"),
        f"  {'Algorithm':<24} {'Time (ms)':>10}   {'vs ZWave':>8}   Notes",
        "  " + _divider()[:76],
    ]
    for name, t in compare_times.items():
        rel = ("baseline" if name == "ZWave Sort"
               else f"{t / t_zwv:.2f}x slower" if t > t_zwv
               else f"{t_zwv / t:.2f}x faster than ZWave")
        lines.append(
            f"  {name:<24} {t * 1_000:>10.3f}   {t / t_zwv * 100:>7.2f}%   {rel}"
        )

    lines += [
        _header(f"ALL-PATTERN COMPARISON  (ms, n = {compare_n:,})"),
        f"  {'Algorithm':<24}" + "".join(f"  {p[:12]:>14}" for p in patterns),
        "  " + _divider(),
    ]
    for name in pattern_results:
        row = f"  {name:<24}"
        for p in patterns:
            row += f"  {pattern_results[name][p] * 1_000:>14.3f}"
        lines.append(row)

    best_p  = min(patterns, key=lambda p: zwave_results[p][-1])
    worst_p = max(patterns, key=lambda p: zwave_results[p][-1])
    best_t  = zwave_results[best_p][-1]
    worst_t = zwave_results[worst_p][-1]

    lines += [
        _header("SUMMARY"),
        f"  Tested patterns    : {', '.join(patterns)}",
        f"  Fastest pattern    : {best_p}  ({best_t * 1_000:.3f} ms at n={sizes[-1]:,})",
        f"  Slowest pattern    : {worst_p}  ({worst_t * 1_000:.3f} ms at n={sizes[-1]:,})",
        f"  Speedup range      : {worst_t / best_t:.1f}x  (best vs worst at n={sizes[-1]:,})",
        "",
        "  How to use these numbers for external comparison:",
        "    - Use the 'RAW TIMES' section (seconds) to compare against any",
        "      other algorithm on the same machine with the same input sizes.",
        "    - The 'few_unique' and 'sorted'/'reversed' times reflect O(n)",
        "      fast-exit paths, not the ZWave core algorithm.",
        "    - For a fair apples-to-apples comparison use the 'random' row.",
        "",
        "=" * 80,
    ]

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Stats saved → {path}\n")
    return text


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("Warming up JIT...")
    _warm  = list(range(1_000))
    _warm2 = random.sample(range(10_000), 1_000)
    zwavesort(_warm)
    zwavesort(_warm2)
    numba_quicksort(_warm)
    numba_radixsort(_warm)
    print("Done.\n")

    correctness = verify()
    if not all(correctness.values()):
        print("Correctness check failed — aborting.")
        raise SystemExit(1)

    zwave_results   = benchmark_zwavesort()
    compare_times   = benchmark_comparison()
    pattern_results = benchmark_patterns_all()

    stats_text = write_stats(correctness, zwave_results, compare_times, pattern_results)
    print(stats_text)

    plot_zwave(SIZES, zwave_results, out=os.path.join(out_dir, "benchmark.png"))
    plot_comparison(compare_times,   out=os.path.join(out_dir, "comparison.png"))
    plot_patterns_all(pattern_results, out=os.path.join(out_dir, "patterns_all.png"))
