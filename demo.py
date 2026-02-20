#!/usr/bin/env python3
"""
pt-compress demo -- benchmark against zlib, bzip2, lzma on various datasets.
"""

import math
import struct
import zlib
import bz2
import lzma
import time
import random
from collections import Counter
from pt_compress import compress, decompress, __version__


def sieve_gaps(limit):
    """Generate prime gaps via simple sieve."""
    s = [True] * (limit + 1)
    s[0] = s[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if s[i]:
            for j in range(i*i, limit + 1, i):
                s[j] = False
    primes = [i for i in range(2, limit + 1) if s[i]]
    return [primes[i+1] - primes[i] for i in range(len(primes) - 1)]


def entropy(data):
    c = Counter(data)
    t = len(data)
    return -sum((n/t) * math.log2(n/t) for n in c.values() if n > 0)


def run(data, label):
    N = len(data)
    mx = max(data)
    mn = min(data)

    # Raw size
    if mx < 256 and mn >= 0:
        raw = N; tag = "uint8"; raw_bytes = bytes(data)
    elif mx < 65536 and mn >= 0:
        raw = N * 2; tag = "uint16"; raw_bytes = struct.pack(f'<{N}H', *data)
    else:
        raw = N * 4; tag = "uint32"; raw_bytes = struct.pack(f'<{N}I', *data)

    # Baselines
    t0 = time.perf_counter()
    zb = zlib.compress(raw_bytes, 9)
    tz = time.perf_counter() - t0

    t0 = time.perf_counter()
    bb = bz2.compress(raw_bytes, 9)
    tb = time.perf_counter() - t0

    t0 = time.perf_counter()
    lb = lzma.compress(raw_bytes, preset=9)
    tl = time.perf_counter() - t0

    # pt-compress
    t0 = time.perf_counter()
    pt = compress(data)
    tp = time.perf_counter() - t0

    # Verify
    assert decompress(pt) == list(data), "ROUND-TRIP FAILED"

    H = entropy(data)
    zs, bs, ls, ps = len(zb), len(bb), len(lb), len(pt)
    best_std = min(zs, bs, ls)

    print(f"\n  {label}")
    print(f"  N={N:,}  max={mx}  H(X)={H:.3f} bits/sym")
    print(f"  {'Method':<20} {'bytes':>10} {'bits/sym':>10} {'time':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8}")
    print(f"  {'raw ('+tag+')':<20} {raw:>10,} {raw*8/N:>10.3f} {'-':>8}")
    print(f"  {'zlib -9':<20} {zs:>10,} {zs*8/N:>10.3f} {tz:>7.3f}s")
    print(f"  {'bzip2 -9':<20} {bs:>10,} {bs*8/N:>10.3f} {tb:>7.3f}s")
    print(f"  {'lzma -9':<20} {ls:>10,} {ls*8/N:>10.3f} {tl:>7.3f}s")
    print(f"  {'pt-compress':<20} {ps:>10,} {ps*8/N:>10.3f} {tp:>7.3f}s")
    print(f"  {'Shannon H(X)':<20} {'':>10} {H:>10.3f} {'-':>8}")

    if ps < best_std:
        gain = (1 - ps / best_std) * 100
        print(f"  >> pt-compress wins by {gain:.1f}%")
    elif ps <= best_std * 1.01:
        print(f"  >> tied")
    else:
        loss = (ps / best_std - 1) * 100
        print(f"  >> best standard wins by {loss:.1f}%")

    return {'zlib': zs, 'bzip2': bs, 'lzma': ls, 'pt': ps, 'raw': raw, 'H': H, 'N': N}


def main():
    print(f"\n  pt-compress v{__version__} -- benchmark")
    print(f"  {'='*58}")

    results = []

    # -- Structured sequences (sieve gaps) --
    print(f"\n  --- Structured sequences ---")
    gaps1 = sieve_gaps(1_000_000)
    results.append(("Prime gaps <10^6", run(gaps1, "Prime gaps (primes < 10^6)")))

    gaps2 = sieve_gaps(10_000_000)
    results.append(("Prime gaps <10^7", run(gaps2, "Prime gaps (primes < 10^7)")))

    # -- Numerical time series --
    print(f"\n  --- Numerical time series ---")
    random.seed(42)

    temps = [20]
    for _ in range(49999):
        temps.append(max(0, min(50, temps[-1] + random.randint(-2, 2))))
    results.append(("Temperature", run(temps, "Temperature (random walk [0,50])")))

    prix = [100]
    for _ in range(49999):
        prix.append(max(1, min(200, prix[-1] + random.choice([-2,-1,-1,0,0,0,1,1,2]))))
    results.append(("Stock prices", run(prix, "Stock prices (random walk)")))

    ecg = [max(0, min(255, int(100 + 30*math.sin(2*math.pi*i/200) + random.gauss(0, 5)))) for i in range(50000)]
    results.append(("ECG signal", run(ecg, "ECG signal (sine + noise)")))

    # -- IID / Random --
    print(f"\n  --- IID / Random ---")
    try:
        import numpy as np
        rng = np.random.default_rng(42)
        rand = list(rng.integers(1, 101, size=50_000))
        results.append(("Random [1,100]", run(rand, "Random uniform [1,100]")))
    except ImportError:
        pass

    biased = [random.choice([0]*7 + [1]*3) for _ in range(50000)]
    results.append(("Bernoulli p=0.3", run(biased, "Bernoulli biased (p=0.3)")))

    # Summary
    print(f"\n  {'='*58}")
    print(f"  SUMMARY")
    print(f"  {'='*58}")
    print(f"\n  {'Dataset':<22} {'pt':>8} {'best std':>10} {'diff':>8} {'Winner':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    wins = 0
    for label, r in results:
        pb = r['pt'] * 8 / r['N']
        best_val = min(r['zlib'], r['bzip2'], r['lzma'])
        sb = best_val * 8 / r['N']
        diff = (1 - r['pt'] / best_val) * 100
        w = "pt" if r['pt'] < best_val else "std"
        if r['pt'] < best_val:
            wins += 1
        print(f"  {label:<22} {pb:>8.3f} {sb:>10.3f} {diff:>+7.1f}% {w:>8}")

    print(f"\n  pt-compress wins {wins}/{len(results)} benchmarks")
    print()


if __name__ == '__main__':
    main()
