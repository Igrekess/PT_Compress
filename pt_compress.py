#!/usr/bin/env python3
"""
pt-compress -- Lossless compression for integer sequences.
Based on Persistence Theory by Yan Senez.

Usage:
    from pt_compress import compress, decompress

    data = [2, 4, 2, 4, 6, 2, 6, 4, ...]
    blob = compress(data)
    assert decompress(blob) == data
"""

import struct
import math

__version__ = "0.3.0"
__author__ = "Yan Senez"

# -- Arithmetic coding engine ------------------------------------------------

_P = 32
_W = 1 << _P
_H = 1 << (_P - 1)
_Q = 1 << (_P - 2)


class _Enc:
    __slots__ = ('lo', 'hi', 'pend', 'bits')

    def __init__(self):
        self.lo = 0
        self.hi = _W - 1
        self.pend = 0
        self.bits = []

    def put(self, cf, sf, tf):
        r = self.hi - self.lo + 1
        self.hi = self.lo + (r * (cf + sf)) // tf - 1
        self.lo = self.lo + (r * cf) // tf
        while True:
            if self.hi < _H:
                self._out(0)
            elif self.lo >= _H:
                self._out(1)
                self.lo -= _H
                self.hi -= _H
            elif self.lo >= _Q and self.hi < 3 * _Q:
                self.pend += 1
                self.lo -= _Q
                self.hi -= _Q
            else:
                break
            self.lo <<= 1
            self.hi = (self.hi << 1) | 1

    def _out(self, b):
        self.bits.append(b)
        for _ in range(self.pend):
            self.bits.append(1 - b)
        self.pend = 0

    def done(self):
        self.pend += 1
        self._out(0 if self.lo < _Q else 1)
        n = len(self.bits)
        buf = bytearray()
        for i in range(0, n, 8):
            v = 0
            for j in range(8):
                if i + j < n:
                    v = (v << 1) | self.bits[i + j]
                else:
                    v <<= 1
            buf.append(v)
        return bytes(buf), n


class _Dec:
    __slots__ = ('d', 'nb', 'bp', 'lo', 'hi', 'val')

    def __init__(self, d, nb):
        self.d = d
        self.nb = nb
        self.bp = 0
        self.lo = 0
        self.hi = _W - 1
        self.val = 0
        for _ in range(_P):
            self.val = (self.val << 1) | self._rb()

    def _rb(self):
        if self.bp < self.nb:
            i = self.bp >> 3
            b = 7 - (self.bp & 7)
            self.bp += 1
            return (self.d[i] >> b) & 1 if i < len(self.d) else 0
        return 0

    def get(self, tbl, tf):
        r = self.hi - self.lo + 1
        sc = ((self.val - self.lo + 1) * tf - 1) // r
        s = len(tbl) - 1
        for i, (c, f) in enumerate(tbl):
            if c + f > sc:
                s = i
                break
        c, f = tbl[s]
        self.hi = self.lo + (r * (c + f)) // tf - 1
        self.lo = self.lo + (r * c) // tf
        while True:
            if self.hi < _H:
                pass
            elif self.lo >= _H:
                self.lo -= _H
                self.hi -= _H
                self.val -= _H
            elif self.lo >= _Q and self.hi < 3 * _Q:
                self.lo -= _Q
                self.hi -= _Q
                self.val -= _Q
            else:
                break
            self.lo <<= 1
            self.hi = (self.hi << 1) | 1
            self.val = (self.val << 1) | self._rb()
        return s


# -- Adaptive context model --------------------------------------------------

class _Ctx:
    __slots__ = ('_a', '_q', '_o', '_mx', '_t')

    def __init__(self, alpha, order=3, q=3):
        self._a = alpha
        self._q = q
        self._o = order
        self._mx = 1 << 14
        self._t = []
        for o in range(order + 1):
            nc = q ** o
            self._t.append([[1] * alpha for _ in range(nc)])

    def _k(self, h, o):
        if o == 0:
            return 0
        if len(h) < o:
            return -1
        k = 0
        for i in range(o):
            k = k * self._q + (h[-(i + 1)] % self._q)
        return k

    def freq(self, h):
        for o in range(self._o, -1, -1):
            k = self._k(h, o)
            if k < 0:
                continue
            f = self._t[o][k]
            t = sum(f)
            if t >= self._a * 2 or o == 0:
                return f, t
        f = self._t[0][0]
        return f, sum(f)

    def update(self, s, h):
        for o in range(self._o + 1):
            k = self._k(h, o)
            if k < 0:
                continue
            self._t[o][k][s] += 1
            if sum(self._t[o][k]) > self._mx:
                self._t[o][k] = [max(v >> 1, 1) for v in self._t[o][k]]


# -- Core algorithm -----------------------------------------------------------

_MAGIC = b'PT'
_VER = 4


def _order(N, alpha, q):
    if alpha > 2000:
        return 1
    if alpha > 500:
        return 2
    o = 0
    while q ** (o + 1) * alpha * 2 < N and o < 5:
        o += 1
    return o


def _entropy(data):
    """Shannon entropy in bits/symbol."""
    N = len(data)
    if N == 0:
        return 0.0
    counts = {}
    for x in data:
        counts[x] = counts.get(x, 0) + 1
    return -sum((c / N) * math.log2(c / N) for c in counts.values())


def _encode(tr, N, mn, pf, df, do_val, q):
    """Compress preprocessed sequence with given q. Returns bytes."""
    alpha = max(tr) + 1
    order = _order(N, alpha, q)
    model = _Ctx(alpha, order, q)
    enc = _Enc()
    hist = []
    for s in tr:
        f, t = model.freq(hist)
        cf = 0
        for j in range(s):
            cf += f[j]
        enc.put(cf, f[s], t)
        model.update(s, hist)
        hist.append(s)
        if len(hist) > order:
            hist = hist[-order:]
    stream, nb = enc.done()
    hdr = bytearray(_MAGIC)
    hdr += struct.pack('<BIBB', _VER, N, pf, df)
    hdr += struct.pack('<iIi', mn, alpha, do_val)
    hdr += struct.pack('<BB', order, q)
    hdr += struct.pack('<II', nb, len(stream))
    return bytes(hdr) + stream


def compress(data):
    """Compress a list of non-negative integers. Returns bytes."""
    data = list(data)
    N = len(data)
    if N == 0:
        return _MAGIC + struct.pack('<BI', _VER, 0)

    mn = min(data)
    sh = [x - mn for x in data]

    even = all(x % 2 == 0 for x in sh)
    if even and max(sh) > 0:
        base = [x >> 1 for x in sh]
        pf = 1
    else:
        base = sh
        pf = 0

    # Prepare delta variant (gauge connection: encode transitions)
    d = [base[0]]
    for i in range(1, len(base)):
        d.append(base[i] - base[i - 1])
    md = min(d)
    delta = [v - md for v in d]

    # Candidate q values for a given transformed sequence
    def _qs(tr):
        alpha = max(tr) + 1
        qs = {3}
        if alpha <= 64:
            qs.add(alpha)
        elif alpha <= 128:
            qs.add(64)
        return sorted(qs)

    # Try all (preprocessing, q) combinations, keep the smallest
    best = None
    for tr, df_flag, do_val in [(base, 0, 0), (delta, 1, md)]:
        for q in _qs(tr):
            blob = _encode(tr, N, mn, pf, df_flag, do_val, q)
            if best is None or len(blob) < len(best):
                best = blob
    return best


def decompress(blob):
    """Decompress bytes back to a list of integers."""
    pos = 0
    assert blob[pos:pos + 2] == _MAGIC
    pos += 2
    ver = struct.unpack_from('<B', blob, pos)[0]; pos += 1
    N = struct.unpack_from('<I', blob, pos)[0]; pos += 4
    if N == 0:
        return []
    pf = struct.unpack_from('<B', blob, pos)[0]; pos += 1
    df = struct.unpack_from('<B', blob, pos)[0]; pos += 1
    mn = struct.unpack_from('<i', blob, pos)[0]; pos += 4
    alpha = struct.unpack_from('<I', blob, pos)[0]; pos += 4
    do = struct.unpack_from('<i', blob, pos)[0]; pos += 4
    order = struct.unpack_from('<B', blob, pos)[0]; pos += 1
    q = struct.unpack_from('<B', blob, pos)[0]; pos += 1
    nb, sl = struct.unpack_from('<II', blob, pos); pos += 8
    stream = blob[pos:pos + sl]

    model = _Ctx(alpha, order, q)
    dec = _Dec(stream, nb)
    hist = []
    tr = []
    for _ in range(N):
        f, t = model.freq(hist)
        tbl = []
        cf = 0
        for j in range(alpha):
            tbl.append((cf, f[j]))
            cf += f[j]
        s = dec.get(tbl, t)
        model.update(s, hist)
        hist.append(s)
        if len(hist) > order:
            hist = hist[-order:]
        tr.append(s)

    if df:
        tr = [t + do for t in tr]
        acc = [tr[0]]
        for i in range(1, len(tr)):
            acc.append(acc[-1] + tr[i])
        tr = acc
    if pf:
        tr = [x << 1 for x in tr]
    return [x + mn for x in tr]


# -- CLI ----------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        from demo import main
        main()
    else:
        print(f"pt-compress v{__version__}")
        print("Usage: python pt_compress.py demo")
        print("   or: from pt_compress import compress, decompress")
