# pt-compress

**250 lines of Python. Zero dependencies. Beats zlib, bzip2, and lzma.**

Lossless compression for integer sequences that exploits hidden arithmetic structure.

```
pt-compress   3.62 bits/sym   <<<
bzip2 -9      3.85 bits/sym
Shannon H(X)  3.91 bits/sym
lzma -9       3.96 bits/sym
zlib -9       4.14 bits/sym
```

That's not a typo. pt-compress goes *below* the marginal Shannon entropy, and beats every standard compressor.

## Try it

```python
from pt_compress import compress, decompress

data = [2, 4, 2, 4, 6, 2, 6, 4, 2, 4, 6, 6, 2, 6, 4, 2]
blob = compress(data)
recovered = decompress(blob)

assert recovered == data  # lossless
print(f"{len(data)} values -> {len(blob)} bytes")
```

```bash
python demo.py   # full benchmark
```

## Benchmark

### Structured sequences (sieve gaps)

| Dataset | pt-compress | bzip2 -9 | lzma -9 | zlib -9 | Shannon H(X) |
|---|---|---|---|---|---|
| Prime gaps < 10^6 (78K) | **3.62** | 3.85 | 3.96 | 4.14 | 3.91 |
| Prime gaps < 10^7 (664K) | **3.69** | 4.06 | 4.14 | 4.39 | 4.17 |
| Lucky number gaps (16K) | **3.55** | 3.97 | - | - | 3.84 |

pt-compress beats bzip2 by 6-11%, lzma by 9-11%, zlib by 13-16%.

### Numerical time series

| Dataset | pt-compress | Best standard | Gain |
|---|---|---|---|
| Stock prices (random walk) | **2.27** | bzip2 = 2.77 | **+18%** |
| Temperature (random walk) | **2.38** | bzip2 = 2.56 | **+7%** |
| ECG signal (sine + noise) | **4.85** | bzip2 = 5.05 | **+4%** |
| Bernoulli (biased coin) | **0.89** | lzma = 1.07 | **+17%** |

### Where it doesn't win

| Dataset | pt-compress | Best standard | Note |
|---|---|---|---|
| Random uniform [1,100] | 6.76 | bzip2 = 6.72 | Tied (no structure) |
| Text / byte streams | 4.20 | bzip2 = 2.27 | BWT wins on repeated words |
| Perfect periodic patterns | 0.015 | bzip2 = 0.009 | LZ77 wins on exact repeats |

## How?

The algorithm is derived from **Persistence Theory**, a mathematical framework that studies informational persistence in number-theoretic sequences.

Two ideas from the theory:
1. **Modular context** -- the residue class of previous values predicts the next value
2. **Gauge connection** -- transitions between consecutive values carry the compressible signal

The compressor automatically selects the best modular resolution and preprocessing for each dataset.

The code is ~250 lines. Read it.

## Requirements

Python 3.10+, nothing else.

## License

MIT

## Author

Yan Senez -- 2026
[Persistence Theory]([Persistence Theory](https://github.com/Igrekess/PersistenceTheory)
)


