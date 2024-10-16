import jax
import jax.numpy as jnp
import numpy as np
import mmh3
from constants import RAW_ARRAYS, BIAS_ARRAYS

class HllSet:
    def __init__(self, p=10):
        if not isinstance(p, int):
            raise ValueError("P must be an integer")
        if p < 4 or p > 18:
            raise ValueError("P must be between 4 and 18")
        self.p = p
        self.m = 2 ** p
        self.counts = np.zeros(self.m, dtype=np.uint32)

    def getbin(self, x):
        return self._getbin(x, self.p)

    @staticmethod
    def _getbin(x, p):
        x = x >> (64 - (p + 1)) + 1
        return int(x % (2 ** p))

    def add(self, x, seed=0):
        h = self.u_hash(x, seed)
        bin_idx = self.getbin(h)
        idx = self._getzeros(h)
        if idx <= 32:
            print((f"type of self.counts: {type(self.counts)}"))
            old_value = self.counts[bin_idx]
            self.counts[bin_idx] = old_value | np.uint32(1 << (idx - 1))

    @staticmethod
    def _getzeros(x):
        return (x & -x).bit_length()

    @staticmethod
    def u_hash(x, seed=0):
        # hash = mmh3.hash(x, seed)
        if seed == 0:
            abs_hash = abs(mmh3.hash64(str(x))[0])
        else:
            abs_hash = abs(mmh3.hash64(str(x), seed)[0])
        return abs_hash % (2**64)

    def alpha(self):
        if self.p == 4:
            return 0.673
        elif self.p == 5:
            return 0.697
        elif self.p == 6:
            return 0.709
        else:
            return 0.7213 / (1 + 1.079 / self.m)

    def bias(self, biased_estimate):
        if self.p < 4 or self.p > 18:
            raise ValueError("We only have bias estimates for P ∈ 4:18")
        rawarray = RAW_ARRAYS[self.p - 4]
        biasarray = BIAS_ARRAYS[self.p - 4]
        firstindex = np.searchsorted(rawarray, biased_estimate)
        if firstindex == len(rawarray):
            return 0.0
        elif firstindex == 0:
            return biasarray[0]
        else:
            x1, x2 = rawarray[firstindex - 1], rawarray[firstindex]
            y1, y2 = biasarray[firstindex - 1], biasarray[firstindex]
            delta = (biased_estimate - x1) / (x2 - x1)
            return y1 + delta * (y2 - y1)

    @staticmethod
    def maxidx(x):
        return int(x).bit_length()

    def count(self):
        harmonic_mean = self.m / np.sum(1.0 / (1 << self.maxidx(i)) for i in self.counts)
        biased_estimate = self.alpha() * self.m * harmonic_mean
        return np.round(biased_estimate - self.bias(biased_estimate))


class HllSetAlgebra:
    @staticmethod
    @jax.jit
    def union(register1, register2):
        return register1 | register2

    @staticmethod
    @jax.jit
    def intersection(register1, register2):
        return register1 & register2

    @staticmethod
    @jax.jit
    def xor(register1, register2):
        return register1 ^ register2

    @staticmethod
    @jax.jit
    def complement(register):
        return ~register

    @staticmethod
    @jax.jit
    def difference(register1, register2):
        return register1 & ~register2