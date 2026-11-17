#!/usr/bin/env python3
"""Compare two backend parity oracle dumps (tests/oracle_train.cu).

    python3 tests/oracle_compare.py reference.bin candidate.bin

Prints per-block max abs diff, max abs diff relative to the reference block's
max magnitude, and (for weights_out) the relative diff of the *update*
(weights_out - weights_in), which is what training actually changed.
Standard library only.
"""
import struct
import sys
from array import array


def load(path):
    blocks = {}
    with open(path, "rb") as f:
        while True:
            tag = f.read(32)
            if len(tag) < 32:
                break
            name = tag.rstrip(b"\0").decode()
            (n,) = struct.unpack("<q", f.read(8))
            a = array("f")
            a.frombytes(f.read(4 * n))
            blocks[name] = a
    return blocks


def stats(ref, cand):
    assert len(ref) == len(cand), (len(ref), len(cand))
    maxdiff = max((abs(r - c) for r, c in zip(ref, cand)), default=0.0)
    scale = max((abs(r) for r in ref), default=0.0) or 1.0
    return maxdiff, maxdiff / scale


def main():
    ref, cand = load(sys.argv[1]), load(sys.argv[2])
    worst = 0.0
    for name in ref:
        d, rel = stats(ref[name], cand[name])
        print(f"{name:14s} n={len(ref[name]):8d} max_abs={d:.3e} max_rel={rel:.3e}")
        if name != "weights_in":
            worst = max(worst, rel)
    if "weights_in" in ref and "weights_out" in ref:
        upd_r = array("f", (o - i for o, i in zip(ref["weights_out"], ref["weights_in"])))
        upd_c = array("f", (o - i for o, i in zip(cand["weights_out"], cand["weights_in"])))
        d, rel = stats(upd_r, upd_c)
        print(f"{'update':14s} n={len(upd_r):8d} max_abs={d:.3e} max_rel={rel:.3e}")
        worst = max(worst, rel)
    names = ["policy", "value", "entropy", "total", "old_kl", "kl", "clipfrac", "imp", "count"]
    if "losses" in ref:
        for i, n in enumerate(names[: len(ref["losses"])]):
            print(f"  loss {n:9s} ref={ref['losses'][i]: .6e} cand={cand['losses'][i]: .6e}")
    print(f"worst relative diff: {worst:.3e}")


if __name__ == "__main__":
    main()
