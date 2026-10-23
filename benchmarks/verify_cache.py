'''
Recompute a random sample of cached benchmark simulations and check that they are reproduced exactly.
The simulations are deterministic for a fixed seed, so a difference means a result corrupted by a hardware fault
(e.g. faulty memory), and the entry is removed from the cache so that it is simulated again.

usage: python benchmarks/verify_cache.py [--tag pbar] [--fraction 0.1] [--workers 4]
'''
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")
import argparse
import glob
import json
import random
from concurrent.futures import ProcessPoolExecutor

from common import OUTPUT_DIR, cache_paths, verify_cached


def _verify(tag:str, meta_path:str) -> tuple[str,float,str]:
    with open(meta_path) as f:
        parameters = json.load(f)["parameters"]
    try:
        return meta_path, verify_cached(tag, parameters), ""
    except Exception as error:
        return meta_path, float("inf"), f"{type(error).__name__}: {error}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="pbar")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    entries = sorted(glob.glob(os.path.join(OUTPUT_DIR, "cache", args.tag, "*.json")))
    random.Random(args.seed).shuffle(entries)
    sample = entries[:max(1, int(round(len(entries) * args.fraction)))]
    print(f"Verifying {len(sample)} of {len(entries)} cached simulations ({args.tag})", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_verify, [args.tag] * len(sample), sample))
    mismatched = [(path, difference, error) for path, difference, error in results if difference != 0.0]
    for path, difference, error in mismatched:
        with open(path) as f:
            parameters = json.load(f)["parameters"]
        print(f"MISMATCH {os.path.basename(path)}: max difference {difference} {error} -> removed from the cache")
        for cached in cache_paths(args.tag, parameters):
            if os.path.exists(cached):
                os.remove(cached)
    print(f"{len(sample) - len(mismatched)} of {len(sample)} reproduced exactly")


if __name__ == "__main__":
    main()
