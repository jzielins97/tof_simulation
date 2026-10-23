'''
Shared helpers for benchmarking the TOF simulation against measured MCP signals.
'''
import os
import sys
import json
import hashlib

import numpy as np
import polars as pl

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in [REPO_DIR, os.path.join(REPO_DIR, "modules")]:
    if path not in sys.path:
        sys.path.insert(0, path)

import calculate_tof as ct # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CALIBRATION_DATA = os.environ.get("TOF_CALIBRATION_DATA", r"C:\Users\jakub\Documents\HCIanalysis\phd_plots\tof_simulation\calibration\calibration_runs.parquet")
HCI_DATA = os.environ.get("TOF_HCI_DATA", r"C:\Users\jakub\Documents\HCIanalysis\data\HCI_PhD.parquet")


def output_path(*parts:str) -> str:
    path = os.path.join(OUTPUT_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def rebin(t:np.ndarray, y:np.ndarray, edges:np.ndarray) -> np.ndarray:
    '''
    Mean of the samples in each bin (NaN for empty bins).
    '''
    index = np.digitize(t, edges) - 1
    inside = (index >= 0) & (index < len(edges) - 1)
    sums = np.bincount(index[inside], weights=y[inside], minlength=len(edges) - 1)
    counts = np.bincount(index[inside], minlength=len(edges) - 1)
    with np.errstate(invalid='ignore'):
        return sums / counts


def simulated_on_grid(signal:pl.DataFrame, edges:np.ndarray) -> np.ndarray:
    '''
    Simulated signal (fraction of particles per bin of the simulation) summed into the bins defined by edges.
    '''
    t = signal["t [s]"].to_numpy()
    y = signal["signal"].to_numpy()
    index = np.digitize(t, edges) - 1
    inside = (index >= 0) & (index < len(edges) - 1)
    return np.bincount(index[inside], weights=y[inside], minlength=len(edges) - 1)


def resample(t:np.ndarray, y:np.ndarray, sim_bin:float, edges:np.ndarray, scale:float = 1.0, shift:float = 0.0) -> np.ndarray:
    '''
    Content of a simulated histogram (values y in bins of width sim_bin centred at t, starting at 0) in the bins defined
    by edges, after the time transformation t -> t * scale + shift. Each simulated bin is spread uniformly over its
    (transformed) width, so the result has no aliasing when the bins of the two grids don't line up.
    '''
    index = np.rint(t / sim_bin - 0.5).astype(np.int64)
    if len(index) == 0:
        return np.zeros(len(edges) - 1)
    content = np.bincount(index, weights=y, minlength=index.max() + 1)
    cumulative = np.concatenate([[0.0], np.cumsum(content)])
    sim_edges = np.arange(len(cumulative)) * sim_bin * scale + shift
    return np.diff(np.interp(edges, sim_edges, cumulative))


def peak_time(t:np.ndarray, y:np.ndarray, search_from:float = 0.0, smooth_bins:int = 5) -> float:
    '''
    Time of the maximum of the smoothed signal after search_from.
    '''
    kernel = np.ones(smooth_bins) / smooth_bins
    smooth = np.convolve(np.nan_to_num(y), kernel, mode='same')
    smooth[t < search_from] = -np.inf
    return float(t[np.argmax(smooth)])


def peak_statistics(t:np.ndarray, y:np.ndarray, window:tuple[float,float], search_from:float = 0.0, smooth_bins:int = 5) -> dict[str,float]:
    '''
    Position of the maximum (after smoothing, searched after search_from) and quantiles of the signal inside a window
    around the maximum. window = (before, after) relative to the maximum [s]. Negative values of the signal are ignored.
    '''
    t_max = peak_time(t, y, search_from, smooth_bins)
    selected = (t >= t_max - window[0]) & (t <= t_max + window[1])
    weights = np.clip(np.nan_to_num(y[selected]), 0, None)
    cumulative = np.cumsum(weights)
    cumulative = cumulative / cumulative[-1]
    quantile = lambda q: float(np.interp(q, cumulative, t[selected]))
    return {"t_max": float(t_max), "t10": quantile(0.10), "t16": quantile(0.16), "t50": quantile(0.50),
            "t84": quantile(0.84), "t90": quantile(0.90), "width": quantile(0.84) - quantile(0.16)}


def cached_simulation(tag:str, parameters:dict) -> tuple[pl.DataFrame,pl.DataFrame,float]:
    '''
    Run calculate_tof with the parameters (or read the cached result). Returns signal, peak and run time [s].
    '''
    signal_path, peak_path, meta_path = cache_paths(tag, parameters)
    if os.path.exists(signal_path) and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return pl.read_parquet(signal_path), pl.read_parquet(peak_path), meta["runtime_s"]
    signal, peak, runtime = run_simulation(parameters)
    # write to temporary files and rename them, so that a crash can't leave a truncated cache entry
    # (the metadata file is written last and marks a complete entry)
    for frame, path in [(signal, signal_path), (peak, peak_path)]:
        frame.write_parquet(path + ".tmp")
        os.replace(path + ".tmp", path)
    with open(meta_path + ".tmp", "w") as f:
        json.dump({"parameters": parameters, "runtime_s": runtime}, f, default=str, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(meta_path + ".tmp", meta_path)
    return signal, peak, runtime


def cache_paths(tag:str, parameters:dict) -> tuple[str,str,str]:
    key = hashlib.sha1(json.dumps(parameters, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return (output_path("cache", tag, f"{key}_signal.parquet"), output_path("cache", tag, f"{key}_peak.parquet"),
            output_path("cache", tag, f"{key}.json"))


def run_simulation(parameters:dict) -> tuple[pl.DataFrame,pl.DataFrame,float]:
    import time
    start = time.time()
    if parameters.get("engine", "legacy") == "legacy":
        np.random.seed(parameters.get("seed", 0) or 0)
    signal, peak = ct.calculate_tof(**parameters)
    return signal, peak, time.time() - start


def verify_cached(tag:str, parameters:dict) -> float:
    '''
    Recompute a cached simulation and return the largest absolute difference of the signal (0 for a reproduced result).
    Used to detect results corrupted by hardware faults (the simulations are deterministic for a fixed seed).
    '''
    signal_path, _, _ = cache_paths(tag, parameters)
    cached = pl.read_parquet(signal_path)
    signal, _, _ = run_simulation(parameters)
    if cached.shape != signal.shape or cached["t [s]"].to_list() != signal["t [s]"].to_list():
        return float("inf")
    return float(np.max(np.abs(cached["signal"].to_numpy() - signal["signal"].to_numpy())))
