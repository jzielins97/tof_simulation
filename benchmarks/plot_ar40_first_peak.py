'''
Fit of the first peak of the Ar40 spectrum (m/q ~ 2: He2+ and fragments of the antiproton annihilation on Ar40)
with the tracking engine, for the thermal cloud and for the plasma initial conditions.
The time calibration (effective flight distance and shift) comes from the positions of the main peaks (ar40_runs.py);
the amplitudes of He2+ and of the fragments (relative yields from Geant4) are fitted to the first peak only.
Requires the cached simulations of ar40_runs.py (run it first).

usage: python benchmarks/plot_ar40_first_peak.py [--cloud "T=30000 K"] [--plasma "N=5e+06 T=1000 K"] [--cloud_particles 10000 --cloud_iterations 9] [--plasma_particles 10000 --plasma_iterations 3]
'''
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")
import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import json
import re

import numpy as np
import polars as pl
from scipy.optimize import nnls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import HCI_DATA, OUTPUT_DIR, cached_simulation, output_path, rebin, resample
from ar40_runs import FLOOR_V, WALL_V, T0_OFFSET, SIM_BIN, fragment_species

RUNS = [427300, 427304, 427316]
NOMINAL_DISTANCE_M = 1.05
BIN = 20e-9 # s
EDGES = np.arange(6e-6, 12e-6 + BIN / 2, BIN)
CENTRES = EDGES[:-1] + BIN / 2
FIT_WINDOW = (7.6e-6, 10.8e-6) # s, first peak (He+ starts at ~11.4 us)
# reference palette (light): text inks and the first three categorical slots
INK, INK_SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"He2+": "#2a78d6", "fragments": "#eb6834", "total": "#1baf7a"}


def load_data() -> tuple[np.ndarray,float]:
    data = pl.read_parquet(HCI_DATA, columns=["Run", "captorius_t", "captorius_V"]).filter(pl.col("Run").is_in(RUNS))
    signals, noises = [], []
    for row in data.iter_rows(named=True):
        t = np.asarray(row["captorius_t"]) - T0_OFFSET
        y = np.asarray(row["captorius_V"])
        signal = -(y - y[t < 0].mean())
        signals.append(rebin(t, signal, EDGES))
        noises.append(np.nanstd(rebin(t, signal, np.arange(-9e-6, -0.5e-6, BIN))))
    return np.nanmean(np.stack(signals), axis=0), float(np.sqrt(np.mean(np.square(noises)) / len(noises)))


def benchmark_parameters(model:str, label:str) -> dict:
    '''
    Parameters of the cached simulation of ar40_runs.py for the configuration (the one with most particles if there are several).
    The plasma simulations are identified also by the abundances of the species used in the last run of ar40_runs.py.
    '''
    numbers = dict((key, float(value)) for key, value in re.findall(r"(N|T)=([0-9.e+]+)", label))
    plasma_weights = None
    if model == "tracking plasma":
        with open(os.path.join(OUTPUT_DIR, "ar40_plasma_weights.json")) as f:
            plasma_weights = json.load(f)
    best = None
    for path in glob.glob(os.path.join(OUTPUT_DIR, "cache", "ar40", "*.json")):
        with open(path) as f:
            parameters = json.load(f)["parameters"]
        if parameters.get("engine") != "tracking" or parameters.get("model") != model.split()[-1]:
            continue
        if parameters["temperature_K"] != numbers["T"] or ("N" in numbers and parameters["n_charges"] != numbers["N"]):
            continue
        if plasma_weights is not None and not np.allclose(parameters["weights"], plasma_weights, rtol=1e-12, atol=0):
            continue
        if best is None or parameters["N_particles"] > best[0]:
            best = (parameters["N_particles"], parameters)
    if best is None:
        raise FileNotFoundError(f"No cached simulation for {model} ({label}), run benchmarks/ar40_runs.py first")
    return best[1]


def high_statistics_parameters(model:str, label:str, fragments:pl.DataFrame, particles:int, iterations:int) -> dict:
    '''
    The benchmark configuration with more simulated ions. The ions of a cloud don't interact, so only He2+ and the
    fragments are simulated; in a plasma all species contribute to the space charge, so all of them are kept.
    '''
    parameters = dict(benchmark_parameters(model, label), N_particles=particles, iterations=iterations)
    if model == "tracking cloud":
        keep = [i for i, name in enumerate(parameters["labels"]) if name == "He2+" or name in fragments["label"].to_list()]
        for key in ["mq", "labels", "charges", "weights"]:
            parameters[key] = [parameters[key][i] for i in keep]
    return parameters


def _simulate(parameters:dict) -> pl.DataFrame:
    return cached_simulation("ar40_first_peak", parameters)[0]


def components(signal:pl.DataFrame, fragments:pl.DataFrame, scale:float, shift:float) -> dict[str,np.ndarray]:
    '''
    He2+ and fragment signals (fraction of simulated ions per bin) after the time calibration t -> t * scale + shift;
    the fragments are summed with their relative yields.
    '''
    def binned(label):
        frame = signal.filter(pl.col("label") == label)
        return resample(frame["t [s]"].to_numpy(), frame["signal"].to_numpy(), SIM_BIN, EDGES, scale, shift)
    yields = fragments["yield"].to_numpy() / fragments["yield"].sum()
    return {"He2+": binned("He2+"), "fragments": sum(w * binned(label) for w, label in zip(yields, fragments["label"]))}


def fit_first_peak(measured:np.ndarray, noise:float, parts:dict[str,np.ndarray]) -> dict:
    window = (CENTRES >= FIT_WINDOW[0]) & (CENTRES <= FIT_WINDOW[1])
    matrix = np.column_stack([parts["He2+"], parts["fragments"]])
    amplitudes, residual = nnls(matrix[window] / noise, np.nan_to_num(measured[window]) / noise)
    fitted = {name: amplitude * parts[name] for name, amplitude in zip(["He2+", "fragments"], amplitudes)}
    total = fitted["He2+"] + fitted["fragments"]
    return {**fitted, "total": total, "chi2_dof": residual**2 / (window.sum() - 2),
            "fractions": {name: fitted[name][window].sum() / max(total[window].sum(), 1e-30) for name in ["He2+", "fragments"]}}


def main():
    parser = argparse.ArgumentParser()
    summary = pl.read_csv(os.path.join(OUTPUT_DIR, "ar40_summary.csv"))
    best = {model: summary.filter(pl.col("model") == model).sort("chi2_dof")["label"][0] for model in ["tracking cloud", "tracking plasma"]}
    parser.add_argument("--cloud", default=best["tracking cloud"])
    parser.add_argument("--plasma", default=best["tracking plasma"])
    parser.add_argument("--cloud_particles", type=int, default=10000, help="ions per species and iteration (cloud)")
    parser.add_argument("--plasma_particles", type=int, default=10000, help="ions per species and iteration (plasma)")
    parser.add_argument("--cloud_iterations", type=int, default=9)
    parser.add_argument("--plasma_iterations", type=int, default=3)
    args = parser.parse_args()
    measured, noise = load_data()
    fragments = fragment_species(12.0, 30)
    configurations = [("tracking cloud", args.cloud, "Thermal cloud"), ("tracking plasma", args.plasma, "Plasma (self-consistent space charge)")]
    jobs = [high_statistics_parameters("tracking cloud", args.cloud, fragments, args.cloud_particles, args.cloud_iterations),
            high_statistics_parameters("tracking plasma", args.plasma, fragments, args.plasma_particles, args.plasma_iterations)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        signals = list(pool.map(_simulate, jobs))

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK, "xtick.color": INK_SECONDARY,
                         "ytick.color": INK_SECONDARY, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True, sharey=True)
    window = (CENTRES >= FIT_WINDOW[0] - 0.4e-6) & (CENTRES <= FIT_WINDOW[1])
    t_us = CENTRES[window] * 1e6
    for ax, (model, label, title), signal, parameters in zip(axes, configurations, signals, jobs):
        calibration = summary.filter(pl.col("model") == model, pl.col("label") == label).row(0, named=True)
        scale, shift = calibration["distance_m"] / NOMINAL_DISTANCE_M, calibration["shift_ns"] * 1e-9
        result = fit_first_peak(measured, noise, components(signal, fragments, scale, shift))
        ax.errorbar(t_us, measured[window] * 1e3, yerr=noise * 1e3, fmt='o', markersize=3, color=INK_SECONDARY,
                    ecolor=GRID, elinewidth=1, label=f"data, runs {RUNS[0]}/{'/'.join(str(run)[-3:] for run in RUNS[1:])}", zorder=1)
        ax.plot(t_us, result["total"][window] * 1e3, color=COLORS["total"], lw=2, label="fit: He$^{2+}$ + fragments", zorder=4)
        ax.plot(t_us, result["He2+"][window] * 1e3, color=COLORS["He2+"], lw=2, label="He$^{2+}$", zorder=3)
        ax.plot(t_us, result["fragments"][window] * 1e3, color=COLORS["fragments"], lw=2, label=r"$\bar{p}$-$^{40}$Ar fragments", zorder=3)
        # direct labels at the maxima of the components (text in ink, not in series colour)
        for name, text in [("He2+", "He$^{2+}$"), ("fragments", "fragments")]:
            values = result[name][window] * 1e3
            if values.max() > 0:
                i = int(np.argmax(values))
                ax.annotate(text, (t_us[i], values[i]), xytext=(24, -26 if name == "fragments" else 12), textcoords="offset points",
                            color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, lw=0.6),
                            bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.85))
        ax.axvspan(FIT_WINDOW[0] * 1e6, FIT_WINDOW[1] * 1e6, color="#f3f2ef", zorder=0, lw=0)
        ax.set_title(f"{title}: {label.replace('N=', 'N = ').replace('T=', 'T = ')} "
                     f"({parameters['N_particles']}×{parameters['iterations']} ions per species)", loc="left", fontsize=11, color=INK)
        ax.text(0.99, 0.96,
                f"He$^{{2+}}$ {result['fractions']['He2+']*100:.0f}% / fragments {result['fractions']['fragments']*100:.0f}% of the peak\n"
                f"$\\chi^2$/dof = {result['chi2_dof']:.2f} in the fit window {FIT_WINDOW[0]*1e6:.1f}-{FIT_WINDOW[1]*1e6:.1f} µs (shaded)\n"
                f"calibration from the main peaks: L = {calibration['distance_m']:.4f} m, shift {calibration['shift_ns']:.0f} ns",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel("MCP signal [mV]")
        # headroom above the data for the legend and the fit results
        ax.set_ylim(-1.0, 1.75 * np.nanmax(measured[window]) * 1e3)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        print(f"{model} ({label}): He2+ {result['fractions']['He2+']*100:.1f}%, fragments {result['fractions']['fragments']*100:.1f}%, chi2/dof {result['chi2_dof']:.2f}")
    axes[-1].set_xlabel("t after launch (trigger + 10 µs) [µs]")
    fig.suptitle(f"First peak of the Ar40 spectrum (m/q ≈ 2), tracking engine, floor {FLOOR_V:.0f} V / wall {WALL_V:.0f} V",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    path = output_path("ar40_first_peak.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
