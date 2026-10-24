'''
Benchmark of the TOF simulation against the Ar40 runs (trap floor 180 V, wall 190 V, catching potential 12 kV).

The measured MCP signals of the runs (default 427300, 427304, 427316) are background subtracted and averaged.
The model of the spectrum is a non-negative combination of simulated species:
 - fragments of antiproton annihilation on Ar40 (Geant4 yields of trapped fragments, fully stripped) with relative
   yields fixed by the simulation and one free amplitude,
 - ions of the residual gas (H2+, He+, He2+, Ar40 q=1..18, with --residual_gas also H2O+, N2+, O2+, CO2+) with free amplitudes.
The time calibration (launch delay with respect to the trigger and effective flight distance) is fitted to the positions
of the main peaks (He2+, He+, Ar3+, Ar2+, Ar+); a free scan of the calibration is reported for comparison.
Configurations: legacy 1D engine (space charge limits), tracking engine with a thermal cloud and with a plasma
(the plasma uses the abundances found with the best cloud fit). Results are written to benchmarks/output/ar40_*.

usage: python benchmarks/ar40_runs.py [--workers 8] [--quick] [--runs 427300 427304 427316] [--B_map aegis --tag aegis]
       [--cloud 3000 30000 30000/3000] [--plasma 1e6@1000 5e6@3000] [--residual_gas]
'''
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from scipy.optimize import nnls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import HCI_DATA, cached_simulation, output_path, peak_time, rebin, resample
import FragmentsDataLoader as loader

T0_OFFSET = 10e-6 # s, trigger before the launch
BIN = 50e-9 # s
SIM_BIN = 10e-9 # s, bin of the simulated signals (tof_dt)
T_RANGE = (5e-6, 45e-6) # s, range of the fit
EDGES = np.arange(0, 50e-6 + BIN / 2, BIN)
CENTRES = EDGES[:-1] + BIN / 2
SHIFTS = np.arange(-600e-9, 600e-9 + 1e-12, 10e-9) # s, scanned time shifts of the simulation
SCALES = np.arange(0.94, 1.16 + 1e-9, 0.0025) # scanned time scale (effective flight distance / 1.05 m), covers 0.99-1.22 m
LOCAL_SCALES = np.arange(-0.02, 0.02 + 1e-9, 0.001) # refinement of the calibration around the peak positions
LOCAL_SHIFTS = np.arange(-150e-9, 150e-9 + 1e-12, 10e-9) # s
FLOOR_V, WALL_V = 180.0, 190.0
NOMINAL_DISTANCE_M = 1.05 # m, distance to the MCP used in the simulations (default of calculate_tof)
# residual gas ions: label, m/q [u/e], charge
GAS_IONS = ([("H2+", 2.01510, 1.0), ("He+", 4.002603, 1.0), ("He2+", 4.002603 / 2, 2.0)]
            + [(f"Ar{q}+", 39.962383 / q, float(q)) for q in range(1, 19)])
# molecular ions of the residual gas (--residual_gas), between and after the Ar2+ and Ar+ peaks
RESIDUAL_GAS_IONS = [("H2O+", 18.010565, 1.0), ("N2+", 28.006148, 1.0), ("O2+", 31.989830, 1.0), ("CO2+", 43.989830, 1.0)]
# main peaks of the measured spectrum and the gas ions expected to produce them
MAIN_PEAKS = [("He2+", 8.4e-6), ("He+", 11.8e-6), ("Ar3+", 21.7e-6), ("Ar2+", 26.4e-6), ("Ar1+", 37.4e-6)]


def load_runs(runs:list[int]) -> dict:
    data = pl.read_parquet(HCI_DATA, columns=["Run", "trap_floor_V", "trap_wall_V", "NegHV1_kV", "captorius_t", "captorius_V"]).filter(pl.col("Run").is_in(runs))
    signals, noises = [], []
    for row in data.iter_rows(named=True):
        t = np.asarray(row["captorius_t"]) - T0_OFFSET
        y = np.asarray(row["captorius_V"])
        signal = -(y - y[t < 0].mean())
        signals.append(rebin(t, signal, EDGES))
        pre = rebin(t, signal, np.arange(-9e-6, -0.5e-6, BIN))
        noises.append(np.nanstd(pre))
        print(f"run {row['Run']}: floor {row['trap_floor_V']} V, wall {row['trap_wall_V']} V, catching {row['NegHV1_kV']} kV, noise {noises[-1]*1e3:.3f} mV per bin")
    return {"runs": data["Run"].to_list(), "signal": np.nanmean(np.stack(signals), axis=0),
            "noise": float(np.sqrt(np.mean(np.square(noises)) / len(noises))), "catching_kV": float(data["NegHV1_kV"].mean())}


def fragment_species(catching_kV:float, max_species:int, half_life:float = 600.0) -> pl.DataFrame:
    '''
    Trapped fragments of the antiproton annihilation on Ar40 (yield per annihilation), fully stripped.
    '''
    data = (loader.create_lf_with_weights("Ar40", half_life=half_life, V=catching_kV, R=0.015, B=5.0)
            .group_by("fragmentZ", "fragmentN")
            .agg(pl.col("atomic_mass").first(), (pl.col("trappable_fraction").sum() / pl.col("total_events").first()).alias("yield"))
            .collect()
            .with_columns(m_over_q=pl.col("atomic_mass") / pl.col("fragmentZ"),
                          label=pl.format("{}-{}{}+", pl.col("fragmentZ") + pl.col("fragmentN"), pl.lit("Z"), pl.col("fragmentZ")))
            .sort("yield", descending=True))
    kept = data.head(max_species)
    print(f"Fragments: {len(data)} species, kept {len(kept)} covering {kept['yield'].sum()/data['yield'].sum()*100:.2f}% of the trapped yield "
          f"({data['yield'].sum():.4f} per annihilation)")
    return kept


def cloud_label(entry:str) -> tuple[str,dict]:
    '''
    Cloud configuration "T" or "T_par/T_perp" [K] -> label and parameters.
    '''
    temperatures = [int(float(value)) if float(value).is_integer() else float(value) for value in entry.split("/")]
    parameters = {"temperature_K": temperatures[0]}
    if len(temperatures) > 1:
        parameters["T_perp_K"] = temperatures[1]
    return f"T={entry} K", parameters


def configurations(quick:bool, cloud:list[str], B_map:str = None) -> list[tuple[str,str,dict]]:
    N, iterations = (2000, 2) if quick else (5000, 3)
    common = dict(trap_floor_V=FLOOR_V, trap_wall_V=WALL_V, tof_range=[0, 60e-6], tof_dt=SIM_BIN, N_particles=N, iterations=iterations,
                  showfig=False, seed=1)
    field = {"B_map": B_map} if B_map else {} # (the legacy engine has no magnetic field)
    configs = []
    for sc_min, sc_max in ([(0.0, None), (4.0, None)] if quick else [(0.0, None), (4.0, None), (0.0, 0.3), (0.2, 1.2), (1.0, 3.0)]):
        configs.append(("legacy engine", f"sc {sc_min}-{sc_max} V", dict(common, engine="legacy", spacecharge_min_V=sc_min, spacecharge_max_V=sc_max)))
    for entry in (cloud[:2] if quick else cloud):
        label, temperatures = cloud_label(entry)
        configs.append(("tracking cloud", label, dict(common, engine="tracking", model="cloud", cloud_radius_mm=1.0, **temperatures, **field)))
    return configs


def plasma_configurations(quick:bool, weights:np.ndarray, plasma:list[str], B_map:str = None) -> list[tuple[str,str,dict]]:
    '''
    Plasma configurations "N@T" (number of charges @ temperature [K]).
    '''
    N, iterations = (1000, 2) if quick else (3000, 2)
    common = dict(trap_floor_V=FLOOR_V, trap_wall_V=WALL_V, tof_range=[0, 60e-6], tof_dt=SIM_BIN, N_particles=N, iterations=iterations,
                  showfig=False, seed=1, engine="tracking", model="plasma", plasma_radius_mm=1.0, weights=list(map(float, weights)))
    field = {"B_map": B_map} if B_map else {}
    grid = [(float(entry.split("@")[0]), int(float(entry.split("@")[1])) if float(entry.split("@")[1]).is_integer() else float(entry.split("@")[1]))
            for entry in (plasma[:1] if quick else plasma)]
    return [("tracking plasma", f"N={n:.0e} T={T:g} K", dict(common, n_charges=n, temperature_K=T, **field)) for n, T in grid]


def _run(model:str, label:str, parameters:dict):
    try:
        signal, _, runtime = cached_simulation("ar40", parameters)
        return model, label, parameters, signal, runtime, None
    except Exception as error:
        return model, label, parameters, None, 0.0, f"{type(error).__name__}: {error}"


class Components:
    '''
    Simulated signals of the fragments (sum weighted by their yields) and of each gas ion,
    on the binning of the data after scaling (t * scale, effective flight distance) and shifting (t + shift) the time.
    '''
    def __init__(self, signal:pl.DataFrame, species:pl.DataFrame, n_fragments:int):
        by_label = {label: frame for (label,), frame in signal.group_by("label")}
        fragment_yields = species.head(n_fragments)["weight"].to_numpy()
        fragment_yields = fragment_yields / fragment_yields.sum()
        frames = [by_label[label] for label in species["label"]]
        # fragments are one component: concatenate their signals weighted with the yields
        fragments_t = np.concatenate([f["t [s]"].to_numpy() for f in frames[:n_fragments]])
        fragments_y = np.concatenate([w * f["signal"].to_numpy() for w, f in zip(fragment_yields, frames[:n_fragments])])
        self.times = [fragments_t] + [f["t [s]"].to_numpy() for f in frames[n_fragments:]]
        self.values = [fragments_y] + [f["signal"].to_numpy() for f in frames[n_fragments:]]

    def matrix(self, scale:float = 1.0, shift:float = 0.0) -> np.ndarray:
        return np.column_stack([resample(t, y, SIM_BIN, EDGES, scale, shift) for t, y in zip(self.times, self.values)])


def _nnls(components:Components, measured:np.ndarray, selected:np.ndarray, noise:float, scale:float, shift:float):
    matrix = components.matrix(scale, shift)[selected] / noise
    amplitudes, residual_norm = nnls(matrix, measured, maxiter=5000)
    return amplitudes, residual_norm**2 / (selected.sum() - matrix.shape[1] - 1)


def peak_calibration(data:dict, unshifted:np.ndarray, species:pl.DataFrame, n_fragments:int) -> tuple[float,float]:
    '''
    Time calibration t_data = scale * t_sim + shift from the positions of the main peaks, each produced by one gas ion.
    Unlike the fit of the whole spectrum it can't be biased by trading amplitudes between species.
    '''
    gas_labels = species.slice(n_fragments)["label"].to_list()
    t_data, t_sim = [], []
    for ion, expected in MAIN_PEAKS:
        around = (CENTRES > expected - 1.5e-6) & (CENTRES < expected + 1.5e-6)
        t_data.append(peak_time(CENTRES[around], data["signal"][around]))
        t_sim.append(peak_time(CENTRES, unshifted[:, 1 + gas_labels.index(ion)]))
    scale, shift = np.polyfit(t_sim, t_data, 1)
    return float(scale), float(shift)


def fit(data:dict, signal:pl.DataFrame, species:pl.DataFrame, n_fragments:int) -> dict:
    '''
    Non-negative fit of the component amplitudes:
     - with the time calibration (effective flight distance 1.05 m * scale and shift) anchored at the positions of the main
       peaks and refined locally (within 2% of the distance and 150 ns) to the best chi2 (main result),
     - with the calibration scanned freely (the amplitudes of many species can compensate a wrong calibration,
       so this minimum isn't always physical),
     - with the nominal flight distance and the shift scanned.
    '''
    selected = (CENTRES >= T_RANGE[0]) & (CENTRES <= T_RANGE[1])
    measured = np.nan_to_num(data["signal"][selected]) / data["noise"]
    components = Components(signal, species, n_fragments)
    unshifted = components.matrix(1.0, 0.0)
    # the peak positions (known only to ~one bin) anchor the calibration, which is then refined locally
    scale_peaks, shift_peaks = peak_calibration(data, unshifted, species, n_fragments)
    chi2, scale, shift = min(((_nnls(components, measured, selected, data["noise"], k, s)[1], k, s)
                              for k in scale_peaks + LOCAL_SCALES for s in shift_peaks + LOCAL_SHIFTS))
    amplitudes, _ = _nnls(components, measured, selected, data["noise"], scale, shift)
    matrix = components.matrix(scale, shift)
    nominal = min(((_nnls(components, measured, selected, data["noise"], 1.0, s)[1], s) for s in SHIFTS))
    free = min(((_nnls(components, measured, selected, data["noise"], k, s)[1], k, s) for k in SCALES for s in SHIFTS[::3]))
    # refine the shift for the best scale
    free = min(((_nnls(components, measured, selected, data["noise"], free[1], s)[1], free[1], s) for s in SHIFTS))
    return {"chi2": chi2, "scale": scale, "shift": shift, "amplitudes": amplitudes, "scale_peaks": scale_peaks, "shift_peaks": shift_peaks,
            "model": matrix @ amplitudes, "components": matrix * amplitudes[None, :], "unshifted": unshifted,
            "free_chi2": free[0], "free_scale": free[1], "free_shift": free[2],
            "free_amplitudes": _nnls(components, measured, selected, data["noise"], free[1], free[2])[0],
            "chi2_nominal": nominal[0], "shift_nominal": nominal[1],
            "chi2_no_shift": _nnls(components, measured, selected, data["noise"], 1.0, 0.0)[1]}


def peak_table(data:dict, results:dict, species:pl.DataFrame, n_fragments:int) -> pl.DataFrame:
    '''
    Position of the main peaks in the data and in the simulated signal of the corresponding gas ion (without the fitted shift),
    dt = t_data - t_sim.
    '''
    rows = []
    gas_labels = species.slice(n_fragments)["label"].to_list()
    for ion, expected in MAIN_PEAKS:
        around = (CENTRES > expected - 1.5e-6) & (CENTRES < expected + 1.5e-6)
        t_data = peak_time(CENTRES[around], data["signal"][around])
        row = {"peak": ion, "t_data_us": t_data * 1e6}
        for name, result in results.items():
            component = result["unshifted"][:, 1 + gas_labels.index(ion)]
            t_sim = peak_time(CENTRES[around], component[around]) if component[around].max() > 0 else np.nan
            row[f"{name} dt_ns"] = (t_data - t_sim) * 1e9
        rows.append(row)
    return pl.DataFrame(rows)


def plot(data:dict, results:dict, species:pl.DataFrame, n_fragments:int, path:str) -> None:
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(len(results) + 1, 1, figsize=(11, 3.2 * (len(results) + 1)), sharex=True)
    for ax, (name, result) in zip(axes, results.items()):
        ax.plot(CENTRES * 1e6, data["signal"] * 1e3, 'k-', lw=0.7, label=f"data (runs {', '.join(map(str, data['runs']))})")
        ax.plot(CENTRES * 1e6, result["model"] * 1e3, '-', color='tab:red', lw=1, label=f"fit, chi2/dof={result['chi2']:.2f}, distance={NOMINAL_DISTANCE_M*result['scale']:.4f} m, shift={result['shift']*1e9:.0f} ns")
        ax.plot(CENTRES * 1e6, result["components"][:, 0] * 1e3, '-', color='tab:green', lw=1, label=r"$\bar{p}$-Ar40 fragments")
        ax.set_title(name, fontsize=9)
        ax.set_ylabel("MCP signal [mV]")
        ax.set_yscale("symlog", linthresh=0.3)
        ax.set_ylim(-0.5, 50)
        ax.grid()
        ax.legend(fontsize=7, loc='upper right')
    ax = axes[-1]
    for i, (name, result) in enumerate(results.items()):
        ax.plot(CENTRES * 1e6, (data["signal"] - result["model"]) / data["noise"], '-', lw=0.6, color=colors[i % 10], label=name)
    ax.axvspan(0, T_RANGE[0] * 1e6, color='lightgray')
    ax.set_ylabel("residual / noise")
    ax.set_xlabel("t after launch (trigger + 10 us) [us]")
    ax.grid()
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary_row(model:str, label:str, result:dict, runtime:float) -> dict:
    return {"model": model, "label": label, "chi2_dof": result["chi2"], "distance_m": NOMINAL_DISTANCE_M * result["scale"],
            "shift_ns": result["shift"] * 1e9,
            "distance_peaks_m": NOMINAL_DISTANCE_M * result["scale_peaks"], "shift_peaks_ns": result["shift_peaks"] * 1e9,
            "chi2_dof_free_scan": result["free_chi2"], "distance_free_scan_m": NOMINAL_DISTANCE_M * result["free_scale"],
            "shift_free_scan_ns": result["free_shift"] * 1e9,
            "chi2_dof_nominal_distance": result["chi2_nominal"], "shift_nominal_ns": result["shift_nominal"] * 1e9,
            "chi2_dof_no_shift": result["chi2_no_shift"],
            "fragment_fraction": result["components"][:, 0].sum() / result["model"].sum(), "runtime_s": runtime}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--runs", type=int, nargs="+", default=[427300, 427304, 427316])
    parser.add_argument("--max_fragments", type=int, default=30)
    parser.add_argument("--B_map", default=None, help="magnetic field of the tracking engine (e.g. aegis), default uniform 5 T")
    parser.add_argument("--cloud", nargs="+", default=["300", "3000", "10000", "30000"],
                        help="cloud temperatures, T or T_par/T_perp [K]")
    parser.add_argument("--plasma", nargs="*", default=["1e6@1000", "1e6@10000", "5e6@1000", "5e6@10000"],
                        help="plasma configurations, number of charges@temperature [K] (none: no plasma)")
    parser.add_argument("--residual_gas", action="store_true", help="add the molecular ions H2O+, N2+, O2+, CO2+ to the fit")
    parser.add_argument("--tag", default="", help="tag added to the names of the output files")
    args = parser.parse_args()
    name = lambda title: f"ar40_{args.tag}_{title}" if args.tag else f"ar40_{title}"
    start = time.time()
    data = load_runs(args.runs)
    fragments = fragment_species(data["catching_kV"], args.max_fragments)
    fragments.write_csv(output_path(name("fragments.csv")))
    gas = GAS_IONS + (RESIDUAL_GAS_IONS if args.residual_gas else [])
    species = pl.concat([fragments.select("label", "m_over_q", pl.col("fragmentZ").cast(pl.Float64).alias("charge"), pl.col("yield").alias("weight")),
                         pl.DataFrame({"label": [g[0] for g in gas], "m_over_q": [g[1] for g in gas],
                                       "charge": [g[2] for g in gas], "weight": [1.0] * len(gas)})])
    n_fragments = len(fragments)
    species_parameters = dict(mq=species["m_over_q"].to_list(), labels=species["label"].to_list(),
                              charges=species["charge"].to_list(), weights=species["weight"].to_list())

    def run_all(configs):
        jobs = [(model, label, {**species_parameters, **parameters}) for model, label, parameters in configs]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            return list(pool.map(_run, *zip(*jobs)))

    simulations = run_all(configurations(args.quick, args.cloud, args.B_map))
    print(f"Simulations done ({time.time()-start:.0f} s)")
    results, rows = {}, []
    for model, label, parameters, signal, runtime, error in simulations:
        if signal is None:
            print(f"{model} {label} failed: {error}")
            continue
        result = fit(data, signal, species, n_fragments)
        results[f"{model} ({label})"] = result
        rows.append(summary_row(model, label, result, runtime))
    # plasma with the abundances of the free-scan fit of the best cloud (number of ions = amplitude of the normalised signal)
    plasma_simulations = []
    if args.plasma:
        cloud = min((r for r in rows if r["model"] == "tracking cloud"), key=lambda r: r["chi2_dof_free_scan"])
        amplitudes = results[f"tracking cloud ({cloud['label']})"]["free_amplitudes"]
        weights = np.concatenate([fragments["yield"].to_numpy() / fragments["yield"].sum() * amplitudes[0], amplitudes[1:]])
        weights = np.maximum(weights, 1e-6 * weights.max())
        # abundances used by the plasma simulations (needed to find them in the cache, e.g. by plot_ar40_first_peak.py)
        with open(output_path(name("plasma_weights.json")), "w") as f:
            json.dump(list(map(float, weights)), f)
        plasma_simulations = run_all(plasma_configurations(args.quick, weights, args.plasma, args.B_map))
    for model, label, parameters, signal, runtime, error in plasma_simulations:
        if signal is None:
            print(f"{model} {label} failed: {error}")
            continue
        result = fit(data, signal, species, n_fragments)
        results[f"{model} ({label})"] = result
        rows.append(summary_row(model, label, result, runtime))
    summary = pl.DataFrame(rows).sort("chi2_dof")
    best = {f"{r['model']} ({r['label']})": results[f"{r['model']} ({r['label']})"] for r in summary.group_by("model").agg(pl.all().sort_by("chi2_dof").first()).iter_rows(named=True)}
    peaks = peak_table(data, best, species, n_fragments)
    abundances = pl.DataFrame({"component": ["fragments"] + species.slice(n_fragments)["label"].to_list(),
                               **{name: result["components"].sum(axis=0) / result["model"].sum() for name, result in best.items()}})
    summary.write_csv(output_path(name("summary.csv")))
    peaks.write_csv(output_path(name("peaks.csv")))
    abundances.write_csv(output_path(name("abundances.csv")))
    pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(40)
    print(summary)
    print(peaks)
    print(abundances)
    plot(data, best, species, n_fragments, output_path(name("fits.png")))
    print(f"Benchmark finished in {time.time()-start:.0f} s")


if __name__ == "__main__":
    main()
