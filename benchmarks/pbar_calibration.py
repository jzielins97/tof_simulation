'''
Benchmark of the TOF simulation against the antiproton calibration runs (trap wall -150 V, scan of the floor voltage).

For every floor voltage the measured MCP signals are averaged (time relative to the pickup of the pulse on the right wall,
which marks the launch) and compared with simulations of:
 - the legacy 1D engine (grid of space charge limits),
 - the tracking engine with the legacy initial conditions (particles moving in both directions),
 - the tracking engine with a thermal cloud (grid of temperatures),
 - the tracking engine with a plasma in thermal equilibrium (grid of number of charges and temperatures).
Metrics: median arrival time (t50) residuals, widths (t84 - t16) and reduced chi2 of the normalised peak shape
after aligning t50 (shape only). Results are written to benchmarks/output/pbar_*.

usage: python benchmarks/pbar_calibration.py [--workers 12] [--quick]
'''
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import CALIBRATION_DATA, cached_simulation, output_path, peak_statistics, peak_time, rebin, simulated_on_grid

PBAR_MQ = 1.007276 # u/e
WALL_V = 150.0
BIN = 10e-9 # s
T_END = 60e-6 # s
SEARCH_FROM = 3e-6 # s, the pickup of the pulse is earlier
EDGES = np.arange(0, T_END + BIN / 2, BIN)
CENTRES = EDGES[:-1] + BIN / 2


def window(t_max:float) -> tuple[float,float]:
    '''
    Window around the peak used for statistics and shape comparison (scales with the time of flight).
    '''
    return 0.12 * t_max, 0.25 * t_max


def load_calibration() -> pl.DataFrame:
    '''
    Background subtracted, inverted signals of runs with a pulse and a detectable peak, averaged for each floor voltage.
    Time is relative to the launch (half-height crossing of the pickup of the pulse).
    '''
    runs = []
    for row in pl.read_parquet(CALIBRATION_DATA).iter_rows(named=True):
        t = np.asarray(row["t_s"]); y = np.asarray(row["signal_V"])
        signal = -(y - y[t > 70e-6].mean())
        early = t < 1e-6
        pickup = signal[early]
        if pickup.max() < 0.05:
            print(f"run {row['run']} (floor {-row['V_launch']:.0f} V): no pulse pickup, skipped")
            continue
        t_launch = t[early][np.argmax(pickup > 0.5 * pickup.max())]
        binned = rebin(t - t_launch, signal, EDGES)
        noise = np.nanstd(binned[(CENTRES > 2e-6) & (CENTRES < 5e-6)])
        late = CENTRES > SEARCH_FROM
        if np.nanmax(binned[late]) < 10 * noise:
            print(f"run {row['run']} (floor {-row['V_launch']:.0f} V): no antiproton peak, skipped")
            continue
        runs.append({"run": row["run"], "floor_V": -row["V_launch"], "t_launch_s": t_launch, "noise_V": noise, "signal": binned})
    data = []
    for floor in sorted({run["floor_V"] for run in runs}):
        group = [run for run in runs if run["floor_V"] == floor]
        noise = np.array([run["noise_V"] for run in group])
        data.append({"floor_V": floor, "runs": [run["run"] for run in group], "t_launch_s": [run["t_launch_s"] for run in group],
                     "noise_V": float(np.sqrt(np.mean(noise**2) / len(group))),
                     "signal": np.nanmean(np.stack([run["signal"] for run in group]), axis=0).tolist()})
    return pl.DataFrame(data)


def configurations(quick:bool) -> list[tuple[str,str,dict]]:
    '''
    (model, label, parameters) of the simulations, without the floor voltage.
    '''
    N, iterations = (4000, 3) if quick else (10000, 5)
    common = dict(mq=PBAR_MQ, labels="pbar", charges=1.0, trap_wall_V=WALL_V, tof_range=[0, T_END], tof_dt=BIN,
                  N_particles=N, iterations=iterations, showfig=False, seed=1)
    configs = []
    for sc_min in ([0.0, 0.2] if quick else [0.0, 0.2, 0.5, 1.0]):
        for depth in ([0.05, 0.3] if quick else [0.05, 0.3, 1.0, 2.0]):
            configs.append(("legacy engine", f"sc {sc_min:.2f}-{sc_min+depth:.2f} V",
                            dict(common, engine="legacy", spacecharge_min_V=sc_min, spacecharge_max_V=sc_min + depth)))
    for T in ([300, 3000, 30000] if quick else [300, 1000, 3000, 10000, 30000, 100000]):
        configs.append(("tracking cloud", f"T={T} K", dict(common, engine="tracking", model="cloud", temperature_K=T, cloud_radius_mm=1.0)))
    plasma_N, plasma_it = (2000, 2) if quick else (5000, 3)
    for n_charges in ([1e5, 1e6] if quick else [3e5, 1e6, 3e6]):
        for T in ([1000] if quick else [300, 1000, 3000]):
            configs.append(("tracking plasma", f"N={n_charges:.0e} T={T} K",
                            dict(common, engine="tracking", model="plasma", n_charges=n_charges, temperature_K=T, plasma_radius_mm=1.0,
                                 N_particles=plasma_N, iterations=plasma_it)))
    return configs


def legacy_tracking_configurations(best_legacy:dict) -> list[tuple[str,str,dict]]:
    base = {k: v for k, v in best_legacy.items() if k != "trap_floor_V"}
    label = f"sc {base['spacecharge_min_V']:.2f}-{base['spacecharge_max_V']:.2f} V"
    return [("tracking legacy IC", label + " both directions", dict(base, engine="tracking", model="legacy", forward_only=False)),
            ("tracking legacy IC forward", label + " forward only", dict(base, engine="tracking", model="legacy", forward_only=True))]


def _run(model:str, label:str, parameters:dict):
    try:
        signal, _, runtime = cached_simulation("pbar", parameters)
        return model, label, parameters, signal, runtime, None
    except Exception as error: # keep the benchmark running (e.g. plasma not confined)
        return model, label, parameters, None, 0.0, f"{type(error).__name__}: {error}"


def evaluate(data:pl.DataFrame, results:list) -> pl.DataFrame:
    rows = []
    data_by_floor = {row["floor_V"]: row for row in data.iter_rows(named=True)}
    for model, label, parameters, signal, runtime, error in results:
        floor = parameters["trap_floor_V"]
        if signal is None:
            rows.append({"model": model, "label": label, "floor_V": floor, "error": error})
            continue
        measured = np.asarray(data_by_floor[floor]["signal"])
        noise = data_by_floor[floor]["noise_V"]
        data_peak = peak_time(CENTRES, measured, SEARCH_FROM)
        data_stats = peak_statistics(CENTRES, measured, window(data_peak), SEARCH_FROM)
        simulated = simulated_on_grid(signal, EDGES)
        sim_stats = peak_statistics(CENTRES, simulated, window(CENTRES[np.argmax(simulated)]))
        # shape comparison after aligning the medians
        shift = data_stats["t50"] - sim_stats["t50"]
        before, after = window(data_stats["t_max"])
        selected = (CENTRES >= data_stats["t_max"] - before) & (CENTRES <= data_stats["t_max"] + after)
        measured_w = np.nan_to_num(measured[selected])
        aligned = np.interp(CENTRES[selected], CENTRES + shift, simulated, left=0, right=0)
        norm_data = measured_w.sum()
        chi2 = np.mean(((measured_w / norm_data - aligned / max(aligned.sum(), 1e-30)) / (noise / norm_data))**2)
        # without alignment (timing and shape)
        unaligned = simulated[selected]
        chi2_raw = np.mean(((measured_w / norm_data - unaligned / max(unaligned.sum(), 1e-30)) / (noise / norm_data))**2)
        rows.append({"model": model, "label": label, "floor_V": floor, "error": None, "runtime_s": runtime,
                     "t50_data": data_stats["t50"], "t50_sim": sim_stats["t50"], "dt50": shift,
                     "tmax_data": data_stats["t_max"], "tmax_sim": sim_stats["t_max"],
                     "width_data": data_stats["width"], "width_sim": sim_stats["width"],
                     "chi2_shape": chi2, "chi2_raw": chi2_raw, "fraction_arrived": float(signal["signal"].sum())})
    return pl.DataFrame(rows, infer_schema_length=None)


def timing_fit(rows:pl.DataFrame) -> dict:
    '''
    Fit dt50 = a + b t50_sim: a is a common time offset, b a relative error of the flight distance.
    '''
    t_sim = rows["t50_sim"].to_numpy(); dt = rows["dt50"].to_numpy()
    b, a = np.polyfit(t_sim, dt, 1)
    residual = dt - (a + b * t_sim)
    return {"offset_ns": a * 1e9, "scale": b, "distance_m": 1.05 * (1 + b), "rms_after_fit_ns": float(np.std(residual) * 1e9),
            "mean_dt50_ns": float(dt.mean() * 1e9), "rms_dt50_ns": float(np.sqrt(np.mean(dt**2)) * 1e9)}


def summarise(metrics:pl.DataFrame) -> tuple[pl.DataFrame,pl.DataFrame]:
    good = metrics.filter(pl.col("error").is_null())
    per_config = (good.group_by("model", "label")
                  .agg(pl.col("chi2_shape").median().alias("median_chi2_shape"),
                       pl.col("chi2_raw").median().alias("median_chi2_raw"),
                       (pl.col("dt50").pow(2).mean().sqrt() * 1e9).alias("rms_dt50_ns"),
                       ((pl.col("width_sim") / pl.col("width_data")).median()).alias("median_width_ratio"),
                       pl.col("runtime_s").mean().alias("runtime_s"),
                       pl.len().alias("floors"))
                  .sort("model", "median_chi2_shape"))
    best = per_config.group_by("model").agg(pl.all().sort_by("median_chi2_shape").first())
    timing = []
    for row in best.iter_rows(named=True):
        fit = timing_fit(good.filter(pl.col("model") == row["model"], pl.col("label") == row["label"]))
        timing.append({**row, **fit})
    return per_config, pl.DataFrame(timing).sort("median_chi2_shape")


def plot(data:pl.DataFrame, metrics:pl.DataFrame, best:pl.DataFrame, results:list) -> None:
    colors = {"legacy engine": "tab:blue", "tracking legacy IC": "tab:purple", "tracking legacy IC forward": "tab:gray",
              "tracking cloud": "tab:orange", "tracking plasma": "tab:green"}
    good = metrics.filter(pl.col("error").is_null())
    best_pairs = list(zip(best["model"], best["label"]))
    # timing residuals and widths
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    data_widths = good.unique("floor_V").sort("floor_V")
    axes[1].plot(data_widths["floor_V"], data_widths["width_data"] * 1e9, 'ko-', label="data")
    for model, label in best_pairs:
        rows = good.filter(pl.col("model") == model, pl.col("label") == label).sort("floor_V")
        axes[0].plot(rows["floor_V"], rows["dt50"] * 1e9, 'o-', color=colors[model], label=f"{model} ({label})")
        axes[1].plot(rows["floor_V"], rows["width_sim"] * 1e9, 'o--', color=colors[model], label=f"{model}")
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].set_ylabel(r"$t_{50}^{data} - t_{50}^{sim}$ [ns]")
    axes[1].set_ylabel(r"width $t_{84}-t_{16}$ [ns]")
    axes[1].set_xlabel("trap floor [V] (wall 150 V)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid()
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path("pbar_timing_widths.png"), dpi=150)
    plt.close(fig)

    # spectra for a selection of floors
    floors = data["floor_V"].to_list()
    selection = [floors[i] for i in np.linspace(0, len(floors) - 1, 6).astype(int)]
    signals = {(model, label, parameters["trap_floor_V"]): signal for model, label, parameters, signal, _, _ in results if signal is not None}
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for ax, floor in zip(axes.ravel(), selection):
        row = data.filter(pl.col("floor_V") == floor).row(0, named=True)
        measured = np.asarray(row["signal"])
        t_max = peak_time(CENTRES, measured, SEARCH_FROM)
        before, after = window(t_max)
        selected = (CENTRES >= t_max - 1.5 * before) & (CENTRES <= t_max + 1.5 * after)
        ax.plot(CENTRES[selected] * 1e6, measured[selected] / np.nansum(measured[selected]), 'k-', lw=0.8, label="data")
        for model, label in best_pairs:
            simulated = simulated_on_grid(signals[(model, label, floor)], EDGES)
            ax.plot(CENTRES[selected] * 1e6, simulated[selected] / max(simulated[selected].sum(), 1e-30), color=colors[model], lw=1, label=model)
        ax.set_title(f"floor {floor:.0f} V, wall {WALL_V:.0f} V (runs {', '.join(map(str, row['runs']))})", fontsize=9)
        ax.set_xlabel("t after launch [us]")
        ax.grid()
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path("pbar_spectra.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    start = time.time()
    data = load_calibration()
    print(data.select("floor_V", "runs", "t_launch_s", "noise_V"))
    data.write_parquet(output_path("pbar_data.parquet"))
    floors = data["floor_V"].to_list()

    def run_all(configs):
        jobs = [(model, label, dict(parameters, trap_floor_V=float(floor))) for model, label, parameters in configs for floor in floors]
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run, *job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if i % 20 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} simulations done ({time.time()-start:.0f} s)", flush=True)
        return results

    results = run_all(configurations(args.quick))
    metrics = evaluate(data, results)
    per_config, _ = summarise(metrics)
    best_legacy_label = per_config.filter(pl.col("model") == "legacy engine").sort("median_chi2_shape")["label"][0]
    best_legacy = next(p for m, l, p, *_ in results if m == "legacy engine" and l == best_legacy_label)
    results += run_all(legacy_tracking_configurations(best_legacy))
    metrics = evaluate(data, results)
    per_config, best = summarise(metrics)
    metrics.write_parquet(output_path("pbar_metrics.parquet"))
    per_config.write_csv(output_path("pbar_configurations.csv"))
    best.write_csv(output_path("pbar_best.csv"))
    errors = metrics.filter(pl.col("error").is_not_null())
    if len(errors):
        print("Failed simulations:")
        print(errors.select("model", "label", "floor_V", "error"))
    pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(60)
    print(per_config)
    print(best)
    plot(data, metrics, best, results)
    print(f"Benchmark finished in {time.time()-start:.0f} s")


if __name__ == "__main__":
    main()
