# Benchmarks against measured TOF spectra

Run from the repository root with the `tof_simulation` environment activated:
```sh
python benchmarks/pbar_calibration.py --workers 8   # antiproton calibration runs (wall -150 V, floor scan)
python benchmarks/ar40_runs.py --workers 8          # Ar40 runs 427300, 427304, 427316 (floor 180 V, wall 190 V)
python benchmarks/verify_cache.py --tag pbar        # recompute 10% of cached simulations and compare
```
`--quick` uses fewer particles and parameters. The data files are read from `HCIanalysis` (change them with
`TOF_CALIBRATION_DATA` and `TOF_HCI_DATA`). Every simulation is cached in `benchmarks/output/cache`, so an interrupted
run continues where it stopped. Plots and tables are written to `benchmarks/output` (not tracked by git).

## Method
- **Antiprotons:**
  - Signals are inverted and the baseline is subtracted. Time zero is the launch, taken as the half-height crossing of the pickup of the pulse on the right wall. Runs without a pulse or without a peak are skipped; 18 floor voltages remain.
  - Each floor voltage is compared with simulations through the median arrival time (t50), the width (t84 − t16), and the chi2 of the normalised peak shape after aligning t50.
  - Timing residuals are fitted with `dt50 = offset + scale * t50_sim`. The scale gives an effective flight distance of `1.05 m * (1 + scale)` (the real distance of the temporary set-up is only known to be 1.0–1.2 m).
  - The fit is done twice: once with a free offset, and once with offset = 0, i.e. the launch at the pulse pickup (columns `distance_offset0_m` and `rms_offset0_ns`).
  - Because the pickup measures the launch time, a model that needs a large offset to match the timing is inconsistent, whatever distance it prefers.
  - `pbar_configurations.csv` lists all configurations; `pbar_best.csv` has the configuration with the best shape for each model.
- **Ar40:**
  - The three runs are averaged; t = trigger + 10 us. The spectrum is fitted with non-negative amplitudes of the simulated species:
    - pbar–Ar40 fragments (Geant4 yields, 12 kV catching potential), with fixed relative yields and one amplitude;
    - H2+, He+, He2+ and Ar 1+ to 18+, each with a free amplitude.
  - The time calibration (launch delay after the trigger and effective flight distance) is fitted to the positions of the five main peaks (He2+, He+, Ar3+, Ar2+, Ar+), each produced by one gas ion. The spectrum is then fitted at that calibration.
  - A free scan of the calibration over 0.99–1.22 m is also reported (`*_free_scan` columns). Its minimum isn't always physical, because the many free amplitudes can compensate for a wrong time axis.
  - Simulated histograms (10 ns bins) are moved to the data bins by spreading each bin uniformly over its stretched width, which avoids aliasing spikes when the two grids don't line up.
  - He+ (m/q 4.003) and Ar10+ (3.996) can't be separated, so their amplitudes are degenerate.
- **First Ar40 peak** (`python benchmarks/plot_ar40_first_peak.py`, after `ar40_runs.py`):
  - The m/q ≈ 2 peak is fitted with He2+ and the fragment sum only, for the best cloud and the best plasma configuration, using the peak-position calibration.
  - The two configurations are re-simulated with more ions (cloud 10000×9, plasma 10000×3 per species). The cloud keeps only He2+ and the fragments, since its ions don't interact; the plasma keeps all species, which share the space charge.
  - Output: `ar40_first_peak.png`.
