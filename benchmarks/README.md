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
  - Timing residuals are fitted with `dt50 = offset + scale * t50_sim`. The scale gives an effective flight distance of `1.05 m * (1 + scale)`.
- **Ar40:**
  - The three runs are averaged; t = trigger + 10 us. The spectrum is fitted with non-negative amplitudes of the simulated species:
    - pbar–Ar40 fragments (Geant4 yields, 12 kV catching potential), with fixed relative yields and one amplitude;
    - H2+, He+, He2+ and Ar 1+ to 18+, each with a free amplitude.
  - A common time shift and a time scale (effective flight distance) are fitted.
  - He+ (m/q 4.003) and Ar10+ (3.996) can't be separated, so their amplitudes are degenerate.
