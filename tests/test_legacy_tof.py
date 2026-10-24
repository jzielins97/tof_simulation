import numpy as np
import polars as pl
import pytest

import calculate_tof as ct
from constants import ATOMIC_UNIT_TO_KEV, SPEED_OF_LIGHT


def _linear_potential_particles(z_i_mm, dV_i, dx_mm=0.5, z_end_mm=20.0, V0=100.0, field_V_per_mm=2.0):
    '''Particles (one per initial condition) in a potential decreasing linearly along z.'''
    z = np.arange(-10.0, z_end_mm + dx_mm / 2, dx_mm)
    potential = pl.DataFrame({"z [mm]": z, "V_launch": V0 - field_V_per_mm * z})
    initial = pl.DataFrame({"iteration": [0] * len(z_i_mm),
                            "particle": list(range(len(z_i_mm))),
                            "z_i [mm]": z_i_mm,
                            "dV_i": dV_i}).with_columns(
        V_launch_i=V0 - field_V_per_mm * pl.col("z_i [mm]") + pl.col("dV_i"),
        V_prepare_i=pl.lit(0.0))
    return (potential.join(initial, how="cross")
            .filter((pl.col("z [mm]") >= pl.col("z_i [mm]")) | (pl.col("z_i [mm]").is_between(pl.col("z [mm]"), pl.col("z [mm]") + dx_mm)))
            .select("iteration", "particle", "z [mm]", "z_i [mm]", "V_launch", "V_launch_i", "V_prepare_i", "dV_i"))


@pytest.mark.parametrize("z_i,dV_i", [(0.13, 0.0), (0.13, 1e-6), (-3.71, 0.5), (5.0, 2.0), (12.26, 10.0)])
def test_times_are_exact_for_constant_field(z_i, dV_i):
    dx, z_end, field, distance_m = 0.5, 20.0, 2.0, 1.0
    mq_eV = 4.0 * ATOMIC_UNIT_TO_KEV * 1000
    particles = _linear_potential_particles([z_i], [dV_i], dx_mm=dx, z_end_mm=z_end, field_V_per_mm=field)
    particles = ct._calculate_velocities(particles, mq=mq_eV, dx_mm=dx)
    remaining = distance_m - particles["z [mm]"].max() / 1000
    times = ct._calculate_times(particles, remaining_distance_m=remaining, dx_mm=dx)

    # analytic solution: uniform acceleration up to the end of the potential and drift afterwards
    acceleration = field * 1e3 / mq_eV * SPEED_OF_LIGHT**2  # m/s^2
    v0 = np.sqrt(2 * dV_i / mq_eV) * SPEED_OF_LIGHT
    v_end = np.sqrt(v0**2 + 2 * acceleration * (z_end - z_i) / 1000)
    t_expected = (v_end - v0) / acceleration + (distance_m - z_end / 1000) / v_end
    assert times["t [s]"].item() == pytest.approx(t_expected, rel=1e-9)


def test_particle_which_cannot_escape_is_trapped():
    particles = _linear_potential_particles([0.0, 0.0], [0.5, 0.5])
    # add a barrier at z = 15 mm which is higher than the energy of the particles
    particles = particles.with_columns(pl.when(pl.col("z [mm]") == 15.0).then(pl.lit(1000.0)).otherwise(pl.col("V_launch")).alias("V_launch"))
    particles = ct._calculate_velocities(particles, mq=1e9, dx_mm=0.5)
    times = ct._calculate_times(particles, remaining_distance_m=1.0, dx_mm=0.5)
    assert times["trapped"].all()
    assert times["t [s]"].is_nan().all()


def test_tof_signal_averages_over_all_iterations():
    dt = 1e-9
    # iteration 0: two particles in bin 5, iteration 1: one particle in bin 5 and one in bin 7, iteration 2: one trapped
    times = pl.DataFrame({"iteration": [0, 0, 1, 1, 2, 2],
                          "particle": [0, 1, 0, 1, 0, 1],
                          "t [s]": [5.2e-9, 5.7e-9, 5.5e-9, 7.1e-9, 5.3e-9, float("nan")]})
    signal = ct._calculate_tof_signal(times, tof_range=[0, 1e-8], tof_dt=dt, iterations=3, N_particles=2,
                                      mq=1.0, label="test", weight=1.0)
    assert signal["t [s]"].to_list() == pytest.approx([5.5e-9, 7.5e-9])
    counts_bin5 = np.array([2, 1, 1])
    counts_bin7 = np.array([0, 1, 0])
    assert signal["signal"].to_list() == pytest.approx([counts_bin5.mean() / 2, counts_bin7.mean() / 2])
    assert signal["signal_err"].to_list() == pytest.approx([counts_bin5.std(ddof=1) / np.sqrt(3) / 2,
                                                            counts_bin7.std(ddof=1) / np.sqrt(3) / 2])
    assert signal.columns == ["t [s]", "signal", "signal_err", "label", "m/q", "weight"]


def test_electrode_voltages():
    electrodes = ["C0", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P14", "HV3"]
    prepare, launch = ct._electrode_voltages(electrodes, trap_floor_V=180, trap_wall_V=190, pulse_wall_to_V=5,
                                             trap_floor=["P10", "P11", "P12"], trap_right_wall=["P13"])
    assert prepare == {"C0": 190, "P7": 190, "P8": 190, "P9": 190, "P10": 180, "P11": 180, "P12": 180,
                       "P13": 190, "P14": 0.0, "HV3": 0.0}
    assert launch == {**prepare, "P13": 5}


def test_trap_center_position():
    # P11 is the middle electrode of the default floor
    assert ct._trap_center_position(["P10", "P11", "P12"]) == pytest.approx(701.75)
    # middle of the gap between P10 (682.0-694.5 mm) and P11 (695.5-708.0 mm)
    assert ct._trap_center_position(["P10", "P11"]) == pytest.approx(695.0)


def test_parse_args_conversions():
    args = ct.parse_args(["--trap_floor_V", "180", "--trap_wall_V", "190", "--thermalised", "false",
                          "--plot_mask", "0x13", "--savedata_mask", "0x05", "--savefig", "True", "--showfig", "0",
                          "--spacecharge_min_V", "None", "--spacecharge_max_V", "2.5", "tof", "--mq", "2"])
    assert args.thermalised is False
    assert args.plot_mask == 0x13 and args.savedata_mask == 0x05
    assert args.savefig is True and args.showfig is False
    assert args.spacecharge_min_V is None and args.spacecharge_max_V == 2.5
    assert ct.parse_args(["--trap_floor_V", "1", "--trap_wall_V", "2", "tof_calibration"]).command == "tof_calibration"


def test_full_simulation_is_normalised():
    np.random.seed(3)
    signal, peak = ct.calculate_tof(mq=2.0013, labels="He2", trap_floor_V=180, trap_wall_V=190,
                                    N_particles=2000, iterations=4, showfig=False)
    # every particle escapes the trap, so the signal sums to one
    assert signal["signal"].sum() == pytest.approx(1.0)
    assert 7e-6 < peak["tof_peak_s"].item() < 1e-5
