import numpy as np
import pytest
from scipy.integrate import quad

import calculate_tof as ct
from constants import ELEMENTARY_CHARGE
from field_solver import AxisymmetricSolver, FieldGrid
from geometry import TrapGeometry
from plasma import Particles, PoissonBoltzmannSolver, Species, find_well, sample_cloud, sample_plasma
from tracker import (ARRIVED, HIT_ELECTRODE, BorisTracker, GuidingCenterTracker, MagneticField, _drift_times)

FLOOR = ["P10", "P11", "P12"]
SPECIES = Species("He2+", mq=2.0013016, charge=2.0)


def _ramp_field(field=1000.0, z_end=0.05):
    '''Potential decreasing linearly (uniform field) until z_end and zero afterwards.'''
    r = np.linspace(0, 0.015, 4)
    z = np.linspace(-0.05, 0.2, 25001)
    values = np.repeat((field * np.clip(z_end - z, 0, None))[None, :], len(r), axis=0)
    return FieldGrid(r, z, values)


def _particles(z, vz, r=None, v_perp=None):
    n = len(z)
    return Particles(z=np.asarray(z, float), r=np.zeros(n) if r is None else np.asarray(r, float), vz=np.asarray(vz, float),
                     v_perp=np.zeros(n) if v_perp is None else np.asarray(v_perp, float), species=np.zeros(n, dtype=int),
                     charge=np.zeros(n))


@pytest.fixture(scope="module")
def trap_fields():
    center = ct._trap_center_position(FLOOR)
    geometry = TrapGeometry.from_ttrap(center)
    solver = AxisymmetricSolver(geometry)
    prepare, launch = ct._electrode_voltages(geometry.names, 180, 190, 0, FLOOR, ["P13"])
    phi_prepare, phi_launch = solver.laplace(prepare), solver.laplace(launch)
    return solver, phi_prepare, phi_launch, find_well(phi_prepare)


def test_uniform_field_forward_and_backward():
    field, z_end, detector = 1000.0, 0.05, 1.0
    phi = _ramp_field(field, z_end)
    tracker = GuidingCenterTracker(phi, phi, 0.015, detector, MagneticField(5.0), dt=1e-10, t_max=1e-4)
    v0 = 2000.0
    result = tracker.track(_particles([0.0, 0.0], [v0, -v0]), np.full(2, SPECIES.q_over_m))
    a = SPECIES.q_over_m * field
    v_end = np.sqrt(v0**2 + 2 * a * z_end)
    t_forward = (v_end - v0) / a + (detector - z_end) / v_end
    assert np.all(result.status == ARRIVED)
    assert result.t[0] == pytest.approx(t_forward, rel=1e-5)
    # the particle moving backwards turns around and passes the start with the same speed
    assert result.t[1] - result.t[0] == pytest.approx(2 * v0 / a, abs=2e-11)


def test_guiding_centre_matches_full_motion(trap_fields):
    _, phi_prepare, phi_launch, well = trap_fields
    rng = np.random.default_rng(5)
    z, r, vz, v_perp = sample_cloud(phi_prepare, well, SPECIES, 20, 3000.0, 1e-3, rng)
    particles = Particles(z, r, vz, v_perp, np.zeros(20, dtype=int), np.zeros(20))
    q_over_m = np.full(20, SPECIES.q_over_m)
    B = MagneticField(5.0)
    gc = GuidingCenterTracker(phi_prepare, phi_launch, 0.015, 1.05, B, dt=1e-10).track(particles, q_over_m)
    boris = BorisTracker(phi_prepare, phi_launch, 0.015, 1.05, B, dt=5e-11).track(particles, q_over_m, rng)
    assert np.all(gc.status == ARRIVED) and np.all(boris.status == ARRIVED)
    assert np.abs(gc.t - boris.t).max() < 1e-9


def test_time_step_convergence(trap_fields):
    _, phi_prepare, phi_launch, well = trap_fields
    z, r, vz, v_perp = sample_cloud(phi_prepare, well, SPECIES, 200, 300.0, 1e-3, np.random.default_rng(6))
    particles = Particles(z, r, vz, v_perp, np.zeros(200, dtype=int), np.zeros(200))
    q_over_m = np.full(200, SPECIES.q_over_m)
    coarse = GuidingCenterTracker(phi_prepare, phi_launch, 0.015, 1.05, MagneticField(5.0), dt=1e-9).track(particles, q_over_m)
    fine = GuidingCenterTracker(phi_prepare, phi_launch, 0.015, 1.05, MagneticField(5.0), dt=1e-10).track(particles, q_over_m)
    assert np.abs(coarse.t - fine.t).max() < 0.1e-9


def test_slower_pulse_delays_particles(trap_fields):
    _, phi_prepare, phi_launch, well = trap_fields
    particles = _particles([well.z_bottom] * 3, [0.0] * 3)
    times = [GuidingCenterTracker(phi_prepare, phi_launch, 0.015, 1.05, MagneticField(5.0), pulse_rise_time=rise)
             .track(particles, np.full(3, SPECIES.q_over_m)).t[0] for rise in [0.0, 100e-9, 500e-9]]
    assert times[0] < times[1] < times[2]


def test_particles_outside_of_electrodes_are_lost(trap_fields):
    _, phi_prepare, phi_launch, well = trap_fields
    particles = _particles([0.0, 0.0], [0.0, 0.0], r=[0.001, 0.0149], v_perp=[0.0, 5e4])
    result = GuidingCenterTracker(phi_prepare, phi_launch, 0.015, 1.05, MagneticField(5.0)).track(particles, np.full(2, SPECIES.q_over_m))
    assert list(result.status) == [ARRIVED, HIT_ELECTRODE]


def test_magnetic_mirror_drift():
    B = MagneticField(5.0, z_profile=np.array([0.0, 0.2, 1.0]), B_profile=np.array([5.0, 5.0, 0.5]))
    v_par, v_perp = 1e4, 5e3
    mu_over_m = v_perp**2 / (2 * 5.0)
    t = _drift_times(np.array([0.1]), np.array([v_par]), np.array([mu_over_m]), B, 1.0)[0]
    expected, _ = quad(lambda z: 1 / np.sqrt(v_par**2 - 2 * mu_over_m * (B.B(z) - 5.0)), 0.1, 1.0, points=[0.2])
    assert t == pytest.approx(expected, rel=1e-5)
    # increasing field reflects particles with large perpendicular energy
    rising = MagneticField(5.0, z_profile=np.array([0.0, 1.0]), B_profile=np.array([5.0, 50.0]))
    assert np.isnan(_drift_times(np.array([0.1]), np.array([1e3]), np.array([mu_over_m]), rising, 1.0)[0])


def test_dynamic_space_charge_spreads_the_bunch(trap_fields):
    solver, phi_prepare, phi_launch, well = trap_fields
    pb = PoissonBoltzmannSolver(solver, phi_prepare, well, [SPECIES], n_charges=3e6, temperature_K=300, B=5.0)
    equilibrium = pb.solve_for_radius(1e-3)
    z, r, vz, v_perp = sample_plasma(equilibrium, phi_prepare, well, SPECIES, 0, 2000, 300, np.random.default_rng(7))
    charge = np.full(2000, pb.n_ions[0] * SPECIES.charge * ELEMENTARY_CHARGE / 2000)
    particles = Particles(z, r, vz, v_perp, np.zeros(2000, dtype=int), charge)
    q_over_m = np.full(2000, SPECIES.q_over_m)
    without = GuidingCenterTracker(phi_prepare, phi_launch, 0.015, 1.05, MagneticField(5.0)).track(particles, q_over_m)
    with_sc = GuidingCenterTracker(phi_prepare, phi_launch, 0.015, 1.05, MagneticField(5.0),
                                   space_charge_solver=solver).track(particles, q_over_m)
    # Coulomb repulsion during the extraction spreads the bunch (front is pushed forward, back is pushed back)
    assert np.all(with_sc.status == ARRIVED)
    assert np.std(with_sc.t) > 1.1 * np.std(without.t)
    assert np.mean(with_sc.t) == pytest.approx(np.mean(without.t), rel=0.01)


def test_tracking_engine_end_to_end():
    common = dict(trap_floor_V=180, trap_wall_V=190, N_particles=3000, iterations=2, showfig=False, seed=3)
    signal, peak = ct.calculate_tof(mq=2.0013016, labels="He2+", charges=2, engine="tracking", model="cloud", **common)
    assert signal["signal"].sum() == pytest.approx(1.0)
    assert 8.5e-6 < peak["tof_peak_s"].item() < 8.9e-6
    # legacy initial conditions moving forward reproduce the 1D engine (up to the 1 mm shift of the potential matrix)
    legacy_like, _ = ct.calculate_tof(mq=2.0013016, labels="He2+", charges=2, engine="tracking", model="legacy", forward_only=True, **common)
    np.random.seed(3)
    legacy, _ = ct.calculate_tof(mq=2.0013016, labels="He2+", trap_floor_V=180, trap_wall_V=190, N_particles=3000, iterations=2, showfig=False)
    mean = lambda s: (s["t [s]"] * s["signal"]).sum() / s["signal"].sum()
    assert mean(legacy_like) == pytest.approx(mean(legacy), rel=3e-3)
