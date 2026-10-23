import numpy as np
import pytest

import calculate_tof as ct
from constants import ATOMIC_MASS_UNIT_KG, BOLTZMAN_CONSTANT_EV, ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY
from field_solver import AxisymmetricSolver, FieldGrid
from geometry import Electrode, TrapGeometry
from plasma import (PoissonBoltzmannSolver, Species, find_well, sample_cloud, sample_legacy, sample_plasma)
from trap import TTrap

FLOOR = ["P10", "P11", "P12"]


@pytest.fixture(scope="module")
def trap_fields():
    center = ct._trap_center_position(FLOOR)
    geometry = TrapGeometry.from_ttrap(center)
    solver = AxisymmetricSolver(geometry)
    prepare, _ = ct._electrode_voltages(geometry.names, 180, 190, 0, FLOOR, ["P13"])
    phi = solver.laplace(prepare)
    return solver, phi, find_well(phi), center


def _synthetic_well(depth=10.0, width=0.01):
    '''Potential A (1 - exp(-z^2 / 2 w^2)) independent of r (harmonic close to the centre).'''
    geometry = TrapGeometry(electrodes=(Electrode("A", -0.1, 0.1),), radius=0.01)
    solver = AxisymmetricSolver(geometry, grid_spacing=0.1e-3, z_margin=0.0)
    values = np.repeat((depth * (1 - np.exp(-solver.z**2 / (2 * width**2))))[None, :], solver.n_r + 1, axis=0)
    return solver, FieldGrid(solver.r, solver.z, values)


def test_find_well(trap_fields):
    _, phi, well, _ = trap_fields
    assert well.z_left < well.z_bottom < well.z_right
    assert 180 < well.phi_bottom < well.phi_barrier < 190
    assert phi(0.0, well.z_right) == pytest.approx(well.phi_barrier)
    # barrier is the maximum between the bottom and the end of the right wall
    z = phi.z
    right = (z > well.z_bottom) & (z < 0.05)
    assert phi.on_axis()[right].max() == pytest.approx(well.phi_barrier)


def test_cloud_in_harmonic_well():
    depth, width = 10.0, 0.01
    _, phi = _synthetic_well(depth, width)
    well = find_well(phi)
    species = Species("test", mq=4.0, charge=1.0)
    kappa = depth / width**2 # V/m^2
    temperature = 0.1 / BOLTZMAN_CONSTANT_EV # kT/q = 0.1 V
    z, r, vz, v_perp = sample_cloud(phi, well, species, 40_000, temperature, 1e-3, np.random.default_rng(1))
    assert z.std() == pytest.approx(np.sqrt(0.1 / kappa), rel=0.03)
    assert vz.std() == pytest.approx(np.sqrt(0.1 * ELEMENTARY_CHARGE / species.mass), rel=0.02)
    assert np.sqrt(np.mean(r**2)) == pytest.approx(np.sqrt(2) * 1e-3, rel=0.03)
    assert np.mean(v_perp**2) == pytest.approx(2 * 0.1 * ELEMENTARY_CHARGE / species.mass, rel=0.03)


def test_cloud_particles_are_trapped(trap_fields):
    _, phi, well, _ = trap_fields
    species = Species("He2+", mq=2.0013, charge=2.0)
    # temperature comparable with the depth of the well
    temperature = well.depth * species.charge / BOLTZMAN_CONSTANT_EV / 3
    z, r, vz, v_perp = sample_cloud(phi, well, species, 5000, temperature, 1e-3, np.random.default_rng(2))
    energy = phi(r, z) + 0.5 * vz**2 / species.q_over_m
    # particles move along the field lines, so the barrier is taken at their radius (higher off axis)
    barrier = phi(r, np.full_like(r, well.z_right))
    assert np.all(energy < barrier)
    assert barrier.min() == pytest.approx(well.phi_barrier, abs=0.05)
    assert np.all((z > well.z_left) & (z < well.z_right))


def test_legacy_sampling(trap_fields):
    _, phi, well, center = trap_fields
    species = Species("He2+", mq=2.0013, charge=2.0)
    left = (TTrap(position=-center)._GetElectrode("P7").GetElectrodeStart()) / 1000
    z, r, vz, _ = sample_legacy(phi, well, species, 5000, np.random.default_rng(3), left, forward_only=True)
    assert np.all(vz >= 0) and np.all(r == 0)
    kinetic = 0.5 * vz**2 / species.q_over_m
    assert np.all(phi(r, z) + kinetic <= well.phi_barrier + 1e-9)
    _, _, vz_both, _ = sample_legacy(phi, well, species, 5000, np.random.default_rng(3), left)
    assert 0.45 < np.mean(vz_both < 0) < 0.55


def test_poisson_boltzmann_low_density_is_gaussian_cloud():
    _, phi = _synthetic_well()
    solver, _ = _synthetic_well()
    well = find_well(phi)
    species = Species("test", mq=4.0, charge=1.0)
    temperature, B, omega = 1000.0, 5.0, 1e5
    pb = PoissonBoltzmannSolver(solver, phi, well, [species], n_charges=1e-3, temperature_K=temperature, B=B)
    equilibrium = pb.solve(omega)
    cyclotron = ELEMENTARY_CHARGE * B / species.mass
    sigma = np.sqrt(BOLTZMAN_CONSTANT_EV * temperature * ELEMENTARY_CHARGE / (species.mass * omega * (cyclotron - omega)))
    assert equilibrium.radius == pytest.approx(2 * sigma, rel=0.02)
    assert np.abs(equilibrium.phi_sc.values).max() < 1e-6


def test_poisson_boltzmann_cold_plasma_has_brillouin_density(trap_fields):
    solver, phi, well, _ = trap_fields
    species = Species("He2+", mq=2.0013, charge=2.0)
    n_charges, temperature, B = 1e6, 100.0, 5.0
    pb = PoissonBoltzmannSolver(solver, phi, well, [species], n_charges=n_charges, temperature_K=temperature, B=B)
    equilibrium = pb.solve_for_radius(1e-3)
    pb.check_confined(equilibrium)
    assert equilibrium.radius == pytest.approx(1e-3, rel=0.01)
    # total charge is conserved
    density = np.exp(equilibrium.log_density[0])
    assert np.sum(density * pb.volumes) * species.charge == pytest.approx(n_charges, rel=1e-9)
    # inside a cold plasma in global thermal equilibrium the density is uniform: n = 2 eps0 m omega (Omega - omega) / q^2
    q = species.charge * ELEMENTARY_CHARGE
    omega = equilibrium.omega
    brillouin = 2 * VACUUM_PERMITTIVITY * species.mass * omega * (q * B / species.mass - omega) / q**2
    assert equilibrium.density_max == pytest.approx(brillouin, rel=0.05)
    z, r, _, _ = sample_plasma(equilibrium, phi, well, species, 0, 5000, temperature, np.random.default_rng(4))
    assert np.sqrt(2 * np.mean(r**2)) == pytest.approx(1e-3, rel=0.05)


def test_plasma_which_does_not_fit_into_the_well_is_rejected(trap_fields):
    solver, phi, well, _ = trap_fields
    pb = PoissonBoltzmannSolver(solver, phi, well, [Species("He2+", 2.0013, 2.0)], n_charges=1e8, temperature_K=1000, B=5.0)
    with pytest.raises(ValueError):
        pb.check_confined(pb.solve_for_radius(2e-3))
