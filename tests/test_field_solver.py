import numpy as np
import pytest
from scipy.special import ive

import calculate_tof as ct
from constants import VACUUM_PERMITTIVITY
from field_solver import AxisymmetricSolver, FieldGrid, validate_against_matrix
from geometry import Electrode, TrapGeometry
from trap import TTrap


def _single_electrode_solver(length=0.2, radius=0.01, grid=0.25e-3):
    geometry = TrapGeometry(electrodes=(Electrode("A", -length / 2, length / 2),), radius=radius)
    return AxisymmetricSolver(geometry, grid_spacing=grid)


@pytest.mark.parametrize("wavelength", [0.5, 0.02])
def test_laplace_matches_bessel_solution(wavelength):
    solver = _single_electrode_solver()
    L = solver.z[-1] - solver.z[0]
    k = np.pi * round(2 * L / wavelength) / L
    wall = np.sin(k * (solver.z - solver.z[0]))
    phi = solver.solve(wall)
    R = solver.geometry.radius
    exact = (ive(0, k * solver.r) / ive(0, k * R) * np.exp(k * (solver.r - R)))[:, None] * wall[None, :]
    # second order finite differences: error ~ (k h)^2 / 12
    assert np.abs(phi.values - exact).max() < 2 * (k * solver.dr)**2


def test_long_electrode_gives_its_voltage_in_the_middle():
    solver = _single_electrode_solver(length=0.2)
    phi = solver.laplace({"A": 7.0})
    assert phi(0.0, 0.0) == pytest.approx(7.0, abs=1e-6)
    assert phi.values[-1, np.argmin(np.abs(solver.z))] == 7.0


def test_superposition():
    geometry = TrapGeometry.from_ttrap(ct._trap_center_position(["P10", "P11", "P12"]))
    solver = AxisymmetricSolver(geometry)
    a = solver.laplace({"P10": 3.0, "P13": -2.0})
    b = solver.laplace({"P11": 5.0})
    both = solver.laplace({"P10": 3.0, "P13": -2.0, "P11": 5.0})
    np.testing.assert_allclose(both.values, a.values + b.values, atol=1e-10)


def test_poisson_uniformly_charged_cylinder():
    solver = _single_electrode_solver(length=0.3)
    rho = np.full((solver.n_r + 1, len(solver.z)), 2e-7)
    phi = solver.poisson(rho)
    middle = np.argmin(np.abs(solver.z))
    exact = 2e-7 * (solver.geometry.radius**2 - solver.r**2) / (4 * VACUUM_PERMITTIVITY)
    np.testing.assert_allclose(phi.values[:, middle], exact, rtol=1e-6, atol=1e-12)


def test_deposit_conserves_charge():
    solver = _single_electrode_solver()
    rng = np.random.default_rng(0)
    r = rng.uniform(0, 0.009, 1000)
    z = rng.uniform(-0.05, 0.05, 1000)
    charge = rng.uniform(1e-15, 2e-15, 1000)
    density = solver.deposit(r, z, charge)
    assert np.sum(density * solver.node_volumes()) == pytest.approx(charge.sum(), rel=1e-12)


def test_field_grid_interpolation_and_gradient_are_exact_for_linear_fields():
    r = np.linspace(0, 0.01, 11)
    z = np.linspace(-0.1, 0.1, 201)
    field = FieldGrid(r, z, 3.0 + 2.0 * r[:, None] ** 0 * z[None, :] * 50.0)
    assert field(np.array([0.0033]), np.array([0.0123]))[0] == pytest.approx(3.0 + 100.0 * 0.0123)
    _, d_dz = field.gradient()
    np.testing.assert_allclose(d_dz.values, 100.0)


def test_solver_reproduces_potential_matrix():
    center = ct._trap_center_position(["P10", "P11", "P12"])
    solver = AxisymmetricSolver(TrapGeometry.from_ttrap(center))
    report = validate_against_matrix(solver, center)
    # electrodes of the C and P traps: same peak value and integral as in the matrix
    for name in ["C3", "C10", "C17", "P3", "P7", "P11", "P13"]:
        assert abs(report[name]["peak_diff"]) < 2e-3, name
        assert abs(report[name]["integral_rel_diff"]) < 5e-3, name
    # the C trap agrees point by point
    assert report["C10"]["max_abs_diff"] < 2e-3
    # the P trap in the matrix is shifted by about 1 mm with respect to the TTrap electrode positions
    trap = TTrap(position=-center)
    z_matrix = (trap.position + trap.dx * np.arange(trap.segments)) / 1000
    trap.SetElectrodeV("P11", 1.0)
    shifted = np.interp(z_matrix - 0.95e-3, solver.z, solver.laplace({"P11": 1.0}).on_axis())
    assert np.abs(shifted - trap.get_final_V()).max() < 2e-3
