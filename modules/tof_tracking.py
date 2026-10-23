'''
Time of flight simulation with the axisymmetric field solver, physical initial conditions and particle tracking.
'''
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np
import polars as pl

from geometry import TrapGeometry
from field_solver import AxisymmetricSolver, FieldGrid
from plasma import (Species, Particles, Well, PlasmaEquilibrium, PoissonBoltzmannSolver, find_well,
                    sample_cloud, sample_legacy, sample_plasma)
from constants import ELEMENTARY_CHARGE
from tracker import MagneticField, GuidingCenterTracker, BorisTracker, ARRIVED, TRAPPED, HIT_ELECTRODE, REFLECTED

MODELS = ("legacy", "cloud", "plasma")
SPACE_CHARGE = ("off", "initial", "dynamic")
TRACKERS = ("gc", "boris")


@dataclass
class TrackingSetup:
    '''
    Fields and equilibrium shared by all iterations.
    '''
    geometry:TrapGeometry
    solver:AxisymmetricSolver
    phi_prepare:FieldGrid
    phi_launch:FieldGrid
    well:Well
    B:MagneticField
    equilibrium:Optional[PlasmaEquilibrium]
    n_ions:np.ndarray # number of ions of each species represented by the simulation (used for space charge)


@dataclass
class TrackingOutput:
    times:pl.DataFrame # iteration, particle, species, t [s], status
    initial:pl.DataFrame # initial conditions of the first iteration
    setup:TrackingSetup


def prepare_fields(voltages_prepare:Mapping[str,float],
                   voltages_launch:Mapping[str,float],
                   center_position_mm:float,
                   electrode_radius_mm:float = 15.0,
                   grid_mm:float = 0.25) -> tuple[TrapGeometry,AxisymmetricSolver,FieldGrid,FieldGrid]:
    geometry = TrapGeometry.from_ttrap(center_position_mm, radius_mm=electrode_radius_mm)
    solver = AxisymmetricSolver(geometry, grid_spacing=grid_mm / 1000)
    return geometry, solver, solver.laplace(voltages_prepare), solver.laplace(voltages_launch)


def setup_tracking(species:Sequence[Species],
                   voltages_prepare:Mapping[str,float],
                   voltages_launch:Mapping[str,float],
                   center_position_mm:float,
                   model:str = "cloud",
                   temperature_K:float = 300.0,
                   plasma_radius_mm:float = 1.0,
                   n_charges:float = 1e5,
                   space_charge:str = "off",
                   B_T:float = 5.0,
                   B_profile:Optional[str] = None,
                   electrode_radius_mm:float = 15.0,
                   grid_mm:float = 0.25,
                   verbose_level:int = 0,
                   B_map:Optional[str] = None) -> TrackingSetup:
    geometry, solver, phi_prepare, phi_launch = prepare_fields(voltages_prepare, voltages_launch, center_position_mm,
                                                               electrode_radius_mm, grid_mm)
    well = find_well(phi_prepare)
    if B_map:
        B = MagneticField.from_field_map(B_map, center_position_mm)
    elif B_profile:
        B = MagneticField.from_csv(B_profile)
    else:
        B = MagneticField(B_T)
    total_charge_weight = sum(s.weight * s.charge for s in species)
    n_ions = np.array([n_charges * s.weight / total_charge_weight for s in species])
    equilibrium = None
    if model == "plasma":
        # without space charge the equilibrium is calculated for a negligible number of charges
        pb_charges = n_charges if space_charge != "off" else 1.0
        pb = PoissonBoltzmannSolver(solver, phi_prepare, well, species, pb_charges, temperature_K, float(B.magnitude(well.z_bottom)))
        equilibrium = pb.solve_for_radius(plasma_radius_mm / 1000)
        pb.check_confined(equilibrium)
        if verbose_level > 0:
            print(f"Plasma equilibrium: omega = {equilibrium.omega:.4e} rad/s, radius = {equilibrium.radius*1e3:.3f} mm, "
                  f"max density = {equilibrium.density_max:.3e} m^-3, max space charge potential = {equilibrium.phi_sc.values.max():.4f} V")
    if verbose_level > 0:
        print(f"Trap well: {well.z_left*1e3:.2f} mm to {well.z_right*1e3:.2f} mm, bottom {well.phi_bottom:.3f} V at {well.z_bottom*1e3:.2f} mm, depth {well.depth:.3f} V")
    return TrackingSetup(geometry, solver, phi_prepare, phi_launch, well, B, equilibrium, n_ions)


def sample_initial_conditions(setup:TrackingSetup,
                              species:Sequence[Species],
                              N_particles:int,
                              rng:np.random.Generator,
                              model:str = "cloud",
                              temperature_K:float = 300.0,
                              T_perp_K:Optional[float] = None,
                              cloud_radius_mm:float = 1.0,
                              z_left_wall:float = None,
                              spacecharge_min_V:Optional[float] = None,
                              spacecharge_max_V:Optional[float] = None,
                              thermalised:bool = True,
                              forward_only:bool = False) -> Particles:
    particles = []
    for index, s in enumerate(species):
        if model == "legacy":
            z, r, vz, v_perp = sample_legacy(setup.phi_prepare, setup.well, s, N_particles, rng, z_left_wall,
                                             spacecharge_min_V, spacecharge_max_V, thermalised, forward_only)
        elif model == "cloud":
            z, r, vz, v_perp = sample_cloud(setup.phi_prepare, setup.well, s, N_particles, temperature_K, cloud_radius_mm / 1000, rng, T_perp_K)
        elif model == "plasma":
            z, r, vz, v_perp = sample_plasma(setup.equilibrium, setup.phi_prepare, setup.well, s, index, N_particles, temperature_K, rng, T_perp_K)
        else:
            raise ValueError(f"Unknown model {model}, available: {MODELS}")
        charge = np.full(N_particles, setup.n_ions[index] * s.charge * ELEMENTARY_CHARGE / N_particles)
        particles.append(Particles(z, r, vz, v_perp, np.full(N_particles, index), charge))
    return Particles.concatenate(particles)


def simulate_tracking(species:Sequence[Species],
                      voltages_prepare:Mapping[str,float],
                      voltages_launch:Mapping[str,float],
                      center_position_mm:float,
                      distance_to_mcp_m:float = 1.05,
                      N_particles:int = 10_000,
                      iterations:int = 10,
                      model:str = "cloud",
                      temperature_K:float = 300.0,
                      T_perp_K:Optional[float] = None,
                      cloud_radius_mm:float = 1.0,
                      plasma_radius_mm:float = 1.0,
                      n_charges:float = 1e5,
                      space_charge:Optional[str] = None,
                      B_T:float = 5.0,
                      B_profile:Optional[str] = None,
                      B_map:Optional[str] = None,
                      pulse_rise_time_s:float = 0.0,
                      dt_s:float = 1e-9,
                      t_max_s:float = 1e-4,
                      grid_mm:float = 0.25,
                      electrode_radius_mm:float = 15.0,
                      tracker:str = "gc",
                      poisson_every:int = 5,
                      seed:Optional[int] = None,
                      z_left_wall_mm:Optional[float] = None,
                      spacecharge_min_V:Optional[float] = None,
                      spacecharge_max_V:Optional[float] = None,
                      thermalised:bool = True,
                      forward_only:bool = False,
                      verbose_level:int = 0) -> TrackingOutput:
    '''
    Simulate arrival times of all species at the detector (distance_to_mcp_m from the centre of the trap).
    z_left_wall_mm is the start of the left wall relative to the centre of the trap (legacy model only).
    '''
    if space_charge is None:
        space_charge = "dynamic" if model == "plasma" else "off"
    if space_charge not in SPACE_CHARGE:
        raise ValueError(f"Unknown space charge mode {space_charge}, available: {SPACE_CHARGE}")
    if tracker not in TRACKERS:
        raise ValueError(f"Unknown tracker {tracker}, available: {TRACKERS}")
    setup = setup_tracking(species, voltages_prepare, voltages_launch, center_position_mm, model, temperature_K,
                           plasma_radius_mm, n_charges, space_charge, B_T, B_profile, electrode_radius_mm, grid_mm, verbose_level,
                           B_map=B_map)
    if verbose_level > 0 and not setup.B.uniform:
        print(f"Magnetic field: {float(setup.B.B(0.0)):.4f} T in the trap, {float(setup.B.B(distance_to_mcp_m)):.4f} T at the detector "
              f"(positive towards the detector)")
    q_over_m_species = np.array([s.q_over_m for s in species])
    if tracker == "gc":
        engine = GuidingCenterTracker(setup.phi_prepare, setup.phi_launch, setup.geometry.radius, distance_to_mcp_m, setup.B,
                                      dt=dt_s, t_max=t_max_s, pulse_rise_time=pulse_rise_time_s,
                                      space_charge_solver=setup.solver if space_charge == "dynamic" else None,
                                      poisson_every=poisson_every)
    else:
        if space_charge == "dynamic":
            raise ValueError("Dynamic space charge is available only with the guiding centre tracker")
        # the time step must resolve the fastest cyclotron motion
        cyclotron = q_over_m_species.max() * max(abs(setup.B.B0), np.abs(setup.B.B_profile).max() if not setup.B.uniform else 0)
        engine = BorisTracker(setup.phi_prepare, setup.phi_launch, setup.geometry.radius, distance_to_mcp_m, setup.B,
                              dt=min(dt_s, 0.1 / cyclotron), t_max=t_max_s, pulse_rise_time=pulse_rise_time_s)
    seeds = np.random.SeedSequence(seed).spawn(iterations)
    times, initial = [], None
    for iteration in range(iterations):
        rng = np.random.default_rng(seeds[iteration])
        particles = sample_initial_conditions(setup, species, N_particles, rng, model, temperature_K, T_perp_K, cloud_radius_mm,
                                              None if z_left_wall_mm is None else z_left_wall_mm / 1000,
                                              spacecharge_min_V, spacecharge_max_V, thermalised, forward_only)
        q_over_m = q_over_m_species[particles.species]
        result = engine.track(particles, q_over_m) if tracker == "gc" else engine.track(particles, q_over_m, rng)
        times.append(pl.DataFrame({"iteration": np.full(len(particles), iteration),
                                   "particle": np.tile(np.arange(N_particles), len(species)),
                                   "species": particles.species,
                                   "z_i [mm]": particles.z * 1e3,
                                   "t [s]": result.t,
                                   "status": result.status}))
        if iteration == 0:
            initial = pl.DataFrame({"species": particles.species, "z [mm]": particles.z * 1e3, "r [mm]": particles.r * 1e3,
                                    "vz [m/s]": particles.vz, "v_perp [m/s]": particles.v_perp})
        if verbose_level > 1:
            counts = np.bincount(result.status, minlength=4) / len(particles) * 100
            print(f"Iteration {iteration}: arrived {counts[ARRIVED]:.2f}%, trapped {counts[TRAPPED]:.2f}%, "
                  f"hit electrodes {counts[HIT_ELECTRODE]:.2f}%, reflected {counts[REFLECTED]:.2f}%")
    return TrackingOutput(times=pl.concat(times), initial=initial, setup=setup)
