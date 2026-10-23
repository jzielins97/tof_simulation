'''
Initial conditions of the particles stored in the trap.

Three models are available:
 - legacy: generalised normal distribution in z and uniform energy between space charge limits (as in the 1D simulation),
 - cloud: non-interacting particles in thermal equilibrium (Debye length larger than the cloud), Boltzmann distribution
   in the trap potential along the field lines and a Gaussian radial profile,
 - plasma: global thermal equilibrium of a (multi-species) non-neutral plasma, the self-consistent space charge potential
   is found by solving the Poisson-Boltzmann equation. All species share the temperature T and the rotation frequency omega:
       n_s(r, z) ~ exp(-[q_s (phi_ext + phi_sc) + m_s omega (Omega_s - omega) r^2 / 2] / kT)
All positions/velocities are in SI units and the radial position is the guiding centre radius.
Potentials are positive for the trapped particles (the sign of the charge is absorbed in the potentials).
'''
from dataclasses import dataclass, field
from collections.abc import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.stats import gennorm

from constants import ATOMIC_MASS_UNIT_KG, ELEMENTARY_CHARGE, BOLTZMAN_CONSTANT_EV, VACUUM_PERMITTIVITY
from field_solver import FieldGrid, AxisymmetricSolver


@dataclass
class Species:
    label:str
    mq:float # m/q [u/e]
    charge:float = 1.0 # charge state [e]
    weight:float = 1.0 # relative number of ions (yield)

    @property
    def mass(self) -> float:
        return self.mq * self.charge * ATOMIC_MASS_UNIT_KG # kg

    @property
    def q_over_m(self) -> float:
        return ELEMENTARY_CHARGE / (self.mq * ATOMIC_MASS_UNIT_KG) # C/kg


@dataclass
class Particles:
    '''
    Initial conditions of the macro-particles.
    '''
    z:np.ndarray # m
    r:np.ndarray # m, guiding centre radius
    vz:np.ndarray # m/s
    v_perp:np.ndarray # m/s, cyclotron velocity
    species:np.ndarray # index of the species
    charge:np.ndarray # C, charge represented by each macro-particle (used for space charge)

    def __len__(self) -> int:
        return len(self.z)

    @staticmethod
    def concatenate(particles:Sequence["Particles"]) -> "Particles":
        return Particles(*[np.concatenate([getattr(p, name) for p in particles]) for name in ["z","r","vz","v_perp","species","charge"]])


@dataclass
class Well:
    '''
    Potential well on the trap axis.
    '''
    z_left:float # m, left edge of the well (potential equal to the barrier)
    z_right:float # m, right edge of the well (position of the barrier)
    z_bottom:float # m
    phi_bottom:float # V
    phi_barrier:float # V

    @property
    def depth(self) -> float:
        return self.phi_barrier - self.phi_bottom


def find_well(phi:FieldGrid, z_center:float = 0.0, tolerance:float = 1e-9) -> Well:
    '''
    Find the potential well on axis around z_center. The depth of the well is set by the lower of the two barriers.
    '''
    V = phi.on_axis()
    z = phi.z
    j = int(np.argmin(np.abs(z - z_center)))
    # go down to the local minimum
    while j > 0 and V[j - 1] < V[j]:
        j -= 1
    while j < len(V) - 1 and V[j + 1] < V[j]:
        j += 1
    j_bottom = j
    # climb to both barriers
    j_right = j_bottom
    while j_right < len(V) - 1 and V[j_right + 1] >= V[j_right] - tolerance:
        j_right += 1
    j_left = j_bottom
    while j_left > 0 and V[j_left - 1] >= V[j_left] - tolerance:
        j_left -= 1
    barrier = min(V[j_left], V[j_right])
    # edges of the well (between the barriers)
    left, right = j_bottom, j_bottom
    while left > j_left and V[left - 1] <= barrier:
        left -= 1
    while right < j_right and V[right + 1] <= barrier:
        right += 1
    return Well(z_left=z[left], z_right=z[right], z_bottom=z[j_bottom], phi_bottom=V[j_bottom], phi_barrier=barrier)


def _sample_cells(weights:np.ndarray, n:int, rng:np.random.Generator) -> tuple[np.ndarray,np.ndarray]:
    '''
    Sample n cells of a 2D array with probability proportional to weights.
    '''
    cumulative = np.cumsum(weights.ravel())
    if cumulative[-1] <= 0:
        raise ValueError("Distribution of particles is empty")
    index = np.searchsorted(cumulative, rng.uniform(0, cumulative[-1], n), side='right')
    index = np.minimum(index, cumulative.size - 1)
    return np.unravel_index(index, weights.shape)


def _cell_centres(phi:FieldGrid, well:Well, r_max:float) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    '''
    Centres of the grid cells inside the well (radius below r_max) and the potential in them.
    '''
    r_c = phi.r[:-1] + phi.dr / 2
    r_c = r_c[r_c < r_max]
    z_c = phi.z[:-1] + phi.dz / 2
    z_c = z_c[(z_c >= well.z_left) & (z_c <= well.z_right)]
    R, Z = np.meshgrid(r_c, z_c, indexing='ij')
    return r_c, z_c, phi(R, Z)


def _thermal_velocities(species:Species, n:int, T_par:float, T_perp:float, rng:np.random.Generator) -> tuple[np.ndarray,np.ndarray]:
    kT_par = BOLTZMAN_CONSTANT_EV * T_par * ELEMENTARY_CHARGE # J
    kT_perp = BOLTZMAN_CONSTANT_EV * T_perp * ELEMENTARY_CHARGE # J
    vz = rng.normal(0.0, np.sqrt(kT_par / species.mass), n)
    # two dimensional Maxwell distribution -> exponential distribution of the perpendicular energy
    v_perp = np.sqrt(2 * rng.exponential(kT_perp, n) / species.mass) if kT_perp > 0 else np.zeros(n)
    return vz, v_perp


def _sample_from_density(phi_total:FieldGrid,
                         well:Well,
                         species:Species,
                         log_density:np.ndarray, # on cell centres from _cell_centres
                         r_c:np.ndarray,
                         z_c:np.ndarray,
                         n:int,
                         T_par:float,
                         T_perp:float,
                         rng:np.random.Generator,
                         max_tries:int = 1000) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    '''
    Sample positions from the density (with the volume element 2 pi r dr dz) and Maxwell velocities.
    Particles with axial energy above the barrier of the well are not trapped and are sampled again.
    '''
    weights = r_c[:, None] * np.exp(log_density - log_density.max())
    barrier = phi_total(r_c, np.full_like(r_c, well.z_right)) # barrier for each radius
    z = np.empty(0); r = np.empty(0); vz = np.empty(0); v_perp = np.empty(0)
    for _ in range(max_tries):
        missing = n - len(z)
        if missing <= 0:
            break
        i, j = _sample_cells(weights, missing, rng)
        r_new = r_c[i] + (rng.uniform(size=missing) - 0.5) * phi_total.dr
        z_new = z_c[j] + (rng.uniform(size=missing) - 0.5) * phi_total.dz
        vz_new, v_perp_new = _thermal_velocities(species, missing, T_par, T_perp, rng)
        axial_energy = phi_total(r_new, z_new) + 0.5 * vz_new**2 / species.q_over_m # V (energy per charge)
        trapped = (axial_energy < barrier[i]) & (z_new > well.z_left) & (z_new < well.z_right)
        z = np.append(z, z_new[trapped]); r = np.append(r, r_new[trapped])
        vz = np.append(vz, vz_new[trapped]); v_perp = np.append(v_perp, v_perp_new[trapped])
    if len(z) < n:
        raise RuntimeError(f"Couldn't sample trapped particles of {species.label}: temperature too high for the depth of the well")
    return z[:n], r[:n], vz[:n], v_perp[:n]


def sample_cloud(phi:FieldGrid,
                 well:Well,
                 species:Species,
                 n:int,
                 temperature_K:float,
                 cloud_radius:float, # m, sigma of the Gaussian radial profile
                 rng:np.random.Generator,
                 T_perp_K:float = None) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    '''
    Non-interacting particles in thermal equilibrium: along the field lines the density follows the Boltzmann distribution
    exp(-q [phi(r, z) - phi_min(r)] / kT), the radial profile is Gaussian with sigma = cloud_radius.
    '''
    T_perp_K = temperature_K if T_perp_K is None else T_perp_K
    kT_V = BOLTZMAN_CONSTANT_EV * temperature_K / species.charge # V
    r_c, z_c, V = _cell_centres(phi, well, r_max=min(phi.r[-1], 6 * cloud_radius))
    log_density = -(V - V.min(axis=1, keepdims=True)) / kT_V - r_c[:, None]**2 / (2 * cloud_radius**2)
    return _sample_from_density(phi, well, species, log_density, r_c, z_c, n, temperature_K, T_perp_K, rng)


def sample_legacy(phi:FieldGrid,
                  well:Well,
                  species:Species,
                  n:int,
                  rng:np.random.Generator,
                  z_left_wall:float,
                  spacecharge_min_V:float = None,
                  spacecharge_max_V:float = None,
                  thermalised:bool = True,
                  forward_only:bool = False) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    '''
    Initial conditions of the 1D model (on axis): z from the generalised normal distribution inside the filled part of the trap,
    kinetic energy uniform between the flattened bottom (spacecharge_min) and spacecharge_max above the bottom of the well.
    The kinetic energy includes the space charge offset, which is released at launch as in the 1D model.
    Particles move initially towards the detector or backwards with equal probability (all forward when forward_only).
    '''
    V = phi.on_axis()
    z = phi.z
    full_trap = (z >= z_left_wall) & (z <= well.z_right) & (V <= well.phi_barrier)
    V_min, V_max = V[full_trap].min(), V[full_trap].max()
    sc_min = V_min if spacecharge_min_V is None else min(V_min + spacecharge_min_V, V_max)
    sc_max = V_max if spacecharge_max_V is None else min(V_min + spacecharge_max_V, V_max)
    filled = full_trap & (V <= sc_max)
    z_filled = z[filled]
    width = z_filled.max() - z_filled.min()
    loc = (z_filled.max() + z_filled.min()) / 2
    beta, scale = (10, width / 2.5) if thermalised else (2, width / 6)
    z_i = gennorm.rvs(beta=beta, loc=loc, scale=scale, size=n, random_state=rng)
    outside = (z_i < z_filled.min()) | (z_i > z_filled.max())
    while outside.any():
        z_i[outside] = gennorm.rvs(beta=beta, loc=loc, scale=scale, size=outside.sum(), random_state=rng)
        outside = (z_i < z_filled.min()) | (z_i > z_filled.max())
    V_i = np.interp(z_i, z, V)
    floor = np.maximum(V_i, sc_min)
    kinetic_V = (floor - V_i) + rng.uniform(0, 1, n) * np.clip(sc_max - floor, 0, None) # V
    speed = np.sqrt(2 * kinetic_V * species.q_over_m)
    direction = np.ones(n) if forward_only else rng.choice([-1.0, 1.0], n)
    return z_i, np.zeros(n), direction * speed, np.zeros(n)


@dataclass
class PlasmaEquilibrium:
    '''
    Solution of the Poisson-Boltzmann equation.
    '''
    phi_sc:FieldGrid # space charge potential [V] on the solver grid
    log_density:list[np.ndarray] # log of the density of each species on the cell centres
    r_c:np.ndarray
    z_c:np.ndarray
    omega:float # rad/s, rotation frequency
    radius:float # m, sqrt(2 <r^2>)
    density_max:float # 1/m^3, maximum total density
    iterations:int
    history:list = field(default_factory=list)


class PoissonBoltzmannSolver:
    '''
    Self-consistent space charge of a plasma in global thermal equilibrium.
    The equation is solved with Newton iterations on a window of the grid around the well, where the space charge potential
    is zero on the electrodes and at the ends of the window. The Jacobian includes the local Debye shielding term.
    '''
    def __init__(self,
                 solver:AxisymmetricSolver,
                 phi_ext:FieldGrid,
                 well:Well,
                 species:Sequence[Species],
                 n_charges:float, # total number of elementary charges in the plasma
                 temperature_K:float,
                 B:float, # T
                 window_margin:float = 0.03 # m
                 ):
        self.solver = solver
        self.species = list(species)
        self.temperature_K = temperature_K
        self.B = B
        self.well = well
        # number of ions of each species (weights are the relative yields)
        total_charge_weight = sum(s.weight * s.charge for s in self.species)
        self.n_ions = np.array([n_charges * s.weight / total_charge_weight for s in self.species])
        # window of the grid
        z = solver.z
        self.j0 = max(int(np.searchsorted(z, well.z_left - window_margin)), 1)
        self.j1 = min(int(np.searchsorted(z, well.z_right + window_margin)), len(z) - 2)
        self.n_r = solver.n_r # unknowns i = 0 .. n_r-1 (electrode at n_r)
        self.n_zw = self.j1 - self.j0 + 1
        self.phi_ext = phi_ext.values[:self.n_r, self.j0:self.j1 + 1]
        r = solver.r[:self.n_r]
        self.r = np.repeat(r[:, None], self.n_zw, axis=1)
        z_w = z[self.j0:self.j1 + 1]
        self.inside_well = np.repeat(((z_w > well.z_left) & (z_w < well.z_right))[None, :], self.n_r, axis=0)
        self.volumes = solver.node_volumes()[:self.n_r, self.j0:self.j1 + 1]
        self.laplacian = self._build_laplacian()

    def _build_laplacian(self) -> sparse.csc_matrix:
        dr, dz = self.solver.dr, self.solver.dz
        n_r, n_z = self.n_r, self.n_zw
        index = np.arange(n_r * n_z).reshape(n_r, n_z)
        rows, cols, vals = [], [], []
        def add(mask, d_i, d_j, coefficient):
            i, j = np.nonzero(mask)
            rows.append(index[i, j]); cols.append(index[i + d_i, j + d_j]); vals.append(coefficient[i, j] if np.ndim(coefficient) else np.full(len(i), coefficient))
        I = np.repeat(np.arange(n_r)[:, None], n_z, axis=1)
        J = np.repeat(np.arange(n_z)[None, :], n_r, axis=0)
        safe_I = np.maximum(I, 1)
        diag = np.where(I == 0, -4 / dr**2, -2 / dr**2) - 2 / dz**2
        add(np.ones_like(I, dtype=bool), 0, 0, diag)
        add(I < n_r - 1, 1, 0, np.where(I == 0, 4 / dr**2, (1 + 1 / (2 * safe_I)) / dr**2))
        add(I > 0, -1, 0, (1 - 1 / (2 * safe_I)) / dr**2)
        add(J < n_z - 1, 0, 1, 1 / dz**2)
        add(J > 0, 0, -1, 1 / dz**2)
        return sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_r * n_z, n_r * n_z))

    def _log_densities(self, phi_sc:np.ndarray, omega:float, T:float) -> list[np.ndarray]:
        '''
        Logarithm of density [1/m^3] of each species on the window nodes (outside of the well the density is zero).
        '''
        kT = BOLTZMAN_CONSTANT_EV * T * ELEMENTARY_CHARGE # J
        logs = []
        for s, n_ions in zip(self.species, self.n_ions):
            q = s.charge * ELEMENTARY_CHARGE
            cyclotron = q * self.B / s.mass
            energy = q * (self.phi_ext + phi_sc) + 0.5 * s.mass * omega * (cyclotron - omega) * self.r**2
            exponent = np.where(self.inside_well, -energy / kT, -np.inf)
            exponent -= exponent.max()
            normalisation = np.sum(np.exp(exponent) * self.volumes)
            logs.append(exponent + np.log(n_ions / normalisation))
        return logs

    def _residual(self, phi_sc:np.ndarray, omega:float, T:float) -> tuple[np.ndarray,list[np.ndarray]]:
        '''
        Residual of the Poisson equation, L phi + rho / eps0, for the Boltzmann densities.
        '''
        logs = self._log_densities(phi_sc, omega, T)
        rho = sum(s.charge * ELEMENTARY_CHARGE * np.exp(log_n) for s, log_n in zip(self.species, logs))
        return self.laplacian @ phi_sc.ravel() + rho.ravel() / VACUUM_PERMITTIVITY, logs

    def _newton_step(self, residual:np.ndarray, logs:list[np.ndarray], T:float) -> np.ndarray:
        '''
        Newton step with the exact Jacobian. The density of species s is n_s = N_s g_s / sum(g_s V), so
        d rho_s / d phi = (q_s^2 e / kT) (-diag(n_s) + n_s (n_s V)^T / N_s): a local (Debye shielding) part and
        a rank-one part from the normalisation, which is included with the Woodbury formula.
        '''
        kT_eV = BOLTZMAN_CONSTANT_EV * T
        diagonal = np.zeros(self.n_r * self.n_zw)
        U, W = [], []
        for s, log_n, n_ions in zip(self.species, logs, self.n_ions):
            n = np.exp(log_n).ravel()
            factor = s.charge**2 * ELEMENTARY_CHARGE / kT_eV / VACUUM_PERMITTIVITY
            diagonal -= factor * n
            U.append(factor * n)
            W.append(n * self.volumes.ravel() / n_ions)
        lu = splu((self.laplacian + sparse.diags(diagonal)).tocsc())
        x = lu.solve(-residual)
        U = np.column_stack(U)
        W = np.column_stack(W)
        Y = lu.solve(U)
        correction = np.linalg.solve(np.eye(U.shape[1]) + W.T @ Y, W.T @ x)
        return (x - Y @ correction).reshape(self.n_r, self.n_zw)

    def _solve_at(self, phi_sc:np.ndarray, omega:float, T:float, tolerance:float, max_iterations:int) -> tuple[np.ndarray,int]:
        '''
        Newton iterations with backtracking line search (until the step is below tolerance * kT/e).
        '''
        kT_V = BOLTZMAN_CONSTANT_EV * T
        residual, logs = self._residual(phi_sc, omega, T)
        norm = np.linalg.norm(residual)
        for iteration in range(1, max_iterations + 1):
            step = self._newton_step(residual, logs, T)
            alpha = 1.0
            for _ in range(30):
                trial = phi_sc + alpha * step
                trial_residual, trial_logs = self._residual(trial, omega, T)
                trial_norm = np.linalg.norm(trial_residual)
                if trial_norm < (1 - 1e-4 * alpha) * norm:
                    break
                alpha /= 2
            phi_sc, residual, logs, norm = trial, trial_residual, trial_logs, trial_norm
            if alpha * np.abs(step).max() < tolerance * kT_V:
                return phi_sc, iteration
        raise RuntimeError(f"Poisson-Boltzmann iterations didn't converge for T={T:.1f} K, omega={omega:.3e} rad/s")

    def solve(self, omega:float, phi_sc_start:np.ndarray = None, tolerance:float = 1e-6, max_iterations:int = 100) -> PlasmaEquilibrium:
        '''
        Solve the Poisson-Boltzmann equation for the rotation frequency omega.
        Without a starting potential, the temperature is lowered gradually from the depth of the well to the requested temperature.
        '''
        phi_sc = np.zeros((self.n_r, self.n_zw)) if phi_sc_start is None else phi_sc_start
        T_target = self.temperature_K
        if phi_sc_start is None:
            T_start = max(T_target, self.well.depth / BOLTZMAN_CONSTANT_EV)
            temperatures = np.geomspace(T_start, T_target, max(int(np.ceil(np.log(T_start / T_target) / np.log(3))) + 1, 1))
        else:
            temperatures = [T_target]
        total_iterations = 0
        for T in temperatures:
            phi_sc, iterations = self._solve_at(phi_sc, omega, T, tolerance, max_iterations)
            total_iterations += iterations
        logs = self._log_densities(phi_sc, omega, T_target)
        return self._equilibrium(phi_sc, logs, omega, total_iterations)

    def check_confined(self, equilibrium:PlasmaEquilibrium, threshold:float = 1e-3) -> None:
        '''
        Raise an error if the plasma isn't confined by the well, i.e. the density at the edges of the well
        or at the electrodes isn't negligible (too many charges for the well).
        '''
        density = sum(np.exp(log_n) for log_n in equilibrium.log_density)
        edges = np.zeros_like(density, dtype=bool)
        z_w = equilibrium.z_c
        inside = (z_w > self.well.z_left) & (z_w < self.well.z_right)
        first, last = np.nonzero(inside)[0][[0, -1]]
        edges[:, [first, last]] = True
        edges[-1, :] = True
        if density[edges].max() > threshold * density.max():
            raise ValueError("Plasma is not confined by the trap: reduce the number of charges, increase the radius or the depth of the well")

    def _equilibrium(self, phi_sc:np.ndarray, logs:list[np.ndarray], omega:float, iterations:int) -> PlasmaEquilibrium:
        density = sum(np.exp(log_n) for log_n in logs)
        weights = density * self.volumes
        radius = np.sqrt(2 * np.sum(weights * self.r**2) / np.sum(weights))
        phi = np.zeros((self.solver.n_r + 1, len(self.solver.z)))
        phi[:self.n_r, self.j0:self.j1 + 1] = phi_sc
        return PlasmaEquilibrium(phi_sc=FieldGrid(self.solver.r, self.solver.z, phi),
                                 log_density=logs, r_c=self.solver.r[:self.n_r], z_c=self.solver.z[self.j0:self.j1 + 1],
                                 omega=omega, radius=radius, density_max=float(density.max()), iterations=iterations)

    def solve_for_radius(self, radius:float, relative_tolerance:float = 0.01, max_steps:int = 40) -> PlasmaEquilibrium:
        '''
        Find the rotation frequency giving the plasma radius sqrt(2 <r^2>) (edge of a uniform column) equal to radius.
        Larger rotation frequency compresses the plasma radially.
        '''
        kT = BOLTZMAN_CONSTANT_EV * self.temperature_K * ELEMENTARY_CHARGE
        q = min(s.charge for s in self.species) * ELEMENTARY_CHARGE
        cyclotron = min(s.charge * ELEMENTARY_CHARGE * self.B / s.mass for s in self.species)
        # cold limit of a Gaussian cloud: sigma^2 = kT / (q B omega), radius = 2 sigma
        omega_low = kT / (q * self.B * (self.solver.geometry.radius / 2)**2)
        omega_high = cyclotron / 2 # Brillouin limit
        if radius >= self.solver.geometry.radius:
            raise ValueError(f"Plasma radius {radius*1e3:.3f} mm must be smaller than the electrode radius")
        high = self.solve(omega_high)
        if high.radius > radius:
            raise ValueError(f"Plasma can't be compressed to {radius*1e3:.3f} mm (radius at the Brillouin limit is {high.radius*1e3:.3f} mm)")
        low = self.solve(omega_low)
        steps = 0
        best = high
        while steps < max_steps:
            omega = np.sqrt(low.omega * high.omega)
            middle = self.solve(omega, phi_sc_start=self._window(best.phi_sc))
            best = middle
            if abs(middle.radius / radius - 1) < relative_tolerance:
                return middle
            if middle.radius > radius:
                low = middle
            else:
                high = middle
            steps += 1
        return best

    def _window(self, phi:FieldGrid) -> np.ndarray:
        return phi.values[:self.n_r, self.j0:self.j1 + 1].copy()


def sample_plasma(equilibrium:PlasmaEquilibrium,
                  phi_ext:FieldGrid,
                  well:Well,
                  species:Species,
                  species_index:int,
                  n:int,
                  temperature_K:float,
                  rng:np.random.Generator,
                  T_perp_K:float = None) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    '''
    Sample particles of one species from the plasma equilibrium.
    '''
    T_perp_K = temperature_K if T_perp_K is None else T_perp_K
    phi_total = phi_ext + equilibrium.phi_sc
    # density is known on the nodes, sample around them (cells centred on nodes)
    log_density = equilibrium.log_density[species_index]
    r_nodes = equilibrium.r_c
    z_nodes = equilibrium.z_c
    inside = (z_nodes > well.z_left) & (z_nodes < well.z_right)
    z, r, vz, v_perp = _sample_from_density(phi_total, well, species, log_density[:, inside],
                                            r_nodes, z_nodes[inside], n, temperature_K, T_perp_K, rng)
    return z, np.abs(r), vz, v_perp
