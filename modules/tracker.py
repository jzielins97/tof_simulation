'''
Tracking of particles released from the trap to the detector.

GuidingCenterTracker: motion of the guiding centres along the magnetic field lines (the cyclotron radius is much smaller
than the scale of the electric field), with force q E_z(r, z, t) - mu dB/dz, where mu = m v_perp^2 / (2 B) is conserved.
The radius of the guiding centre follows the field line, r^2 B = const. The integration uses the leapfrog
(kick-drift-kick) scheme with a fixed time step. After the particle leaves the region with electric field it drifts
to the detector, in a non-uniform magnetic field its parallel velocity changes due to conservation of mu.

BorisTracker: full 3D motion (Boris pusher) for validation of the guiding centre approximation.

The potential of the electrodes changes from the prepare to the launch configuration linearly within pulse_rise_time
starting at t = 0. Space charge of the particles can be included by solving the Poisson equation during the extraction.
'''
from dataclasses import dataclass
import csv

import numpy as np

from field_solver import FieldGrid, AxisymmetricSolver
from plasma import Particles

# status of the particles after tracking
ARRIVED = 0
TRAPPED = 1 # still in the trap after t_max
HIT_ELECTRODE = 2
REFLECTED = 3 # reflected by the magnetic mirror on the way to the detector


@dataclass
class MagneticField:
    '''
    Axial magnetic field Bz(z) on the axis (uniform when no profile is given).
    Off axis the paraxial approximation gives B_r = -(r/2) dBz/dz.
    '''
    B0:float # T, field used when there is no profile
    z_profile:np.ndarray = None # m, relative to the centre of the trap
    B_profile:np.ndarray = None # T

    @classmethod
    def from_csv(cls, path:str) -> "MagneticField":
        '''
        Profile from a csv file with columns z_m (relative to the trap centre) and Bz_T.
        '''
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        z = np.array([float(row["z_m"]) for row in rows])
        B = np.array([float(row["Bz_T"]) for row in rows])
        order = np.argsort(z)
        return cls(B0=float(np.interp(0.0, z[order], B[order])), z_profile=z[order], B_profile=B[order])

    @property
    def uniform(self) -> bool:
        return self.z_profile is None

    def B(self, z:np.ndarray) -> np.ndarray:
        if self.uniform:
            return np.full_like(np.asarray(z, dtype=float), self.B0)
        return np.interp(z, self.z_profile, self.B_profile)

    def dB_dz(self, z:np.ndarray) -> np.ndarray:
        if self.uniform:
            return np.zeros_like(np.asarray(z, dtype=float))
        return np.interp(z, self.z_profile, np.gradient(self.B_profile, self.z_profile))


@dataclass
class TrackingResult:
    t:np.ndarray # s, arrival time at the detector (NaN if the particle didn't arrive)
    status:np.ndarray # ARRIVED, TRAPPED, HIT_ELECTRODE or REFLECTED


def _exit_position(fields:list[FieldGrid], relative_threshold:float = 1e-6, margin_cells:int = 3) -> float:
    '''
    Position along the axis after which the electric potentials are negligible
    (a few cells after the last significant value, where the interpolated field vanishes).
    '''
    z_exit = fields[0].z[0]
    for phi in fields:
        V = np.abs(phi.on_axis())
        significant = np.nonzero(V > relative_threshold * max(V.max(), 1e-30))[0]
        if len(significant):
            z_exit = max(z_exit, phi.z[min(significant[-1] + margin_cells, len(phi.z) - 1)])
    return z_exit


def _drift_times(z_start:np.ndarray, v_par:np.ndarray, mu_over_m:np.ndarray, B:MagneticField, z_detector:float,
                 n_points:int = 2000, chunk:int = 200_000) -> np.ndarray:
    '''
    Time of the field-free drift from z_start to the detector. In a non-uniform magnetic field the parallel velocity
    follows from conservation of energy and magnetic moment: v_par^2(z) = v_par^2(z_start) - 2 mu/m (B(z) - B(z_start)).
    Returns NaN for particles reflected by the magnetic mirror.
    '''
    if B.uniform:
        return (z_detector - z_start) / v_par
    times = np.empty(len(z_start))
    s = np.linspace(0, 1, n_points)
    for begin in range(0, len(z_start), chunk):
        end = begin + chunk
        z = z_start[begin:end, None] + (z_detector - z_start[begin:end, None]) * s[None, :]
        v2 = v_par[begin:end, None]**2 - 2 * mu_over_m[begin:end, None] * (B.B(z) - B.B(z_start[begin:end, None]))
        reflected = np.any(v2 <= 0, axis=1)
        inverse = 1 / np.sqrt(np.clip(v2, 1e-300, None))
        times[begin:end] = np.trapezoid(inverse, z, axis=1)
        times[begin:end][reflected] = np.nan
    return times


class _ElectricField:
    '''
    Axial and radial electric fields of the electrodes during the pulse and of the space charge.
    '''
    def __init__(self, phi_prepare:FieldGrid, phi_launch:FieldGrid, pulse_rise_time:float):
        self.Er_prepare, self.Ez_prepare = [g * -1.0 for g in phi_prepare.gradient()]
        self.Er_launch, self.Ez_launch = [g * -1.0 for g in phi_launch.gradient()]
        self.pulse_rise_time = pulse_rise_time
        self.Er_sc = None
        self.Ez_sc = None

    def switch(self, t:float) -> float:
        if self.pulse_rise_time <= 0:
            return 1.0
        return float(np.clip(t / self.pulse_rise_time, 0.0, 1.0))

    def set_space_charge(self, phi_sc:FieldGrid) -> None:
        self.Er_sc, self.Ez_sc = [g * -1.0 for g in phi_sc.gradient()]

    def Ez(self, r:np.ndarray, z:np.ndarray, t:float, radial:tuple[np.ndarray,np.ndarray] = None) -> np.ndarray:
        '''
        Axial field; radial can hold precomputed radial interpolation weights (FieldGrid.radial_cell) for the radii r.
        '''
        radial = self.Ez_launch.radial_cell(r) if radial is None else radial
        s = self.switch(t)
        E = self.Ez_launch(r, z, radial) if s == 1.0 else (1 - s) * self.Ez_prepare(r, z, radial) + s * self.Ez_launch(r, z, radial)
        if self.Ez_sc is not None:
            E = E + self.Ez_sc(r, z, radial)
        return E

    def Er(self, r:np.ndarray, z:np.ndarray, t:float) -> np.ndarray:
        s = self.switch(t)
        E = self.Er_launch(r, z) if s == 1.0 else (1 - s) * self.Er_prepare(r, z) + s * self.Er_launch(r, z)
        if self.Er_sc is not None:
            E = E + self.Er_sc(r, z)
        return E


class GuidingCenterTracker:
    def __init__(self,
                 phi_prepare:FieldGrid,
                 phi_launch:FieldGrid,
                 electrode_radius:float, # m
                 z_detector:float, # m
                 B:MagneticField,
                 dt:float = 1e-9, # s
                 t_max:float = 1e-4, # s
                 pulse_rise_time:float = 0.0, # s
                 space_charge_solver:AxisymmetricSolver = None, # solver used for dynamic space charge (None to switch off)
                 poisson_every:int = 5):
        self.field = _ElectricField(phi_prepare, phi_launch, pulse_rise_time)
        self.electrode_radius = electrode_radius
        self.z_detector = z_detector
        self.B = B
        self.dt = dt
        self.t_max = t_max
        self.space_charge_solver = space_charge_solver
        self.poisson_every = poisson_every
        self.z_exit = min(_exit_position([phi_prepare, phi_launch]), z_detector)

    def _update_space_charge(self, r:np.ndarray, z:np.ndarray, charge:np.ndarray) -> None:
        rho = self.space_charge_solver.deposit(r, z, charge)
        self.field.set_space_charge(self.space_charge_solver.poisson(rho))

    def track(self, particles:Particles, q_over_m:np.ndarray) -> TrackingResult:
        '''
        Track particles; q_over_m [C/kg] for each particle.
        '''
        n = len(particles)
        t_arrival = np.full(n, np.nan)
        status = np.full(n, TRAPPED)
        z = particles.z.astype(float).copy()
        vz = particles.vz.astype(float).copy()
        B_start = self.B.B(z)
        mu_over_m = particles.v_perp**2 / (2 * B_start)
        flux = particles.r**2 * B_start # r^2 B is conserved along the field line
        qm = np.asarray(q_over_m, dtype=float)
        uniform = self.B.uniform

        def radius(flux_now, z_now):
            return np.sqrt(flux_now / self.B.B(z_now))

        def hits_electrode(flux_now, mu_now, qm_now, z_now):
            B_now = self.B.B(z_now)
            cyclotron_radius = np.sqrt(2 * mu_now * B_now) / (qm_now * B_now)
            return (radius(flux_now, z_now) + cyclotron_radius >= self.electrode_radius) & (z_now < self.z_exit)

        lost = hits_electrode(flux, mu_over_m, qm, z)
        status[lost] = HIT_ELECTRODE
        # state of the particles which are still tracked
        active = np.nonzero(~lost)[0]
        z_a, vz_a, qm_a, mu_a, flux_a, charge_a = z[active], vz[active], qm[active], mu_over_m[active], flux[active], particles.charge[active]
        r_a = radius(flux_a, z_a) # constant in a uniform magnetic field
        base_a, weight_a = self.field.Ez_launch.radial_cell(r_a)

        def acceleration(z_now, t_now):
            if uniform:
                return qm_a * self.field.Ez(r_a, z_now, t_now, (base_a, weight_a))
            return qm_a * self.field.Ez(radius(flux_a, z_now), z_now, t_now) - mu_a * self.B.dB_dz(z_now)

        if self.space_charge_solver is not None and len(active):
            self._update_space_charge(r_a if uniform else radius(flux_a, z_a), z_a, charge_a)
        t = 0.0
        a = acceleration(z_a, t)
        step = 0
        while len(active) and t < self.t_max:
            vz_a = vz_a + 0.5 * self.dt * a
            z_previous = z_a
            z_a = z_a + self.dt * vz_a
            t += self.dt
            step += 1
            if self.space_charge_solver is not None and step % self.poisson_every == 0:
                self._update_space_charge(r_a if uniform else radius(flux_a, z_a), z_a, charge_a)
            a = acceleration(z_a, t)
            vz_a = vz_a + 0.5 * self.dt * a
            # particles leaving the electric field drift to the detector
            exited = z_a >= self.z_exit
            done = exited.any()
            if done:
                index = active[exited]
                # time when the particle crossed z_exit (linear within the step)
                t_cross = t - self.dt * (z_a[exited] - self.z_exit) / np.maximum(z_a[exited] - z_previous[exited], 1e-30)
                drift = _drift_times(np.full(len(index), self.z_exit), vz_a[exited], mu_a[exited], self.B, self.z_detector)
                t_arrival[index] = t_cross + drift
                status[index] = np.where(np.isnan(drift), REFLECTED, ARRIVED)
            if not uniform:
                # in a non-uniform field the guiding centre and the cyclotron radius change along the field line
                lost = ~exited & hits_electrode(flux_a, mu_a, qm_a, z_a)
                if lost.any():
                    status[active[lost]] = HIT_ELECTRODE
                    exited = exited | lost
                    done = True
            if done:
                keep = ~exited
                active, z_a, vz_a, a = active[keep], z_a[keep], vz_a[keep], a[keep]
                qm_a, mu_a, flux_a, r_a, charge_a = qm_a[keep], mu_a[keep], flux_a[keep], r_a[keep], charge_a[keep]
                base_a, weight_a = base_a[keep], weight_a[keep]
        return TrackingResult(t=t_arrival, status=status)


class BorisTracker:
    '''
    Full 3D tracking with the Boris pusher (for validation, the time step must resolve the cyclotron motion).
    '''
    def __init__(self,
                 phi_prepare:FieldGrid,
                 phi_launch:FieldGrid,
                 electrode_radius:float,
                 z_detector:float,
                 B:MagneticField,
                 dt:float = 1e-10,
                 t_max:float = 1e-4,
                 pulse_rise_time:float = 0.0):
        self.field = _ElectricField(phi_prepare, phi_launch, pulse_rise_time)
        self.electrode_radius = electrode_radius
        self.z_detector = z_detector
        self.B = B
        self.dt = dt
        self.t_max = t_max
        self.z_exit = min(_exit_position([phi_prepare, phi_launch]), z_detector)

    def _fields(self, x, y, z, t):
        r = np.hypot(x, y)
        safe_r = np.where(r > 0, r, 1.0)
        Er = self.field.Er(r, z, t)
        Ez = self.field.Ez(r, z, t)
        Bz = self.B.B(z)
        Br = -0.5 * r * self.B.dB_dz(z)
        E = np.stack([Er * x / safe_r, Er * y / safe_r, Ez])
        Bv = np.stack([Br * x / safe_r, Br * y / safe_r, Bz])
        return E, Bv

    def track(self, particles:Particles, q_over_m:np.ndarray, rng:np.random.Generator = None) -> TrackingResult:
        '''
        The particles start on a cyclotron orbit around the guiding centre (random gyration phase).
        '''
        rng = np.random.default_rng() if rng is None else rng
        n = len(particles)
        qm = np.asarray(q_over_m, dtype=float)
        B_start = self.B.B(particles.z)
        cyclotron_frequency = qm * B_start
        # guiding centre at (r, 0, z), velocity perpendicular to B with random direction
        phase = rng.uniform(0, 2 * np.pi, n)
        v = np.stack([particles.v_perp * np.cos(phase), particles.v_perp * np.sin(phase), particles.vz.astype(float)])
        # x = x_gc - (v x b) / Omega for a positive charge
        x = np.stack([particles.r - v[1] / cyclotron_frequency, v[0] / cyclotron_frequency, particles.z.astype(float)])
        t_arrival = np.full(n, np.nan)
        status = np.full(n, TRAPPED)
        active = np.ones(n, dtype=bool)
        t = 0.0
        while active.any() and t < self.t_max:
            index = np.nonzero(active)[0]
            xa, va, qma = x[:, index], v[:, index], qm[index]
            E, Bv = self._fields(xa[0], xa[1], xa[2], t + 0.5 * self.dt)
            v_minus = va + 0.5 * self.dt * qma * E
            tv = 0.5 * self.dt * qma * Bv
            sv = 2 * tv / (1 + np.sum(tv**2, axis=0))
            v_prime = v_minus + np.cross(v_minus, tv, axis=0)
            v_plus = v_minus + np.cross(v_prime, sv, axis=0)
            va = v_plus + 0.5 * self.dt * qma * E
            z_previous = xa[2].copy()
            xa = xa + self.dt * va
            t += self.dt
            x[:, index], v[:, index] = xa, va
            exited = xa[2] >= self.z_exit
            if exited.any():
                ei = index[exited]
                B_exit = self.B.B(xa[2][exited])
                v_perp2 = va[0][exited]**2 + va[1][exited]**2
                t_cross = t - self.dt * (xa[2][exited] - self.z_exit) / np.maximum(xa[2][exited] - z_previous[exited], 1e-30)
                drift = _drift_times(np.full(len(ei), self.z_exit), va[2][exited], v_perp2 / (2 * B_exit), self.B, self.z_detector)
                t_arrival[ei] = t_cross + drift
                status[ei] = np.where(np.isnan(drift), REFLECTED, ARRIVED)
                active[ei] = False
            lost = np.hypot(xa[0], xa[1]) >= self.electrode_radius
            if lost.any():
                status[index[lost & ~exited]] = HIT_ELECTRODE
                active[index[lost]] = False
        return TrackingResult(t=t_arrival, status=status)
