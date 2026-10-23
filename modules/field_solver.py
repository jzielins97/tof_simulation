'''
Axisymmetric (r, z) solver of the Laplace and Poisson equations inside a cylindrical electrode stack.
For cylindrical electrodes this gives the full 3D potential, phi(x, y, z) = phi(r, z).

The equations are discretised with the five-point finite-difference stencil of the cylindrical Laplacian
on a uniform grid (r_i = i dr, z_j = z_0 + j dz):
    (phi[i+1] - 2 phi[i] + phi[i-1]) / dr^2 + (phi[i+1] - phi[i-1]) / (2 r_i dr) + (phi[j+1] - 2 phi[j] + phi[j-1]) / dz^2 = -rho / eps0
with 4 (phi[1] - phi[0]) / dr^2 for the radial part on the axis (regularity, dphi/dr = 0).
The potential at the electrode radius is set by the electrodes (linear in the gaps) and the ends of the tube are grounded.
The linear system is solved directly: the z part is diagonalised with the discrete sine transform (DST-I)
and for each z mode the radial tridiagonal system is solved with the Thomas algorithm.
'''
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.fft import dst

from constants import VACUUM_PERMITTIVITY
from geometry import TrapGeometry


@dataclass
class FieldGrid:
    '''
    Values of a scalar field (e.g. potential in V) on the uniform (r, z) grid with bilinear interpolation.
    '''
    r:np.ndarray # m, shape (nr,)
    z:np.ndarray # m, shape (nz,)
    values:np.ndarray # shape (nr, nz)

    @property
    def dr(self) -> float:
        return self.r[1] - self.r[0]

    @property
    def dz(self) -> float:
        return self.z[1] - self.z[0]

    def radial_cell(self, r:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        '''
        Index of the first value of the radial cell in the flattened values and the weight of the next row,
        can be reused for interpolation of fields on the same grid at constant radius.
        '''
        fr = np.clip(np.abs(np.asarray(r, dtype=float)) / self.dr, 0, len(self.r) - 1.000001)
        i = fr.astype(np.int64)
        return i * len(self.z), fr - i

    def __call__(self, r:np.ndarray, z:np.ndarray, radial:tuple[np.ndarray,np.ndarray] = None) -> np.ndarray:
        '''
        Bilinear interpolation (positions outside of the grid are clipped to its edge).
        '''
        base, ar = self.radial_cell(r) if radial is None else radial
        fz = np.clip((np.asarray(z, dtype=float) - self.z[0]) / self.dz, 0, len(self.z) - 1.000001)
        j = fz.astype(np.int64)
        az = fz - j
        index = base + j
        v = self.values.ravel()
        nz = len(self.z)
        v00 = v.take(index)
        v01 = v.take(index + 1)
        v10 = v.take(index + nz)
        v11 = v.take(index + nz + 1)
        return (1 - ar) * (v00 + az * (v01 - v00)) + ar * (v10 + az * (v11 - v10))

    def gradient(self) -> tuple["FieldGrid","FieldGrid"]:
        '''
        Derivatives d/dr and d/dz of the field (second order finite differences).
        '''
        d_dr, d_dz = np.gradient(self.values, self.dr, self.dz, edge_order=2)
        d_dr[0, :] = 0.0 # axial symmetry
        return FieldGrid(self.r, self.z, d_dr), FieldGrid(self.r, self.z, d_dz)

    def on_axis(self) -> np.ndarray:
        return self.values[0]

    def __add__(self, other:"FieldGrid") -> "FieldGrid":
        return FieldGrid(self.r, self.z, self.values + other.values)

    def __sub__(self, other:"FieldGrid") -> "FieldGrid":
        return FieldGrid(self.r, self.z, self.values - other.values)

    def __mul__(self, factor:float) -> "FieldGrid":
        return FieldGrid(self.r, self.z, self.values * factor)

    __rmul__ = __mul__


class AxisymmetricSolver:
    '''
    Fast direct solver of the five-point finite-difference Laplace/Poisson equations in the (r, z) plane.
    '''
    def __init__(self,
                 geometry:TrapGeometry,
                 grid_spacing:float = 0.25e-3, # m
                 z_margin:float = None # m, grounded tube length added on both sides of the stack (default 4 R)
                 ):
        self.geometry = geometry
        R = geometry.radius
        self.n_r = max(int(round(R / grid_spacing)), 2) # number of radial cells
        self.dr = R / self.n_r
        z_margin = 4 * R if z_margin is None else z_margin
        z_start = geometry.z_min - z_margin
        z_end = geometry.z_max + z_margin
        self.n_z = int(np.ceil((z_end - z_start) / grid_spacing)) - 1 # number of interior points along z
        self.dz = (z_end - z_start) / (self.n_z + 1)
        self.r = np.arange(self.n_r + 1) * self.dr # includes the electrode radius
        self.z = z_start + np.arange(self.n_z + 2) * self.dz # includes grounded ends
        # eigenvalues of the second difference along z with Dirichlet ends (DST-I modes)
        n = np.arange(1, self.n_z + 1)
        self._lambda_z = -4 / self.dz**2 * np.sin(np.pi * n / (2 * (self.n_z + 1)))**2
        # radial tridiagonal matrix (unknowns i = 0 .. n_r-1, i = n_r is the electrode)
        i = np.arange(self.n_r)
        self._lower = np.where(i > 0, (1 - 1 / (2 * np.maximum(i, 1))) / self.dr**2, 0.0)
        self._upper = np.where(i > 0, (1 + 1 / (2 * np.maximum(i, 1))) / self.dr**2, 4 / self.dr**2)
        self._diag = np.where(i > 0, -2 / self.dr**2, -4 / self.dr**2)
        self._node_volumes = None

    def _solve(self, rhs:np.ndarray) -> np.ndarray:
        '''
        Solve L phi = rhs for the interior unknowns, rhs has shape (n_r, n_z).
        '''
        rhs_hat = dst(rhs, type=1, axis=1, norm='ortho')
        # Thomas algorithm for every z mode at once
        n_r = self.n_r
        c_prime = np.empty((n_r, self.n_z))
        d_prime = np.empty((n_r, self.n_z))
        diag = self._diag[0] + self._lambda_z
        c_prime[0] = self._upper[0] / diag
        d_prime[0] = rhs_hat[0] / diag
        for i in range(1, n_r):
            denominator = self._diag[i] + self._lambda_z - self._lower[i] * c_prime[i - 1]
            c_prime[i] = self._upper[i] / denominator
            d_prime[i] = (rhs_hat[i] - self._lower[i] * d_prime[i - 1]) / denominator
        phi_hat = np.empty((n_r, self.n_z))
        phi_hat[-1] = d_prime[-1]
        for i in range(n_r - 2, -1, -1):
            phi_hat[i] = d_prime[i] - c_prime[i] * phi_hat[i + 1]
        return dst(phi_hat, type=1, axis=1, norm='ortho')

    def solve(self, wall:np.ndarray, rho:np.ndarray = None) -> FieldGrid:
        '''
        Potential for the given potential at the electrode radius (shape (n_z+2,), values at self.z)
        and charge density rho [C/m^3] (shape (n_r+1, n_z+2), values at grid nodes, optional).
        '''
        rhs = np.zeros((self.n_r, self.n_z))
        if rho is not None:
            rhs -= rho[:-1, 1:-1] / VACUUM_PERMITTIVITY
        # known potential at the electrode radius moves to the right hand side
        rhs[-1] -= self._upper[-1] * wall[1:-1]
        phi = np.zeros((self.n_r + 1, self.n_z + 2))
        phi[:-1, 1:-1] = self._solve(rhs)
        phi[-1] = wall
        phi[:, 0] = wall[0]
        phi[:, -1] = wall[-1]
        return FieldGrid(self.r, self.z, phi)

    def laplace(self, voltages:Mapping[str,float]) -> FieldGrid:
        '''
        Potential created by electrodes set to the voltages (electrodes missing in the mapping are grounded).
        '''
        return self.solve(self.geometry.wall_potential(self.z, voltages))

    def poisson(self, rho:np.ndarray) -> FieldGrid:
        '''
        Potential of the charge density rho [C/m^3] given at grid nodes with all electrodes grounded.
        '''
        return self.solve(np.zeros(self.n_z + 2), rho)

    def node_volumes(self) -> np.ndarray:
        '''
        Volume [m^3] associated with each node of the grid (used for depositing charge).
        '''
        if self._node_volumes is None:
            volumes = 2 * np.pi * self.r * self.dr
            volumes[0] = np.pi * (self.dr / 2)**2
            volumes[-1] = 2 * np.pi * self.r[-1] * self.dr / 2
            self._node_volumes = np.repeat(volumes[:, None], self.n_z + 2, axis=1) * self.dz
        return self._node_volumes

    def deposit(self, r:np.ndarray, z:np.ndarray, charge:np.ndarray) -> np.ndarray:
        '''
        Charge density [C/m^3] on the grid nodes from point charges [C] at (r, z), cloud-in-cell weighting.
        '''
        fr = np.abs(r) / self.dr
        fz = (z - self.z[0]) / self.dz
        inside = (fr < self.n_r) & (fz >= 0) & (fz < self.n_z + 1)
        i = fr[inside].astype(np.int64)
        j = fz[inside].astype(np.int64)
        ar = fr[inside] - i
        az = fz[inside] - j
        q = charge[inside]
        shape = (self.n_r + 1, self.n_z + 2)
        index = np.concatenate([np.ravel_multi_index((i + di, j + dj), shape) for di in (0, 1) for dj in (0, 1)])
        weights = np.concatenate([q * wi * wj for wi in (1 - ar, ar) for wj in (1 - az, az)])
        density = np.bincount(index, weights=weights, minlength=shape[0] * shape[1]).reshape(shape)
        return density / self.node_volumes()


def validate_against_matrix(solver:AxisymmetricSolver, center_position_mm:float) -> dict[str,dict[str,float]]:
    '''
    Compare the on-axis potential of each electrode (set to 1 V) with the potential matrix used by TTrap.
    Returns maximum absolute difference, difference of the peak values and of the integrals for each electrode.
    '''
    from trap import TTrap
    trap = TTrap(position=-center_position_mm)
    z_matrix = (trap.position + trap.dx * np.arange(trap.segments)) / 1000
    report = {}
    for name in solver.geometry.names:
        if name in solver.geometry.grounded:
            continue
        trap.SetEverythingToZero()
        trap.SetElectrodeV(name, 1.0)
        matrix = trap.get_final_V()
        fd = np.interp(z_matrix, solver.z, solver.laplace({name: 1.0}).on_axis())
        report[name] = {"max_abs_diff": float(np.max(np.abs(fd - matrix))),
                        "peak_diff": float(fd.max() - matrix.max()),
                        "integral_rel_diff": float(fd.sum() / matrix.sum() - 1) if matrix.sum() > 0 else float("nan"),
                        "peak_shift_mm": float((z_matrix[np.argmax(fd)] - z_matrix[np.argmax(matrix)]) * 1000)}
    return report
