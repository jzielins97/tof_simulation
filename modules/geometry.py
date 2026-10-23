'''
Geometry of the cylindrical electrode stack used by the field solver.
All positions are in metres along the trap axis, with z = 0 in the middle of the trap floor.
'''
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np

from trap import TTrap


@dataclass(frozen=True)
class Electrode:
    name:str
    z_start:float # m
    z_end:float # m


@dataclass(frozen=True)
class TrapGeometry:
    electrodes:tuple[Electrode,...] # sorted along z
    radius:float # inner radius of all electrodes [m]
    grounded:tuple[str,...] = () # electrodes which are always grounded

    @classmethod
    def from_ttrap(cls,
                   center_position_mm:float,
                   radius_mm:float = 15.0,
                   grounded:Sequence[str] = tuple(TTrap.GROUNDED_ELECTRODES)) -> "TrapGeometry":
        '''
        Electrodes of the AEgIS 5 T trap (TTrap) with z = 0 at center_position_mm (TTrap coordinates).
        '''
        trap = TTrap()
        electrodes = tuple(Electrode(e.GetName(),
                                     (e.GetElectrodeStart() - center_position_mm) / 1000,
                                     (e.GetElectrodeEnd() - center_position_mm) / 1000)
                           for e in sorted(trap.electrodes, key=lambda e: e.GetElectrodeStart()))
        return cls(electrodes=electrodes, radius=radius_mm / 1000, grounded=tuple(grounded))

    @property
    def names(self) -> list[str]:
        return [e.name for e in self.electrodes]

    @property
    def z_min(self) -> float:
        return self.electrodes[0].z_start

    @property
    def z_max(self) -> float:
        return self.electrodes[-1].z_end

    def wall_potential(self, z:np.ndarray, voltages:Mapping[str,float]) -> np.ndarray:
        '''
        Potential at the electrode radius as a function of z.
        In the gaps between electrodes the potential changes linearly, outside of the stack it is grounded.
        '''
        z = np.asarray(z, dtype=float)
        V = np.array([0.0 if e.name in self.grounded else float(voltages.get(e.name, 0.0)) for e in self.electrodes])
        # potential is defined on the edges of the electrodes and interpolated linearly in between
        z_nodes = np.ravel([[e.z_start, e.z_end] for e in self.electrodes])
        V_nodes = np.repeat(V, 2)
        wall = np.interp(z, z_nodes, V_nodes, left=0.0, right=0.0)
        return wall
