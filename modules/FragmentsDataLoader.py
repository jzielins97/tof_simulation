import os
from typing import Optional
import polars as pl
import numpy as np
from constants import (ATOMIC_UNIT_TO_KEV, PROTON_MASS_KEV, NEUTRON_MASS_KEV, ELECTRON_MASS_KEV,
                       SPEED_OF_LIGHT, BOLTZMAN_CONSTANT_EV)

# constants
# location of the parquet files with Geant4 fragments (can be changed with FRAGMENTS_DATA_PATH environment variable)
DATA_PATH = os.environ.get("FRAGMENTS_DATA_PATH", "C:\\Users\\jakub\\cernbox\\PBarInSphereSim\\nuclid_chart\\parquet")
NUCLEON_MASS = (PROTON_MASS_KEV + NEUTRON_MASS_KEV)/2 # keV/c^2

def build_data_path(filename:str)->str:
    return os.path.join(DATA_PATH,filename)

def build_path(*folders):
    return os.path.join(*folders)

def create_lf_with_weights(isotope:str,
                           half_life:float = 10 * 60,
                           V:float=10,
                           R:float=0.015,
                           B:float=5,
                           r0:float=0.0,
                           max_events:Optional[int]=1_000_000)->pl.LazyFrame:
    '''
    Load fragments from antiproton annihilation on the isotope and calculate the probability of trapping each fragment.
    The fragments (fully stripped, charge Z) are emitted isotropically at radius r0 [m] inside a Penning trap
    with wall potential V [kV], electrode radius R [m] and magnetic field B [T].
    A fragment with kinetic energy E [keV] emitted at angle theta to the magnetic field is trapped when:
     - axially: E cos^2(theta) <= Z V
     - radially: the cyclotron orbit starting at r0 stays inside the electrodes, r0 + 2 r_c <= R,
       with r_c = sqrt(2 m E) sin(theta) / (Z B)
    For isotropic emission cos(theta) is uniformly distributed, so the trappable fraction is cos(theta_1) - cos(theta_2).
    Only events with id < max_events are used (None uses all events) and total_events is limited accordingly.
    '''
    if r0 >= R:
        raise ValueError(f"Annihilation radius r0={r0} m must be smaller than electrode radius R={R} m")
    # maximum cyclotron radius of a trapped fragment
    rc_max = (R - r0) / 2

    # load the data
    events_counts = pl.scan_parquet(build_data_path("events_counts.parquet"))
    fragments = pl.scan_parquet(build_data_path(f"{isotope}_fragments.parquet"))
    if max_events is not None:
        # limit number of annihilation events (not number of fragments) used for the yields
        fragments = fragments.filter(pl.col("event") < max_events)
        events_counts = events_counts.with_columns(pl.min_horizontal(pl.col("total_events").cast(pl.Int64),pl.lit(max_events)).alias("total_events"))

    fragments = (fragments
                 .filter((pl.col("fragmentA")!=pl.col("targetA")) | (pl.col("fragmentZ")!=pl.col("targetZ")),
                         pl.col("fragmentZ")>0)
                 .join(events_counts,on=["targetA","targetZ"])
                 .with_columns(fragmentN=pl.col("fragmentA")-pl.col("fragmentZ"))
                 .select(["event","targetA","targetZ","fragmentA","fragmentZ","fragmentN","energy_keV","total_events"])
                 .cast({"fragmentN":pl.Int64,"fragmentZ":pl.Int64}))

    isotope_info = (pl.scan_parquet(build_data_path("isotopes_info.parquet"))
                    .filter(pl.col("Z")>0)
                    .rename({"N":"fragmentN","Z":"fragmentZ"})
                    # stable isotopes don't have half-life in the IAEA table
                    .with_columns(pl.col("half_life_s").fill_null(np.inf),
                                  # if the mass is missing use the mass number times the atomic mass unit
                                  pl.col("atomic_mass").fill_null((pl.col("fragmentZ")+pl.col("fragmentN")).cast(pl.Float64)))
                    .with_columns(mass_keV=pl.col("atomic_mass").mul(ATOMIC_UNIT_TO_KEV))
                    # energy above which none of the emission angles is trappable
                    .with_columns(energy_max=(pl.col("fragmentZ") * V / (rc_max*B*SPEED_OF_LIGHT / 1e3 * (pl.col("fragmentZ") / 2 / pl.col("mass_keV") / V).sqrt()).arctan().cos()**2).ceil()))

    data = fragments.join(isotope_info,on=["fragmentN","fragmentZ"]).filter(pl.col("half_life_s")>half_life)

    # calculate weights for the energy
    data = (data.with_columns(cos_theta_1=(V * pl.col("fragmentZ") / pl.col("energy_keV")).sqrt().clip(upper_bound=1.0),
                              sin_theta_2=(rc_max * SPEED_OF_LIGHT / 1e3 * B * pl.col("fragmentZ") / (2 * pl.col("mass_keV") * pl.col("energy_keV")).sqrt()).clip(upper_bound=1.0))
            .with_columns(theta_1=pl.col("cos_theta_1").arccos(),
                          theta_2=pl.col("sin_theta_2").arcsin(),
                          trappable_fraction=(pl.col("cos_theta_1") - (1 - pl.col("sin_theta_2")**2).sqrt()).clip(0.0,1.0))
            )
    return data
