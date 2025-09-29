import os
import polars as pl
import numpy as np

# constants
DATA_PATH = "c:\\Users\\jzielins\\Documents\\Python Scripts\\FragmentsHCI\\data"
ATOMIC_UNIT_TO_KEV = 931483.6148 # keV / au
PROTON_MASS_KEV = 938272.08816 # keV/c^2
NEUTRON_MASS_KEV = 939565.42052 # keV/c^2
NUCLEON_MASS = (PROTON_MASS_KEV + NEUTRON_MASS_KEV)/2 # keV/c^2
SPEED_OF_LIGHT = 299792458 # m/s
BOLTZMAN_CONSTANT_EV = 8.617333262e-5 # eV/K

def build_data_path(filename:str)->str:
    return os.path.join(DATA_PATH,filename)

def build_path(*folders):
    return os.path.join(*folders)

def create_lf_with_weights(isotope:str,half_life:float = 10 * 60,V:float=10,R:float=0.015,B:float=5)->pl.LazyFrame:
    # load the data
    events_counts = pl.scan_parquet(build_data_path("events_counts.parquet"))

    fragments = (pl.scan_parquet(build_data_path(f"{isotope}_fragments.parquet")).head(1_000_000)
                 .filter((pl.col("fragmentA")!=pl.col("targetA")) | (pl.col("fragmentZ")!=pl.col("targetZ")),
                         pl.col("fragmentZ")>0)
                 .join(events_counts,on=["targetA","targetZ"])
                 .with_columns(fragmentN=pl.col("fragmentA")-pl.col("fragmentZ"))
                 .select(["event","targetA","targetZ","fragmentA","fragmentZ","fragmentN","energy_keV","total_events"])
                 .cast({"fragmentN":pl.Int64,"fragmentZ":pl.Int64}))
    
    isotope_info = (pl.scan_parquet(build_data_path("isotopes_info.parquet"))
                    .filter(pl.col("Z")>0)
                    .rename({"N":"fragmentN","Z":"fragmentZ"})
                    .with_columns(pl.col("half_life_s").fill_null(np.inf), # 10*pl.col("half_life_s").max()
                                  mass_keV=pl.col("atomic_mass").mul(ATOMIC_UNIT_TO_KEV))
                    .with_columns(pl.col("atomic_mass").fill_null(pl.col("fragmentZ")+pl.col("fragmentN")),
                                  pl.col("mass_keV").fill_null((pl.col("fragmentZ")+pl.col("fragmentN"))*NUCLEON_MASS),
                                  energy_max=np.ceil(pl.col("fragmentZ") * V / np.cos(np.arctan(R*B*SPEED_OF_LIGHT / 1e3 * np.sqrt( pl.col("fragmentZ") / 2 / pl.col("mass_keV") / V)))**2)
                                  ))
    
    data = fragments.join(isotope_info,on=["fragmentN","fragmentZ"]).filter(pl.col("half_life_s")>half_life)
    
    # calculate weights for the energy
    data = (data.with_columns(theta_1=(np.arccos(np.sqrt(V * pl.col("fragmentZ") / pl.col("energy_keV")))).fill_nan(0),
                              theta_2=(np.arcsin( R * SPEED_OF_LIGHT / 1e3 * B * pl.col("fragmentZ") / np.sqrt(2 * pl.col("mass_keV") * pl.col("energy_keV")))).fill_nan(np.pi/2))
            .with_columns(trappable_fraction=(pl.col("theta_2") - pl.col("theta_1")).clip(0)/np.pi*2)
            )
    return data