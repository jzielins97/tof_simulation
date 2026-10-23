
# Documentation
[`docs/tof_simulation.md`](docs/tof_simulation.md) explains the physics of the simulation (fragment trapping, potentials, initial conditions, extraction and the TOF spectrum) and how to use it (commands, options, outputs, choice of parameters).

# Local environment (miniforge)
The pinned dependencies are listed in `requirements.txt`. To recreate the environment:
```sh
conda create -n tof_simulation -c conda-forge python=3.12 --file requirements.txt
conda activate tof_simulation
```
The same file works with pip (`pip install -r requirements.txt`) in any Python 3.12 environment.
Run the tests with `python -m pytest` from the repository root.

# Model conventions
- **Trapping of fragments** (`modules/FragmentsDataLoader.py`): fragments (fully stripped, charge Z) are emitted isotropically at radius `r0` inside a trap with wall potential `V` [kV], electrode radius `R` [m] and field `B` [T]. A fragment with energy E at angle θ to the field is trapped when E cos²θ ≤ ZV (axial) and r0 + 2 r_c ≤ R (radial, r_c = p sinθ/(ZB)). For isotropic emission the trappable fraction is cos θ1 − cos θ2. The yields use the first `max_events` (default 1,000,000) annihilation events. The data folder can be changed with the `FRAGMENTS_DATA_PATH` environment variable.
- **Time of flight** (`calculate_tof.py`): between points of the potential the force is assumed constant (exact time 2Δz/(v_a+v_b)). After the end of the potential the particle drifts with its final velocity to the MCP. Particles which cannot leave the trap are reported as trapped and are not counted.
- **Signal**: fraction of the simulated particles per time bin, averaged over all iterations (empty bins count as zero). The time of a bin is its centre.

# Tracking engine (`--engine tracking`)
The default `legacy` engine is the 1D model described above. The tracking engine replaces it with:
- **Field solver** (`modules/field_solver.py`, `modules/geometry.py`): five-point finite differences of the Laplace/Poisson equation on an (r, z) grid (default 0.25 mm). For cylindrical electrodes this is the full 3D potential. The electrode positions come from `TTrap`, the radius is 15 mm, the potential is linear in the gaps, and the ends of the tube are grounded. The system is solved directly (sine transform along z, tridiagonal solves along r) in about 10 ms. On axis it agrees with `ElectrodesPotentialMap.txt`: the C trap matches to <0.1 %. The P-trap rows of the map are shifted by about +0.95 mm relative to the `TTrap` electrode positions, and the HV electrodes differ in shape.
- **Initial conditions** (`modules/plasma.py`, `--model`):
  - `cloud`: non-interacting particles in thermal equilibrium (`--temperature_K`, `--T_perp_K`). Along the field lines they follow the Boltzmann distribution in the trap potential, the radial profile is Gaussian (`--cloud_radius_mm`), and velocities are Maxwellian.
  - `plasma`: global thermal equilibrium of a non-neutral plasma with all species. The self-consistent space charge is found by solving the Poisson–Boltzmann equation with Newton iterations. The rotation frequency is chosen to give `--plasma_radius_mm` (sqrt(2⟨r²⟩), i.e. the edge of a uniform column). `--n_charges` elementary charges are shared between species according to their weights. A plasma that does not fit into the well is rejected.
  - `legacy`: the generalised normal z distribution and uniform energies of the 1D model. Half of the particles start moving backwards unless `--forward_only True`.
  - Only particles with axial energy below the barrier are kept.
- **Tracking** (`modules/tracker.py`, `--tracker`):
  - `gc` (default): guiding centres move along the field lines with force qE_z − μ dB/dz (leapfrog, `--dt_s`). After the end of the electric field they drift to the MCP. In a non-uniform field (`--B_profile` csv with columns `z_m,Bz_T`) the magnetic moment is conserved.
  - `boris`: full 3D motion, for validation.
  - The right wall can be pulsed with a finite rise time (`--pulse_rise_time_s`).
  - `--space_charge dynamic` recomputes the space charge field from the particles every `--poisson_every` steps; `initial` uses space charge only for the initial equilibrium.
- **Charge states**: needed for thermal velocities and space charge. Use `tof --charge`. `tof_fragments` reads a `charge` or `fragmentZ` column.
- **Output**: same columns as the legacy engine. Plot bit 0x20 shows the potentials and the initial conditions. Save bits: 0x02 on-axis potentials, 0x04 initial conditions, 0x08 arrival times.

Example (He²⁺ released from a plasma with 10⁶ charges, 1 mm radius at 300 K):
```sh
python calculate_tof.py --trap_floor_V 180 --trap_wall_V 190 --engine tracking --model plasma --n_charges 1e6 --plasma_radius_mm 1 --temperature_K 300 --plot_mask 0x30 tof --mq 2.0013 --charge 2
```
Typical run times (25000 particles per iteration, one species): cloud ~1 s, plasma with dynamic space charge ~12 s. With 10 species: ~30 s and ~70 s.

# Setup on lxplus
This program requires newer version of polars, which isn't available on lxplus as of 01/10/2025.
This means that the package must be installed locally and then used by the batch system.
The instruction on how to install custom libraries is provided in (brachdocs)[https://batchdocs.web.cern.ch/specialpayload/python.html]
In short, one needs to do:
1. source LCG with python:
```sh
$ . /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-centos9-gcc15-opt/setup.sh
```
2. Install in your eos directory polars:
```sh
$ PYTHONUSERBASE=/eos/user/j/jzielins/.local/ pip3 install --user polars==1.32.00
```
## Update files with your path
There are two scripts that use this location. One is the bin/setup.sh script that can be used to load environment to test the program. The other is SubmitTof.sub script for submitting scripts on HTCondor service. In both cases you only need to update the path to your user directory on eos.

usage: calculate_tof.py [-h] --trap_floor_V TRAP_FLOOR_V --trap_wall_V TRAP_WALL_V [--pulse_wall_to_V PULSE_WALL_TO_V] [--distance_to_mcp_m DISTANCE_TO_MCP_M] [--spacecharge_min_V SPACECHARGE_MIN_V]
                        [--spacecharge_max_V SPACECHARGE_MAX_V] [--trap_left_wall TRAP_LEFT_WALL [TRAP_LEFT_WALL ...]] [--trap_floor TRAP_FLOOR [TRAP_FLOOR ...]]
                        [--trap_right_wall TRAP_RIGHT_WALL [TRAP_RIGHT_WALL ...]] [--thermalised THERMALISED] [--N_particles N_PARTICLES] [--iterations ITERATIONS] [--tof_range TOF_RANGE TOF_RANGE]
                        [--tof_dt TOF_DT] [--verbose_level VERBOSE_LEVEL] [--savedata_mask SAVEDATA_MASK] [--plot_mask PLOT_MASK] [--dpi DPI] [--savefig SAVEFIG] [--showfig SHOWFIG]
                        [--file_prefix FILE_PREFIX] [--fig_format FIG_FORMAT]
                        {tof,tof_pbar,tof_fragments} ...

This program simulates the time of flight of particles with specified m/q ratio released after being trapped inside a Penning trap. The model uses multiple simplification assumptions, such as the 1-D nature, particle distribution independent from of particle type of plasma temperature, no magnetic field effects.

positional arguments:
  {tof,tof_pbar,tof_fragments}

optional arguments:
  -h, --help            show this help message and exit
  --trap_floor_V TRAP_FLOOR_V
                        Potential set on electrodes used as floor of the trap.
  --trap_wall_V TRAP_WALL_V
                        Potential set on electrodes used as right and left wall of the trap (note: for left wall all electrodes left of the floor electrodes are set)
  --pulse_wall_to_V PULSE_WALL_TO_V
                        Expected voltage on the right wall electrodes during the launch of particles from the trap
  --distance_to_mcp_m DISTANCE_TO_MCP_M
                        Distance between the middle electrode used for the floor of the trap and the MCP used for measuring the tof spectra
  --spacecharge_min_V SPACECHARGE_MIN_V
                        Minimum space charge felt by the particles
  --spacecharge_max_V SPACECHARGE_MAX_V
                        Maximum space charge felt by the particles
  --trap_left_wall TRAP_LEFT_WALL [TRAP_LEFT_WALL ...]
                        Electrodes used as left wall of the trap
  --trap_floor TRAP_FLOOR [TRAP_FLOOR ...]
                        Electrodes used as floor of the trap
  --trap_right_wall TRAP_RIGHT_WALL [TRAP_RIGHT_WALL ...]
                        Electrodes used as right wall of the trap
  --thermalised THERMALISED
                        Whether to use z distribution as in thermalised case (flatten gaussian) or non-thermalised case (pure gaussian)
  --N_particles N_PARTICLES
                        Number of particles simulated inside the trap for each iteration
  --iterations ITERATIONS
                        Number of iterations of simulated N_particles used for calculating average tof spectra
  --tof_range TOF_RANGE TOF_RANGE
                        Expected range of the tof signal. It is used for binning the simulated tofs of random particles
  --tof_dt TOF_DT       Bin size for calculating final tof spectra (note: the number of bins created by the program can be estimated as (tof_range[1]-tof_range[0])/tof_dt)
  --verbose_level VERBOSE_LEVEL
                        Control the amount of prints from the program. Available levels:0 - no prints; 1 -> print only the tof peaks dataframe at the end; 2 -> print information about current m/q and the ToF
                        signals DataFrame; 3 -> print DataFrames of all calculation steps)
  --savedata_mask SAVEDATA_MASK
                        Bit mask in hex representation for controlling amount of dataframes saved (note: location of data will be <script_path>/data/):0x00 - don't save any DataFrames; 0x01 - save final tof
                        and fitted peak position; 0x02 - save potential dataframe;0x04 - save velocities; 0x08 - save times per particle. Combine masks to save multiple dataframes (0x05 will use 0x01 and
                        0x04 masks)
  --plot_mask PLOT_MASK
                        Bit mask in hex representation for controlling the amount of plots created by the program. Available levels: 0x00 - no plots; 0x10 - plot final tof signal; 0x01 - plot potential
                        shape; 0x02 - plot initial conditions; 0x04 - plot velocities and times; 0x08 - plot fitted peak. Combine masks to plot multiple plots (0x13 will plot final tof, potential shape, and
                        initial conditions)
  --dpi DPI             Dpi of plotted figures (note: all figures are saved with dpi of 600)
  --savefig SAVEFIG     Wheter created figures should be saved (note: location of the plots is <script_path>/plots/)
  --showfig SHOWFIG     Show plots at the end. When this flag is False, figures aren't stored in the memory
  --file_prefix FILE_PREFIX
                        Prefix added to file name of the plots and data. File names indicate information about parameters used in the simulation <file name>. If the prefix is added, the file name is updated
                        to <file_prefix>_<file name>
  --fig_format FIG_FORMAT
                        Format of saved figures