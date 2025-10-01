
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