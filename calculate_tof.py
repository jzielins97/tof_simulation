# handling of arguments
import argparse
import sys
import os
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__),"modules"))
# print(sys.path)
from typing import Union,Optional 
# custom modules for fragments analysis 
from collections.abc import Callable
import FragmentsDataLoader as loader
from trap import TTrap
# plotting
import seaborn as sns
import matplotlib.pyplot as plt
# data analysis
import polars as pl
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from scipy.optimize import curve_fit
from scipy.stats import gennorm

def _bool(string:str)->bool:
    '''
    Convert string to bool, where "true" or "1" returns True, and anything else gives False. Comparisson is not case sensitive.
    '''
    return string.lower() in ['true','1']

parser = argparse.ArgumentParser(description="This program simulates the time of flight of particles with specified m/q ratio trapped inside a Penning trap. The model uses multiple simplification assumptions, such as the 1-D nature, paerticle distribution independent from of particle type of plasma temperature, no magnetic field effects.")
subparser = parser.add_subparsers(dest='command',required=True)
# common parameters --------------------------->
parser.add_argument('--trap_floor_V',type=float,required=True,
                    help="Potential set on electrodes used as floor of the trap.")
parser.add_argument('--trap_wall_V',type=float,required=True,
                    help="Potential set on electrodes used as right and left wall of the trap (note: for left wall all electrodes left of the floor electrodes are set)")
parser.add_argument('--pulse_wall_to_V',type=float,default=0,
                    help="Expected voltage on the right wall electrodes during the launch of particles from the trap")
parser.add_argument('--distance_to_mcp_m',type=float,default=1.05,
                    help="Distance between the middle electrode used for the floor of the trap and the MCP used for measuring the tof spectra")
parser.add_argument('--spacecharge_min_V',type=float,default=None,
                    help="Minimum space charge felt by the particles")
parser.add_argument('--spacecharge_max_V',type=float,default=None,
                    help="Maximum space charge felt by the particles")
parser.add_argument('--trap_left_wall',nargs='+',type=str,default=['P7','P8','P9'],
                    help="Electrodes used as left wall of the trap")
parser.add_argument('--trap_floor',nargs='+',type=str,default=['P10','P11','P12'],
                    help="Electrodes used as floor of the trap")
parser.add_argument('--trap_right_wall',nargs='+',type=str,default=['P13'],
                    help="Electrodes used as right wall of the trap")
parser.add_argument('--thermalised',default=True,
                    help="Whether to use z distribution as in thermalised case (flatten gaussian) or non-thermalised case (pure gaussian)")
parser.add_argument('--N_particles',type=int,default=10_000,
                    help="Number of particles simulated inside the trap for each iteration")
parser.add_argument('--iterations',type=int,default=10,
                    help="Number of iterations of simulated N_particles used for calculating average tof spectra")
parser.add_argument('--tof_range',nargs=2,type=float,default=[0,1e-4],
                    help="Expected range of the tof signal. It is used for binning the simulated tofs of random particles")
parser.add_argument('--tof_dt',type=float,default=1e-9,
                    help="Bin size for calculating final tof spectra (note: the number of bins created by the program can be estimated as (tof_range[1]-tof_range[0])/tof_dt)")
parser.add_argument('--verbose_level',type=int,default=0,
                    help="Control the amount of prints from the program. Available levels:0 - no prints; 1 -> print only the tof peaks dataframe at the end; 2 -> print information about current m/q and the ToF signals DataFrame; 3 -> print DataFrames of all calculation steps)")
parser.add_argument('--savedata_mask',default=0,
                    help="Bit mask in hex representation for controlling amount of dataframes saved (note: location of data will be <script_path>/data/):0x00 - don't save any DataFrames; 0x01 - save final tof and fitted peak position; 0x02 - save potential dataframe;0x04 - save velocities; 0x08 - save times per particle. Combine masks to save multiple dataframes (0x05 will use 0x01 and 0x04 masks)")
parser.add_argument('--plot_mask',default=0,
                    help="Bit mask in hex representation for controlling the amount of plots created by the program. Available levels: 0x00 - no plots; 0x10 - plot final tof signal; 0x01 - plot potential shape; 0x02 - plot initial conditions; 0x04 - plot velocities and times; 0x08 - plot fitted peak. Combine masks to plot multiple plots (0x13 will plot final tof, potential shape, and initial conditions)")
parser.add_argument('--dpi',type=int,default=200,
                    help="Dpi of plotted figures (note: all figures are saved with dpi of 600)")
parser.add_argument('--savefig',default=False,
                    help="Wheter created figures should be saved (note: location of the plots is <script_path>/plots/)")
parser.add_argument('--showfig',default=True,
                    help="Show plots at the end. When this flag is False, figures aren't stored in the memory")
parser.add_argument('--file_prefix',type=str,default='',
                    help="Prefix added to file name of the plots and data. File names indicate information about parameters used in the simulation <file name>. If the prefix is added, the file name is updated to <file_prefix>_<file name>")
parser.add_argument('--fig_format',type=str,default='eps',
                    help="Format of saved figures")
# genreal tof --------------------------------->
tof_parser = subparser.add_parser("tof")
tof_parser.add_argument('--mq', type=float, required=True,
                        help="m/q ratio of the particle in the trap [in a.u./e]")
tof_parser.add_argument('--label',type=str,default=None,
                        help="Label for the particle in the dataframe")
tof_parser.add_argument('--weight',type=float,default=None,
                        help="How much the signal of this particle contributes to total ToF signal")
# pbar tof ------------------------------------>
pbar_parser = subparser.add_parser("tof_pbar")
pbar_parser.add_argument('--weight',type=float,default=1,
                         help="How much the signal of this particle contributes to total ToF signal")
# fragments tof ------------------------------->
fragments_parser = subparser.add_parser("tof_fragments")
fragments_parser.add_argument('--fragments_data_path',type=str,required=True,
                              help="Path to the parquet file with yields of fragments from Geant4 simulations")

args = sys.argv[1:]
try:
    dash_index = args.index("--")
    args, cmd_args = args[:dash_index], args[dash_index+1:]
except ValueError:
    args, cmd_args = args, []
args = parser.parse_args(args)
if isinstance(args.thermalised,str):
    args.thermalised = _bool(args.thermalised)
if isinstance(args.plot_mask,str):
    args.plot_mask = int(args.plot_mask,base=16)
if isinstance(args.savefig,str):
    args.savefig = _bool(args.savefig)
if isinstance(args.showfig,str):
    args.showfig = _bool(args.showfig)
if isinstance(args.savedata_mask,str):
    args.savedata_mask = int(args.savedata_mask,base=16)

# for fitting the peak from tof and finding the maximum position
def _fit_function(x, a, b, c, d=0):
    return a * np.exp(-((x - b)/c)**2) + d


def _exp(x, a, b, c, d):
    return a*np.exp(c*(x-b)) + d


def _lin(x, a, b):
    return a*x + b


def _maxwell_pdf(v,m,T):
    a = np.sqrt(loader.BOLTZMAN_CONSTANT_EV * T / m)
    return np.sqrt(2/np.pi)*v**2/a**3*np.exp(-v**2/(2*a**2))


def _build_filename(prefix:str,
                    title:str,
                    suffix:str,
                    extension:str)->str:
    '''
    Helper function for creating filename by combining prefix, title, suffix, and extension
    '''
    filename = title
    if prefix != '':
        filename = prefix + "_" + filename
    if suffix != '':
        filename = filename + "_" + suffix
    return filename + f".{extension}"


def _save_fig(fig:plt.Figure,
              title:str,
              file_prefix:str = '',
              file_suffix:str = '',
              fig_format:str = 'eps')->None:
    '''
    Function for building the path to the plot and saving image
    '''
    filename = _build_filename(file_prefix,title,file_suffix,fig_format)
    fig.savefig(os.path.join(os.path.dirname(__file__),'plots',filename),dpi=600)


def _save_data(data:pl.DataFrame,
               title:str,
               file_prefix:str,
               file_suffix:str)->None:
    '''
    Helper function for consistant saving of data
    '''
    filename = filename = _build_filename(file_prefix,title,file_suffix,"parquet")
    data.write_parquet(os.path.join(os.path.dirname(__file__),'data',filename))


def _calculate_potential_shape(trap_floor_V:float = 150,
                              trap_wall_V:float = 160,
                              pulse_wall_to_V:float = 0,
                              trap_left_wall:Union[str,list[str]] = ['P7','P8','P9'],
                              trap_floor:Union[str,list[str]] = ['P10','P11','P12'],
                              trap_right_wall:Union[str,list[str]] = ['P13'],
                              verbose_level:int=0
                              )->pl.DataFrame:
    '''
    Calculate the shape of potentials for trapping and releasing particles.
    '''
    # format parameters
    if isinstance(trap_left_wall,str):
        trap_left_wall=[trap_left_wall]
    if isinstance(trap_floor,str):
        trap_floor=[trap_floor]
    if isinstance(trap_right_wall,str):
        trap_right_wall=[trap_right_wall]

    # start by creating the potentials
    # create the trap object to find the middle position of the trap well
    aegis_trap = TTrap()
    center_position = 0
    if len(trap_floor) % 2 == 0:
        center_position = (aegis_trap._GetElectrode(trap_floor[int(len(trap_floor)/2)-1]).GetElectrodeEnd() + aegis_trap._GetElectrode(trap_floor[int(len(trap_floor)/2)]).GetElectrodeStart())/2
    else:
        center_position = aegis_trap._GetElectrode(trap_floor[int(len(trap_floor)/2)]).GetElectrodeCenter()
    # update trap object centered at the trap well
    aegis_trap = TTrap(position=-center_position)
    if verbose_level > 2:
        aegis_trap.Print()
    
    electrodes = aegis_trap.GetElectrodeNames()
    # reset the potentials in the trap
    aegis_trap.SetEverythingToZero()
    # set potentials for the trapping well
    for electrode in electrodes:
        if electrode in trap_floor:
            aegis_trap.SetElectrodeV(electrode,trap_floor_V)
        else:
            aegis_trap.SetElectrodeV(electrode,trap_wall_V)
        if electrode == trap_right_wall[-1]:
            break
    trap_potential_prepare = aegis_trap.get_final_V()

    # set potentials after pulsing right wall
    for electrode in trap_right_wall:
        aegis_trap.SetElectrodeV(electrode,pulse_wall_to_V)

    trap_potential_launch = aegis_trap.get_final_V()

    potential_df = pl.DataFrame({'z [mm]':np.array([aegis_trap.position + i*aegis_trap.dx for i in range(aegis_trap.segments)]),
                                 'V_prepare':trap_potential_prepare,
                                 'V_launch':trap_potential_launch})
    
    # find the parameters (max, min, z_start, z_end, etc) of the trapping potential
    # the trapping is done between the first electrode of the left wall and the last electrode of the right wall
    # the maximum trapping potential is defined by the right wall
    trap_max = (potential_df.filter(pl.col('z [mm]').is_between(0,aegis_trap._GetElectrode(trap_right_wall[-1]).GetElectrodeEnd()))
                .filter(pl.col("V_prepare") == pl.col("V_prepare").max()))
    # select potential that is the actual trap
    potential_df = (potential_df.with_columns(((pl.col('z [mm]').is_between(aegis_trap._GetElectrode(trap_left_wall[0]).GetElectrodeStart(),trap_max['z [mm]'])
                                                &(pl.col('V_prepare') <= trap_max['V_prepare'])).alias("full_trap"))))
    
    # add a flag for limiting plotting
    potential_df = potential_df.with_columns(interest_region = (pl.col('z [mm]') > aegis_trap._GetElectrode(trap_left_wall[0]).GetElectrodeStart()) & (pl.col("V_launch") > 0))
    return potential_df


def _fill_trap(potential_df:pl.DataFrame,
              spacecharge_min_V:Optional[float] = None,
              spacecharge_max_V:Optional[float] = None,
              verbose_level:int = 0
              )->tuple[pl.DataFrame,float,float]:
    '''
    Find region of the trap which is filled with particles
    '''
    V_min = potential_df.filter("full_trap").select(pl.min("V_prepare")).item()
    V_max = potential_df.filter("full_trap").select(pl.max("V_prepare")).item()
    
    # we fill the trap
    if spacecharge_min_V is not None:
        # fill as static value
        spacecharge_min_V = V_min + spacecharge_min_V
        if spacecharge_min_V > V_max:
            spacecharge_min_V = V_max
    else:
        # fill as a fraction of the full depth of the well
        spacecharge_min_V = V_min

    # we fill the trap
    if spacecharge_max_V is not None:
        # fill as static value
        spacecharge_max_V = V_min + spacecharge_max_V
        if spacecharge_max_V > V_max:
            spacecharge_max_V = V_max
    else:
        # fill as a fraction of the full depth of the well
        spacecharge_max_V = V_max

    # find region of the trap filled with particles
    potential_df = potential_df.with_columns((pl.col("full_trap") & (pl.col("V_prepare") <= spacecharge_max_V)).alias("filled_trap"))
    potential_df = potential_df.with_columns(pl.when("filled_trap",pl.col("V_prepare") <= spacecharge_min_V).
                                             then(pl.lit(spacecharge_min_V))
                                             .when("filled_trap")
                                             .then(pl.col("V_prepare"))
                                             .alias("V_prepare_filled_min"))
    
    potential_df = potential_df.with_columns((spacecharge_max_V - pl.col("V_prepare_filled_min")).alias("dV_filled"))

    # calculate available potentials during launch
    potential_df = potential_df.with_columns(pl.when("filled_trap",pl.col("V_prepare") <= spacecharge_min_V)
                                             .then(pl.col("V_launch") + spacecharge_min_V - pl.col('V_prepare'))
                                             .when("filled_trap")
                                             .then(pl.col("V_launch"))
                                             .alias("V_launch_filled_min"))
    if verbose_level > 1:
        print("Calculated potentials")
        print(potential_df)

    return potential_df,spacecharge_min_V,spacecharge_max_V


def _plot_potential_shape(potential_df:pl.DataFrame,
                         trap_floor_V:float,
                         trap_wall_V:float,
                         spacecharge_min_V:Optional[float],
                         spacecharge_max_V:Optional[None],
                         dpi:int = 200,
                         savefig:bool = True,
                         file_prefix:str = '',
                         file_suffix:str = '',
                         fig_format:str= 'eps'
                         )->tuple[plt.Figure,plt.Axes]:
    '''
    Plot potentials and trapping regions together with parameters of the trap
    '''
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(6,1,(1,4))
    # plot full potential
    ax.plot(potential_df.filter("interest_region")['z [mm]'],potential_df.filter("interest_region")['V_launch'],color='tab:orange',label='launch potential')
    ax.plot(potential_df.filter("interest_region")['z [mm]'],potential_df.filter("interest_region")['V_prepare'],color='tab:blue',label='full potential')

    interest_region_size = potential_df.select(pl.sum("interest_region")).item()

    # plot trap potential
    V_full_min = potential_df.filter('full_trap').filter(pl.col("V_prepare")==pl.min("V_prepare")).select('z [mm]',"V_prepare")
    V_full_max = potential_df.filter('full_trap').filter(pl.col("V_prepare")==pl.max("V_prepare")).select('z [mm]',"V_prepare")
    ax.axvline(potential_df.filter("full_trap").select(pl.first('z [mm]')).item(),color='green',linestyle='--')
    ax.axvline(potential_df.filter("full_trap").select(pl.last('z [mm]')).item(),color='green',linestyle='--')
    ax.plot(potential_df.filter("interest_region")['z [mm]'],[V_full_max['V_prepare']]*interest_region_size,color='green',linestyle='--',label='trap well')
    ax.plot(potential_df.filter("interest_region")['z [mm]'],[V_full_min['V_prepare']]*interest_region_size,color='green',linestyle='--')
    
    #plot filled trap
    V_filled_min = potential_df.filter('filled_trap').filter(pl.col("V_prepare")==pl.min("V_prepare")).select('z [mm]',"V_prepare")
    V_filled_max = potential_df.filter('filled_trap').filter(pl.col("V_prepare")==pl.max("V_prepare")).select('z [mm]',"V_prepare")
    ax.axvline(potential_df.filter('filled_trap').select(pl.last('z [mm]')).item(),color='tab:purple',linestyle=':')
    ax.axvline(potential_df.filter('filled_trap').select(pl.first('z [mm]')).item(),color='tab:purple',linestyle=':')
    ax.plot(potential_df.filter("interest_region")['z [mm]'],[V_filled_max['V_prepare']]*interest_region_size,color='tab:purple',linestyle=':',label='filled trap')
    ax.plot(potential_df.filter("interest_region")['z [mm]'],[V_filled_min['V_prepare']+ spacecharge_min_V if spacecharge_min_V else V_full_min['V_prepare']]*interest_region_size,color='tab:purple',linestyle=':')
    
    # plot maximum trap potential
    ax.plot(V_full_max['z [mm]'],V_full_max['V_prepare'],'^',markersize=4,color='tab:red',label="trap $V_{max}$")
    # plot trap minimum
    ax.plot(V_full_min['z [mm]'],V_full_min['V_prepare'],'v',markersize=4,color='tab:red',label="trap $V_{min}$") # ,markerfacecolor='white'

    # ax00.set_xlabel('Distance from center trap electrode [mm]')
    ax.set_xticklabels = []
    ax.set_ylabel('Potential [V]')
    ax.set_xlabel("distance from center electrode [mm]")
    ax.grid()

    # print the pulsed part and shade the area

    ax.fill_between(potential_df.filter("full_trap")['z [mm]'],
                    potential_df.filter("full_trap")['V_prepare_filled_min'],
                    potential_df.filter("full_trap")['V_prepare_filled_min'] + potential_df.filter("full_trap")['dV_filled'],
                    hatch='////',color='cornflowerblue',facecolor='lightsteelblue',
                    label="trap volume")
    
    ax.fill_between(potential_df.filter("filled_trap")['z [mm]'],
                    potential_df.filter("filled_trap")['V_launch_filled_min'],
                    potential_df.filter("filled_trap")['V_launch_filled_min'] + potential_df.filter("filled_trap")['dV_filled'],
                    hatch=r'\\\\',color='orange',facecolor='moccasin',
                    label="trap volume\nat launch")

    ax.legend(loc='lower right', bbox_to_anchor=(1.02,1.02),ncol=4) # ,prop={'size':6}

    # create bottom panel for table
    ax = fig.add_subplot(6,1,(5,6))
    # table with trap well parameters
    tbl_rows = ["Full trap","Filled trap"]
    tbl_cols = ["depth [V]","min [V]", "max [V]","width [mm]", "from [mm]","to [mm]"]
    ax.table(cellText=[[f"{(V_full_max['V_prepare'] - V_full_min['V_prepare']).item():.2f}",
                        f"{V_full_min['V_prepare'].item():.2f}",
                        f"{V_full_max['V_prepare'].item():.2f}",
                        f"{potential_df.filter('full_trap').select(pl.last('z [mm]')).item() - potential_df.filter('full_trap').select(pl.first('z [mm]')).item():.2f}",
                        f"{potential_df.filter('full_trap').select(pl.first('z [mm]')).item():.2f}",
                        f"{potential_df.filter('full_trap').select(pl.last('z [mm]')).item():.2f}"],
                       [f"{(V_filled_max['V_prepare'] - V_filled_min['V_prepare']).item():.2f}",
                        f"{V_filled_min['V_prepare'].item():.2f}",
                        f"{V_filled_max['V_prepare'].item():.2f}",
                        f"{potential_df.filter('filled_trap').select(pl.last('z [mm]')).item() - potential_df.filter('filled_trap').select(pl.first('z [mm]')).item():.2f}",
                        f"{potential_df.filter('filled_trap').select(pl.first('z [mm]')).item():.2f}",
                        f"{potential_df.filter('filled_trap').select(pl.last('z [mm]')).item():.2f}"]
                       ],
             rowLabels=tbl_rows,
             colLabels=tbl_cols,
             loc='lower center')
    
    # table with trap parameters used
    tbl_cols = ["wall [V]","floor [V]","full depth [V]", "spacecharge [eV]"]
    tbl_rows = ["Set trap"]
    # ax11 = ax[1,1] 
    ax.table(cellText=[[f"{trap_wall_V:.0f}",
                        f"{trap_floor_V:.0f}",
                        f"{trap_wall_V-trap_floor_V:.0f}",
                        f"{spacecharge_min_V if spacecharge_min_V else 0:.2f}-{spacecharge_max_V if spacecharge_max_V else trap_wall_V-trap_floor_V:.2f}"]
                       ],
             colLabels=tbl_cols,
             rowLabels=tbl_rows,
             loc='upper center')
    ax.set_axis_off()
    ax.grid()
    fig.tight_layout()
    if savefig:
        _save_fig(fig,"potential_shape",file_prefix,file_suffix,fig_format)
    
    return fig, ax


def _calculate_gennorm_params(potential_df:pl.DataFrame,
                              thermalised:float,
                              verbose_level:int = 0
                              )->tuple[float,float,float]:
    loc = potential_df.filter(pl.col("filled_trap")).select(pl.col('z [mm]').max() + pl.col('z [mm]').min()).item() / 2
    if thermalised:
        scale = potential_df.filter(pl.col("filled_trap")).select((pl.col('z [mm]').max() - pl.col('z [mm]').min())).item()/2.5
        beta = 10
    else:
        # generalised normal distribution with beta=2 is a standard normal distribution
        scale = potential_df.filter(pl.col("filled_trap")).select((pl.col('z [mm]').max() - pl.col('z [mm]').min())).item()/6
        beta = 2
    if verbose_level > 1:
        print("Initial conditions calculated with:")
        print(f" -> mean = {loc} mm")
        print(f" -> scale = {scale}")
        print(f" -> beta = {beta}")

    return loc, scale, beta


def _calculate_initial_conditions(potential_df:pl.DataFrame,
                                 spl_V_prepare:BSpline,
                                 spl_V_launch:BSpline,
                                 spl_dV:BSpline,
                                 loc:float,
                                 scale:float,
                                 beta:float,
                                 iterations:int,
                                 N_particles:int,
                                 dx_mm:float,
                                 verbose_level:int = 0
                                 )->tuple[pl.DataFrame,float,float,float]:
    '''
    Generate particles inside trapped region according to uniform distribution in potential and generlised normal distribution in z
    '''
    # we are using generalised normal distribution for the z distribution
    
    z_random = gennorm.rvs(beta=beta,loc=loc,scale=scale,size=(iterations,N_particles))
    V_launch_random = spl_V_launch(z_random)
    V_prepare_random = spl_V_prepare(z_random)
    dV_random = np.random.uniform(0,1,size=(iterations,N_particles)) * spl_dV(z_random)
    z_limit = potential_df.filter("filled_trap").select(pl.min('z [mm]')).item()

    random_initial_conditions = (pl.DataFrame({"iteration":np.arange(0,iterations),
                                               "particle":np.array([np.arange(0,N_particles)]*iterations),
                                               'z_i [mm]':z_random,
                                               "V_launch_i":V_launch_random+dV_random,
                                               "V_prepare_i":V_prepare_random+dV_random,
                                               "dV_i":dV_random})
                                 .explode("particle","z_i [mm]","V_launch_i","V_prepare_i","dV_i"))
    
    particles_df = (potential_df.filter("interest_region",pl.col('z [mm]') >= z_limit)
                    .select("z [mm]","V_launch")
                    .join(random_initial_conditions,how='cross')
                    .filter((pl.col('z [mm]')>=pl.col("z_i [mm]")) | (pl.col("z_i [mm]").is_between(pl.col('z [mm]'), pl.col('z [mm]') + dx_mm)))
                    .select("iteration","particle",'z [mm]',"z_i [mm]","V_launch","V_launch_i","V_prepare_i","dV_i")
                    )
    
    if verbose_level > 2:
        print(particles_df)
    
    return particles_df


def _plot_initial_conditions(particles_df:pl.DataFrame,
                            potential_df:pl.DataFrame,
                            loc,
                            scale,
                            beta,
                            dpi:int = 200,
                            savefig:bool = True,
                            file_prefix:str = '',
                            file_suffix:str = '',
                            fig_format:str= 'eps'
                            )->tuple[plt.Figure,plt.Axes]:
    '''
    Plot sample points from the random population
    '''
    fig,ax = plt.subplots(dpi=dpi)
    # plot full potential
    ax.plot(potential_df.filter("full_trap")['z [mm]'],potential_df.filter("full_trap")['V_launch'],color='tab:orange',label='launch potential',zorder=0)
    ax.plot(potential_df.filter("full_trap")['z [mm]'],potential_df.filter("full_trap")['V_prepare'],color='tab:blue',label='full potential',zorder=3)

    ax.set_xlabel('distance from center electrode [mm]')
    ax.set_ylabel('Potential [V]')
    ax.grid()

    # print the pulsed part and shade the area

    ax.fill_between(potential_df.filter("full_trap")['z [mm]'],
                    potential_df.filter("full_trap")['V_prepare_filled_min'],
                    potential_df.filter("full_trap")['V_prepare_filled_min'] + potential_df.filter("full_trap")['dV_filled'],
                    hatch='////',color='cornflowerblue',facecolor='lightsteelblue',
                    label="trap volume",
                    zorder=1)

    ax.fill_between(potential_df.filter("full_trap")['z [mm]'],
                    potential_df.filter("full_trap")['V_launch_filled_min'],
                    potential_df.filter("full_trap")['V_launch_filled_min'] + potential_df.filter("full_trap")['dV_filled'],
                    hatch=r'\\\\',color='orange',facecolor='moccasin',
                    label="trap volume\nat launch",
                    zorder=4)

    random_particles = (particles_df.filter(pl.col("iteration")==0,
                                  pl.col("particle") < 500,
                                  pl.col("z_i [mm]") > pl.col('z [mm]'))
                                  .select(pl.first("z_i [mm]","V_prepare_i","V_launch_i").over("particle")))
    ax.plot(random_particles["z_i [mm]"],random_particles["V_prepare_i"],'o',markersize=1,color='gainsboro',label='trapped particles',zorder=2)
    ax.plot(random_particles["z_i [mm]"],random_particles["V_launch_i"],'o',markersize=1,color='tab:gray',label='launched particles',zorder=5)
    ax.set_xlabel('distance from center electrode [mm]')
    ax.set_ylabel('Potential [V]')
    ax2 = ax.twinx()
    ax2.plot(potential_df.filter("full_trap")['z [mm]'],gennorm.pdf(potential_df.filter("full_trap")['z [mm]'],beta=beta,loc=loc,scale=scale),color='black',linestyle='--',label='density')
    ax2.set_ylabel("Density [a.u.]")
    ax.legend(loc='lower center')
    ax2.legend()
    fig.tight_layout()
    if savefig:
        _save_fig(fig,"initial_conditions",file_prefix,file_suffix,fig_format)
    return fig,ax


def _calculate_velocities(particles_df:pl.DataFrame,
                         mq:float, # m/q in keV
                         dx_mm:float,
                         verbose_level:int = 0
                         )->pl.DataFrame:
    '''
    Calculate valocities baed on the initial conditions
    '''
    # calculate v(z) first
    particles_df = (particles_df.with_columns(pl.when(pl.col('z [mm]')>=pl.col("z_i [mm]"))
                                              .then(pl.col('V_launch_i') - pl.col("V_launch"))
                                              .when(pl.col("z_i [mm]").is_between(pl.col('z [mm]'),pl.col('z [mm]')+dx_mm))
                                              .then(pl.col("dV_i"))
                                              .alias("dV")
                                              )
                                 .with_columns((np.sqrt(2 * pl.col("dV") / mq) * loader.SPEED_OF_LIGHT).alias("v [m/s]")))
    
    if verbose_level > 2:
        print("Calculated velocities:")
        print(particles_df)

    return particles_df


def _calculate_times(particles_df:pl.DataFrame,
                    remaining_distance_m:float,
                    dx_mm:float,
                    verbose_level:int = 0
                    )->pl.DataFrame:
    '''
    Calculate times it takes each particle to reach the detector.
    '''
    times_df = (particles_df.with_columns(pl.when(pl.col('z [mm]')>=pl.col("z_i [mm]"),pl.col('v [m/s]')>0)
                                          .then(dx_mm/1000 / pl.col("v [m/s]"))
                                          .when(pl.col("z_i [mm]").is_between(pl.col('z [mm]'),pl.col('z [mm]')+dx_mm),pl.col('v [m/s]')>0)
                                          .then((pl.col("z_i [mm]") - pl.col('z [mm]')) / 1000 / pl.col("v [m/s]"))
                                          .fill_null(0)
                                          .alias("dt [s]"))
                .group_by('iteration','particle',"z_i [mm]")
                .agg((remaining_distance_m / pl.col("v [m/s]").max()).alias('t_base [s]'),
                     pl.col("dt [s]").sum().alias('t_inside_trap [s]'))
                .with_columns((pl.col("t_base [s]") + pl.col("t_inside_trap [s]")).alias('t [s]'))
                )
    
    if verbose_level>2:
        print("Calculated times:")
        print(times_df)
    return times_df


def _plot_velocities_times(potential_df:pl.DataFrame,
                          particles_df:pl.DataFrame,
                          times_df:pl.DataFrame,
                          dpi:int = 200,
                          savefig:bool = True,
                          file_prefix:str = '',
                          file_suffix:str = '',
                          fig_format:str= 'eps'
                          )->tuple[plt.Figure,list[plt.Axes]]:
    '''
    Plot average velocities and final times of particles depending on the initial position of the particle inside the trap
    '''
    fig, ((ax00,ax01),(ax10,ax11)) = plt.subplots(2,2,dpi=dpi,gridspec_kw={'wspace':0.05,'hspace':0.1})
    z_bins = np.arange(particles_df.select(pl.min("z_i [mm]")).item(),particles_df.select(pl.max("z_i [mm]")).item(),0.1)
    z_in_legend = [z_bins[0], z_bins[len(z_bins)//4], z_bins[len(z_bins)//2], z_bins[-len(z_bins)//4],z_bins[-1]]
    colors = sns.color_palette('viridis',len(z_bins))
    v_plot = (particles_df.filter(pl.col('z [mm]') > pl.col('z_i [mm]'))
              .with_columns(cut=pl.col("z_i [mm]").cut(z_bins,include_breaks=True))
              .unnest("cut").rename({'breakpoint':"z_i_bin"})
              .group_by("z [mm]","z_i_bin").agg(pl.col("v [m/s]").mean(),
                                                  pl.col("v [m/s]").std().alias("v_std [m/s]")))

    t_plot = (times_df.with_columns(cut=pl.col("z_i [mm]").cut(z_bins,include_breaks=True))
              .unnest("cut").rename({'breakpoint':"z_i_bin"})
              .group_by("z_i_bin").agg(pl.col("t [s]").mean(),
                                       pl.col("t [s]").std().alias("t_std [s]")))
    
    for j,zi in enumerate(z_bins): # initial_conditions
        # plot average velocity
        v = v_plot.filter(pl.col("z_i_bin") == zi)
        if len(v) > 0:
            ax00.plot(v['z [mm]'],v['v [m/s]'],'.',color=colors[j],label=f"{np.round(zi)}" if zi in z_in_legend else None)
            ax00.errorbar(v['z [mm]'],v['v [m/s]'],yerr=v['v_std [m/s]'],fmt='none',color=colors[j])
        
        # plot average t
        t_i = t_plot.filter(pl.col("z_i_bin") == zi)
        if len(t_i) > 0:
            ax10.plot(t_i['z_i_bin'],t_i['t [s]'],'.',color=colors[j],label=f"{np.round(zi)}" if zi in z_in_legend else None)
            ax10.errorbar(t_i['z_i_bin'],t_i['t [s]'],yerr=t_i["t_std [s]"],fmt='none',color=colors[j])

    full_trap_range = potential_df.filter("full_trap").select(pl.first('z [mm]').alias("first"),pl.last('z [mm]').alias("last"))
    filled_trap_range = potential_df.filter("filled_trap").select(pl.first('z [mm]').alias("first"),pl.last('z [mm]').alias("last"))

    for ax in [ax00,ax10]:
        ax.axvline(full_trap_range['first'].item(),color='tab:green',linestyle='--')
        ax.axvline(full_trap_range['last'].item(),color='tab:green',linestyle='--')
        ax.axvline(filled_trap_range['first'].item(),color='tab:purple',linestyle=':')
        ax.axvline(filled_trap_range['last'].item(),color='tab:purple',linestyle=':')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
        ax.grid()

    ax00.set_xticklabels([])
    ax01.set_yticklabels([])
    ax10.set_xlim(*ax00.get_xlim())
    ax10.set_xlabel("distance from center electrode [mm]")
    ax00.set_ylabel(r"$\left<v\right> [m/s]$")
    ax10.set_ylabel(r"$\left<t_{TOF} \right> [s]$")
    ax10.legend(title=r'$z_{i}$ [mm]')

    # plot initial velocity distribution
    ax01.hist(particles_df.filter(pl.col("v [m/s]").is_not_nan()).sort("z_i [mm]").select(pl.col("v [m/s]").first().over(["z_i [mm]"]))["v [m/s]"],
            #   orientation='horizontal',
              bins=20,
              label=r'$v_{i}$')
    # ax01.set_ylim(*ax00.get_ylim())
    ax01.grid()
    ax01.set_xlabel(r"$v_{initial}$ [m/s]")
    ax01.set_ylabel("Count")
    ax01.yaxis.set_label_position("right")
    ax01.yaxis.tick_left()
    # ax01.set_xscale("log")
    ax11.set_axis_off()

    fig.tight_layout()
    if savefig:
        _save_fig(fig,"average_v-t",file_prefix,file_suffix,fig_format)
    return fig,[ax00,ax01,ax10,ax11]


def _calculate_tof_signal(times_df:pl.DataFrame,
                         tof_range:list[float],
                         tof_dt:float,
                         iterations:int,
                         N_particles:int,
                         mq:float,
                         label:str,
                         weight:float,
                         verbose_level:int = 0
                         )->pl.DataFrame:
    '''
    Bin the calculated times to create an expected signal
    '''
    # bin the times to create a signal
    times_df = (times_df.with_columns(cut=pl.col('t [s]').cut(np.arange(*tof_range,tof_dt),include_breaks=True))
                .unnest("cut").rename({'breakpoint':"t_bin [s]"})
                .filter(pl.col("t_bin [s]") != np.inf)
                .group_by('t_bin [s]','iteration').agg(pl.col("t [s]").count().alias("signal"))
                .group_by('t_bin [s]').agg(pl.col("signal").mean()/N_particles,
                                           (pl.col("signal").std()/np.sqrt(iterations)/N_particles).alias("signal_err"))
                .sort('t_bin [s]')
                .rename({'t_bin [s]':'t [s]'})
                .with_columns(pl.lit(label).alias("label"),
                              pl.lit(mq).alias("m/q"),
                              pl.lit(weight).alias("weight"))
                )
                            
    if verbose_level > 2:
        print("Calculated signal:")
        print(times_df)
    return times_df


def _fit(times_df:pl.DataFrame,
         fit_fcn:Callable,
         verbose_level:int = 0
         )->tuple[pl.DataFrame,list[float],list[list[float]]]:
    try:
        fit_df = times_df.filter(pl.col("signal") > 0.6*pl.col("signal").max())
        p0 = [fit_df.select(pl.col("signal").max()).item(), # the maximum of the signal as the amplitude
              fit_df.filter(pl.col('signal') == pl.col('signal').max()).select(pl.mean('t [s]')).item(), # time of the maximum as the average
              fit_df.select((pl.col("t [s]").max() - pl.col("t [s]").min())/2).item(), # width of the region as the sigma
             ]
        popt,pcov = curve_fit(fit_fcn,
                                fit_df["t [s]"],
                                fit_df["signal"],
                                sigma=fit_df["signal_err"],
                                p0=p0)
        tof_peak = pl.DataFrame({"label":times_df.select(pl.first('label')).item(),
                                 "m/q":times_df.select(pl.first('m/q')).item(),
                                 "tof_peak_s":popt[1],
                                 "tof_peak_err_s":np.sqrt(pcov[1,1])})
        
        if verbose_level > 1:
            print(f"Calculated ToF: {popt[1]}+-{np.sqrt(pcov[1,1])} s")
    except Exception as e:
        popt = None
        pcov = None
        print(e)
        print(f"WARNING:couldn't fit gaussian to the peak for mq={times_df.select(pl.first('m/q')).item()}")
        tof_peak = (pl.DataFrame({"label":times_df.select(pl.first('label')).item(),
                                  "m/q":times_df.select(pl.first('m/q')).item(),
                                  "tof_peak_s":np.nan,
                                  "tof_peak_err_s":np.nan}))
        
    return tof_peak,popt,pcov


def _plot_tof_signal(times_df:pl.DataFrame,
                    color:float,
                    tof_dt:float,
                    fit_fcn:Callable,
                    popt_fit:Optional[list[float]] = None,
                    pcov_fit:Optional[list[list[float]]] = None,
                    calibration:bool = False,
                    dpi:int = 200,
                    savefig:bool = True,
                    file_prefix:str = '',
                    file_suffix:str = '',
                    fig_format:str= 'eps'
                    )->tuple[plt.Figure,plt.Axes]:
    '''
    Plot binned tof signal
    '''
    fig,ax = plt.subplots(dpi=dpi)
    ax.errorbar(times_df['t [s]'],times_df['signal'],
                times_df['signal_err'],
                fmt='o',
                capsize=0.1,
                markersize=2,
                color=color, # sns.color_palette("Set2",1)[0],
                label=times_df.select(pl.first("label")).item(),
                zorder=1)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("signal [a.u.]")
    ax.legend(loc='upper right',bbox_to_anchor=(1.0,1.0)) # prop={'size':6}

    if popt_fit is not None:
        _plot_fit(ax,np.arange(times_df.select(pl.min('t [s]')).item(),times_df.select(pl.max('t [s]')).item(),tof_dt),fit_fcn,popt_fit,pcov_fit)

    if calibration:
        _plot_calibration_tof(ax)

    ax.grid()
    fig.tight_layout()
    if savefig:
        _save_fig(fig,"tof_signal",file_prefix,file_suffix,fig_format)
    return fig,ax


def _plot_fit(ax:plt.Axes,
              t_s:Union[list[float],np.ndarray],
              fit_fcn:Callable,
              popt_fit:list[float],
              pcov_fit:list[list[float]])->plt.Axes:
    '''
    Function for ploting the gauss fit on top of the signal plot
    '''
    ax.plot(t_s,fit_fcn(t_s,*popt_fit),color='black',label='fit',zorder=2)
    ax.axvline(popt_fit[1],0,1,linestyle='--',color='tab:gray',label=r"$ToF_{fit}$") # ,linewidth=0.3
    ax.axvspan(popt_fit[1] - np.sqrt(pcov_fit[1,1]),popt_fit[1]+np.sqrt(pcov_fit[1,1]),0,1,color="lightgray")

    return ax


def _plot_calibration_tof(ax:plt.Axes)->plt.Axes:
    ax.axvline(6.493e-6,0, 1,linestyle='--',label=r"$ToF_{\bar{p}}^{calib}$")
    return ax


def _plot_total_tof_signal(tof_signal:pl.DataFrame,
                          many_species:bool,
                          dpi:int = 200,
                          savefig:bool = True,
                          file_prefix:str = '',
                          file_suffix:str = '',
                          fig_format:str= 'eps'
                          )->tuple[plt.Figure,plt.Axes]:
    # plot total ToF signal
    colors = sns.color_palette("Set2",8)
    fig,ax = plt.subplots(dpi=dpi)

    ToF_signal_total = (tof_signal.group_by('t [s]')
                        .agg((pl.col("signal")*pl.col("weight")).sum(),
                             (pl.col("signal_err")*pl.col("weight")).sum())
                        .with_columns(lower=pl.col("signal")-pl.col("signal_err"),
                                      upper=pl.col("signal")+pl.col("signal_err")))
    
    ax.errorbar(ToF_signal_total['t [s]'],ToF_signal_total['signal'],ToF_signal_total['signal_err'],fmt='o',markersize=2,color=colors[0],linewidth=1,label='total signal')  
    ax.set_xlabel("t [s]")
    ax.set_ylabel("signal [a.u.]")

    if many_species:
        # select first 5 fragments with highest contribution
        labels_to_plot = pl.Series(tof_signal.unique(['label']).sort("weight",descending=True).select(pl.col("label").head(6))).to_list()
        ToF_signal_other = (tof_signal.filter(pl.col("label").is_in(labels_to_plot).not_())
                            .group_by('t [s]').agg((pl.col("signal")*pl.col("weight")).sum(),
                                                    (pl.col("signal_err")*pl.col("weight")).sum())
                            .sort('t [s]')
        )
        for i,label in enumerate(labels_to_plot):
            tof_tmp = tof_signal.filter(pl.col("label") == label).sort('t [s]').with_columns(pl.col("signal")*pl.col("weight"))
            ax.plot(tof_tmp['t [s]'],tof_tmp['signal'],label=label,linewidth=1,color=colors[1+i])
        ax.plot(ToF_signal_other['t [s]'],ToF_signal_other['signal'],linewidth=1,label="others",color=colors[-1])

    ax.legend(loc='upper right',bbox_to_anchor=(1.0,1.0)) # prop={'size':6}
    ax.grid()
    fig.tight_layout()
    if savefig:
        _save_fig(fig,"tof_signal",file_prefix,file_suffix,fig_format)
    return fig,ax


def calculate_tof(mq:Union[float,list[float]],
                  labels: Union[str,list[str],None] = None,
                  weights: Union[float,list[float],None] = None,
                  trap_floor_V:float=args.trap_floor_V, # V
                  trap_wall_V:float=args.trap_wall_V, # V
                  pulse_wall_to_V:float=args.pulse_wall_to_V, # V
                  distance_to_mcp_m:float=args.distance_to_mcp_m, # m
                  spacecharge_min_V:Optional[float]=args.spacecharge_min_V, # V
                  spacecharge_max_V:Optional[float]=args.spacecharge_max_V, # V
                  trap_left_wall:Union[str,list[str]]=args.trap_left_wall,
                  trap_floor:Union[str,list[str]]=args.trap_floor,
                  trap_right_wall:Union[str,list[str]]=args.trap_right_wall,
                  thermalised:bool=args.thermalised,
                  N_particles:int=args.N_particles,
                  iterations:int=args.iterations,
                  tof_range:list[float]=args.tof_range, # s
                  tof_dt:float=args.tof_dt, # s
                  verbose_level:int=args.verbose_level,
                  plot_mask:int=args.plot_mask,
                  dpi:int=args.dpi,
                  savedata_mask:int=args.savedata_mask,
                  savefig:bool=args.savefig,
                  showfig:bool=args.showfig,
                  file_prefix:str=args.file_prefix,
                  fig_format:str=args.fig_format)->tuple[pl.DataFrame,pl.DataFrame]:
    # make parameters into proper format
    if isinstance(mq,(float,int)):
        mq = [mq]
    if labels is None:
        labels = [f"{mq_i}" for mq_i in mq]
    elif isinstance(labels,str):
        labels = [labels]
    if weights is None:
        weights = [1]*len(mq)
    elif isinstance(weights,(float,int)):
        weights = [weights]

    potential = _calculate_potential_shape(trap_floor_V=trap_floor_V,
                                           trap_wall_V=trap_wall_V,
                                           pulse_wall_to_V=pulse_wall_to_V,
                                           trap_left_wall=trap_left_wall,
                                           trap_floor=trap_floor,
                                           trap_right_wall=trap_right_wall,
                                           verbose_level=verbose_level)
    
    potential,spacecharge_min_V,spacecharge_max_V = _fill_trap(potential_df=potential,
                           spacecharge_min_V=spacecharge_min_V,
                           spacecharge_max_V=spacecharge_max_V,
                           verbose_level=verbose_level)
    file_suffix = f"floor={trap_floor_V}V_wall={trap_wall_V}V_spacecharge={spacecharge_min_V:.2f}-{spacecharge_max_V:.2f}V_dt={tof_dt*1e9:.0f}ns_N={N_particles}x{iterations}"
    
    if savedata_mask & 0x02:
        _save_data(data=potential,title='potential',file_prefix=file_prefix,file_suffix=file_suffix)

    if bool(plot_mask & 0x01) & (showfig | savefig):
        print("Plotting fig with potentials")
        fig1,_ = _plot_potential_shape(potential_df=potential,
                                       trap_floor_V=trap_floor_V,
                                       trap_wall_V=trap_wall_V,
                                       spacecharge_min_V=spacecharge_min_V,
                                       spacecharge_max_V=spacecharge_max_V,
                                       dpi=dpi,
                                       savefig=savefig,
                                       file_prefix=file_prefix,
                                       file_suffix=file_suffix,
                                       fig_format=fig_format)
        if not showfig:
            plt.close(fig1)

    # create interpolation functions for calculating intial conditions
    spl_V_launch = make_interp_spline(potential.filter("filled_trap")["z [mm]"],potential.filter("filled_trap")["V_launch_filled_min"],k=1)
    spl_V_prepare = make_interp_spline(potential.filter("filled_trap")["z [mm]"],potential.filter("filled_trap")["V_prepare_filled_min"],k=1)
    spl_dV = make_interp_spline(potential.filter("filled_trap")["z [mm]"],potential.filter("filled_trap")["dV_filled"],k=1)
    loc,scale,beta = _calculate_gennorm_params(potential,thermalised,verbose_level)

    # for each m/q caclulate expected tof signal for particles inside trap
    ToF_signal:list[pl.DataFrame] = []
    # for each signal find the peak position
    ToF_peak:list[pl.DataFrame] = []

    for i,mqi in enumerate(mq):
        if verbose_level > 1:
            print(f"m/q={mqi}")
        mq_eV = mqi * loader.ATOMIC_UNIT_TO_KEV * 1000

        ToF_expected_lower = (distance_to_mcp_m - potential.filter("filled_trap").select(pl.last('z [mm]')).item()/1000) * np.sqrt((mq_eV) / (2*trap_floor_V)) / loader.SPEED_OF_LIGHT
        ToF_expected_upper = (distance_to_mcp_m - potential.filter("filled_trap").select(pl.first('z [mm]')).item()/1000) * np.sqrt((mq_eV) / (2*trap_floor_V)) / loader.SPEED_OF_LIGHT
        ToF_expected = distance_to_mcp_m * np.sqrt((mq_eV) / (2*trap_floor_V)) / loader.SPEED_OF_LIGHT
        if verbose_level > 1:
            print("Distance to mcp:",distance_to_mcp_m,"m")
            print("Distance range:",distance_to_mcp_m - potential.filter("filled_trap").select(pl.last('z [mm]')).item()/1000,"m :", distance_to_mcp_m - potential.filter("filled_trap").select(pl.first('z [mm]')).item()/1000,"m")
            print("Expected ToF:",ToF_expected,"s")
            print("Expected ToF range:",ToF_expected_lower,ToF_expected_upper,"s")

        particles = _calculate_initial_conditions(potential_df=potential,
                                                  spl_V_prepare=spl_V_prepare,
                                                  spl_V_launch=spl_V_launch,
                                                  spl_dV=spl_dV,
                                                  loc=loc,
                                                  scale=scale,
                                                  beta=beta,
                                                  iterations=iterations,
                                                  N_particles=N_particles,
                                                  dx_mm=TTrap.dx,
                                                  verbose_level=verbose_level)
        
        if bool(plot_mask & 0x02) & (showfig | savefig):
            fig2,_ = _plot_initial_conditions(particles_df=particles,
                                              potential_df=potential,
                                              loc=loc,
                                              scale=scale,
                                              beta=beta,
                                              dpi=dpi,
                                              savefig=savefig,
                                              file_prefix=file_prefix,
                                              file_suffix=f"mq={mqi}_"+file_suffix,
                                              fig_format=fig_format)
            if not showfig:
                plt.close(fig2)
        
        particles = _calculate_velocities(particles_df=particles,
                                          mq=mq_eV,
                                          dx_mm=TTrap.dx,
                                          verbose_level=verbose_level)
        
        if savedata_mask & 0x04:
            _save_data(data=particles,title='particles',file_prefix=file_prefix,file_suffix=f"mq={mqi}_"+file_suffix)
        
        # distance_to_mcp_ is from the center electrode.
        remaining_distance_m = distance_to_mcp_m - particles.select(pl.max('z [mm]')).item()/1000
        times = _calculate_times(particles_df=particles,
                                 remaining_distance_m=remaining_distance_m,
                                 dx_mm=TTrap.dx,
                                 verbose_level=verbose_level)
        
        if savedata_mask & 0x08:
            _save_data(data=times,title='times',file_prefix=file_prefix,file_suffix=f"mq={mqi}_"+file_suffix)

        if bool(plot_mask & 0x04) & (showfig | savefig):
            fig3,_ = _plot_velocities_times(potential_df=potential,
                                            particles_df=particles,
                                            times_df=times,
                                            dpi=dpi,
                                            savefig=savefig,
                                            file_prefix=file_prefix,
                                            file_suffix=f"mq={mqi}_"+file_suffix,
                                            fig_format=fig_format)
            if not savefig:
                plt.close(fig3)

        # bin the times to create a signal
        times = _calculate_tof_signal(times_df=times,
                                      tof_range=tof_range,
                                      tof_dt=tof_dt,
                                      iterations=iterations,
                                      N_particles=N_particles,
                                      mq=mqi,
                                      label=labels[i],
                                      weight=weights[i],
                                      verbose_level=verbose_level)
        ToF_signal.append(times)

        peak, popt, pcov = _fit(times_df=times,
                                fit_fcn=_fit_function,
                                verbose_level=verbose_level)
        ToF_peak.append(peak)
        
        if bool(plot_mask & 0x08) & (showfig | savefig):
            # plot signal for mq_eV
            fig4,_ = _plot_tof_signal(times_df=times,
                                      color=sns.color_palette("Set2",8)[0],
                                      tof_dt=tof_dt,
                                      fit_fcn=_fit_function,
                                      popt_fit=popt,
                                      pcov_fit=pcov,
                                      calibration=False,
                                      dpi=dpi,
                                      savefig=savefig,
                                      file_prefix=file_prefix,
                                      file_suffix=f"mq={mqi}_"+file_suffix,
                                      fig_format=fig_format)
            if not showfig:
                plt.close(fig4)
            
        # end of the for loop over list of  M/Q
    ToF_signal = pl.concat(ToF_signal)
    ToF_peak = pl.concat(ToF_peak)
    if savedata_mask & 0x01:
        _save_data(data=ToF_signal,title="tof_signal",file_prefix=file_prefix,file_suffix=file_suffix)
        _save_data(data=ToF_peak,title="tof_peak",file_prefix=file_prefix,file_suffix=file_suffix)
        
    if verbose_level > 1:
        print("ToF signals:")
        print(ToF_signal)
    if verbose_level > 0:
        print("ToF peak values:")
        print(ToF_peak)

    if bool(plot_mask & 0x10) & (showfig | savefig):
        fig5,_ = _plot_total_tof_signal(tof_signal=ToF_signal,
                                        many_species=len(mq)>1,
                                        dpi=dpi,
                                        savefig=savefig,
                                        file_prefix=file_prefix,
                                        file_suffix=file_suffix,
                                        fig_format=fig_format)
        if not showfig:
            plt.close(fig5)

    return ToF_signal, ToF_peak


#################################################################
def tof()->None:
    calculate_tof(mq=args.mq,
                  labels=args.label,
                  weights=args.weight)

def tof_pbar()->None:
    calculate_tof(mq=1.007276,
                  labels=r"$\bar{p}$",
                  weights=args.weight,
                  plot_mask=0x1f if args.plot_mask == 0 else args.plot_mask,
                  file_prefix="pbar" if args.file_prefix=='' else args.file_prefix)


def tof_fragments()->None:
    df_simulated_data = pl.read_parquet(loader.build_data_path("trappable_Ar40_m-over-q.parquet"))
    df_simulated_data = df_simulated_data.with_columns(pl.col("tof_s").sub(pl.col("tof_exp_s")).alias("dt"))
    print(df_simulated_data)

    calculate_tof(mq = df_simulated_data['m_over_q'],
                  labels=df_simulated_data['label'],
                  weights=df_simulated_data['trappable_count'],
                  plot_mask=0x11 if args.plot_mask == 0 else args.plot_mask,
                  file_prefix="Ar40" if args.file_prefix=='' else args.file_prefix)

def tof_calibration()->None:
    mq_pbar=1.007276 # a.u./e

    tof_pbar_150 = pl.concat([calculate_tof(mq=mq_pbar,
                                            labels=r'$\bar{p}$',
                                            trap_floor_V=floor,
                                            trap_wall_V=150,
                                            verbose_level=1,
                                            plot_level=0x1b if floor in [30,50,80,110,130] else 0,
                                            showfig=True if floor in [30,50,80,110,130] else 0)[1].with_columns(pl.lit(150).alias("wall_V"),
                                                                                                                pl.lit(floor*1.0).alias("floor_V")) 
                              for floor in range(30,135,5)])

    tof_pbar_190 = pl.concat([calculate_tof(mq_pbar,
                                            labels=r'$\bar{p}$',
                                            trap_floor_V=floor,
                                            trap_wall_V=190,
                                            verbose_level=1,
                                            plot_level=0x1b if floor in [155,175] else 0,
                                            showfig=True if floor in [155,175] else False)[1].with_columns(pl.lit(190).alias("wall_V"),
                                                                                                           pl.lit(floor*1.0).alias("floor_V"))
                              for floor in range(155,180,5)]) # range(30,135,5)
    
    tof_pbar = pl.concat([tof_pbar_150,tof_pbar_190])

    tof_pbar = (tof_pbar.with_columns((1/pl.col("tof_peak_s").pow(2.0)).alias("1/t^2"),
                                      (mq_pbar*loader.PROTON_MASS_KEV * 1000 / loader.SPEED_OF_LIGHT**2 / 2/pl.col("tof_peak_s")**2.0).alias("m/2qt^2"))
                    )
    print(tof_pbar)

    # fits according to the pbar calibration from the paper
    # wall = 160 V
    x = np.linspace(30,200,4)
    y1 = _lin(x, 1.264, -7.6)
    # wall = 190 V
    y2 = _lin(x, 1.159, 7.2)

    cpalette = sns.color_palette('hls',7)

    fig_calib = plt.figure(dpi=150)
    ax_calib = fig_calib.add_subplot()
    ax_calib.set_xlabel(r"$\frac{m}{2 q t^{2}}~[s^{-2}]$")
    ax_calib.set_ylabel("$V_{floor}$ [V]")
    ax_calib.plot(x,y1,label='fit(data) $V_{wall}=160$ V',color=cpalette[0])
    ax_calib.plot(x,y2,label='fit(data) $V_{wall}=190$ V',color=cpalette[1])

    popt = []
    pcov = []
    for i,wall in enumerate(tof_pbar.select(pl.col('wall_V').unique())['wall_V']):
        tof_i = tof_pbar.filter(pl.col("wall_V") == wall)
        ax_calib.plot(tof_i["m/2qt^2"],tof_i["floor_V"],'o',label='$t_{max(peak)}$'+f" (wall={wall} V)",color=cpalette[2+2*i])
        popt_i,pcov_i = curve_fit(_lin,tof_i["m/2qt^2"],tof_i["floor_V"],p0=[1.05**2,0])
        popt.append(popt_i)
        pcov.append(pcov_i)
        print(popt_i)
        x = np.array(ax_calib.get_xlim())
        # ax_calib.plot(x,lin_fit(x,*popt_i),'r--',label=f"fit (wall={wall} V)",color=cpalette[2+2*i+1])

    ax_calib.legend()
    ax_calib.legend()

    print(f"mq={mq_pbar*loader.ATOMIC_UNIT_TO_KEV} keV")
    print("A(wall_V=160 V)=1.264")
    print(f"A(wall=160 V)={popt[0][0]}")
    print(f"L={np.sqrt(popt[0][0])}")
    print("A(wall=190 V)=1.159")
    print(f"A(wall=190 V)={popt[1][0]}")
    print(f"L={np.sqrt(popt[1][0])}")
    fig_calib.tight_layout()
    fig_calib.savefig(os.path.join(os.path.dirname(__file__),"plots","pbar_calibration.eps"),dpi=600)
    plt.show()


#################################################################


if __name__ == "__main__":
    for key in args.__dict__:
        print(f'{key}:({type(args.__dict__[key])}){args.__dict__[key]}')
    if args.command == 'tof':
        tof()
    elif args.command == 'tof_pbar':
        tof_pbar()
    elif args.command == 'tof_fragments':
        tof_fragments()
    elif args.command == 'tof_calibration':
        tof_calibration()
    if args.showfig:
        plt.show()
