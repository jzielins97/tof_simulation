#!/bin/sh
# This script executes a full chain of path calculation from ROOT files to final parquet file

pwd
analysis_path="/afs/cern.ch/work/j/jzielins/public/tof_simulation"
analysis_script=calculate_tof.py
# check if analysis script was coppied
if [ ! -f ${analysis_script} ]; then
    echo "ERROR: file ${analysis_script} wasn't copied correctly." >&2
    exit 1
fi
# create subdirectories for plots and data
mkdir data plots
# check if all the files were copied
ls -A
# activate python from cvmfs
. /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh
# add custom libraries for python (polars)
export PYTHONPATH=./site-packages/:$PYTHONPATH

# handle parameters
if [ $# -lt 5 ];then
    echo "ERROR: not enough parameters" >&2
    echo "usage: SimulateTof <cluster id> <process id> [--parameter value] <command:[tof,tof_pbar,tof_fragments]> [--command_parameter value]" >&2
    echo "All parameters are defined in calculate_path.py script." >&2
    python calculate_tof.py -h >&2
    exit 1
fi

# default arguments for keeping everything clean
CLUSTERID=$1
PROCESSID=$2
echo "Running job ${CLUSTERID}.${PROCESSID}"
# remove the parameters
shift
shift
# parse arguments
pre_args=() # general parameters before the command
post_args=() # command specific parameters after the command
read_command=0
while [ $# -gt 0 ]; do
  case "$1" in
      --*)
	  # parameter case
	  if [ $read_command -eq 1 ];then
	      # command specific parameters
	      post_args+=($1 $2)
	  else
	      # general parameters
	      pre_args+=($1 $2)
	  fi
	  shift
	  shift
	  ;;
    *)
	command=$1
	read_command=1
	shift
	;;
  esac
done

# run simulation code
echo python calculate_tof.py "${pre_args[@]}" "${command}" "${post_args[@]}"
python calculate_tof.py "${pre_args[@]}" "${command}" "${post_args[@]}"

# check if any data was created
if [ "$(ls -A data)" ]; then
    # cp data/* ${analysis_path}/data/
    echo "Created following data files:"
    ls data
else
    echo "WARNING: no data created" >&2
fi

# check if any plot was created
if [ "$(ls -A plots)" ]; then
    # cp plots/* ${analysis_path}/plots/
    echo "Created following plots"
    ls plots
else
    echo "WARNING: no plots created" >&2
fi
