#!/bin/sh
# This script executes a full chain of path calculation from ROOT files to final parquet file

if [ $# -lt 4 ];then
    echo "ERROR:not enough paramters" >&2
    exit 2
fi

# main path where the thing starts
INIT_PATH=$( pwd )

# default arguments for keeping everything clean
CLUSTERID=$1
PROCESSID=$2

TMPDIR=$INIT_PATH/tof_$CLUSTERID.$PROCESSID
mkdir $TMPDIR
if [ -d "$TMPDIR" ];then
    cd $TMPDIR
else
    echo "WARNING: file ${TMPDIR} was not created, used a tmpdir as the name"
    TMPDIR=tmpdir
    mkdir $TMPDIR
    cd $TMPDIR
    pwd
fi

# other arguments
SPACECHARGE_MIN=$3
SPACECHARGE_MAX=$4

analysis_path="/afs/cern.ch/work/j/jzielins/public/tof_simulation"
analysis_script=calculate_tof.py

for file in ${analysis_script} modules
do
    cp -r ${analysis_path}/$file $TMPDIR/
done

if [ ! -f ${analysis_script} ]; then
    echo "ERROR: file ${analysis_script} wasn't copied correctly." >&2
    exit 1
fi
# create subdirectories for plots and data
mkdir data plots

# check if all the files were copied
ls
ls modules

# activate python venv
source ${analysis_path}/venv/bin/activate

# run simulation code
python calculate_tof.py --trap_wall_V 150  --trap_floor_V 120 --spacecharge_min_V ${SPACECHARGE_MIN} --spacecharge_max_V ${SPACECHARGE_MAX} --savedata_mask 0xf --plot_mask 1f --showfig False --savefig True --fig_format png --verbose_level 3 tof_pbar

# check if any data was created
if [ "$(ls -A data)" ]; then
    cp data/* ${analysis_path}/data/
else
    echo "WARNING: no data created" >&2
fi

# check if any plot was created
if [ "$(ls -A plots)" ]; then
    cp plots/* ${analysis_path}/plots/
else
    echo "WARNING: no plots created" >&2
fi

cd $INIT_PATH
if [[ -d $TMPDIR && $TMPDIR != "/afs/cern.ch/user/j/jzielins" ]];then
    echo "I'm in $( pwd ) trying to rm ${TMPDIR}"
    rm -r $TMPDIR
fi
