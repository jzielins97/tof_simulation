#!/bin/sh
# This script executes a full chain of path calculation from ROOT files to final parquet file

if [ $# -lt 7 ];then
    echo "ERROR:not enough paramters" >&2
    exit 2
fi

# # main path where the thing starts
# INIT_PATH=$( pwd )

# # default arguments for keeping everything clean
CLUSTERID=$1
PROCESSID=$2
echo "Running job ${CLUSTERID}.${PROCESSID}"
# TMPDIR=$INIT_PATH/tof_$CLUSTERID.$PROCESSID
# mkdir $TMPDIR
# if [ -d "$TMPDIR" ];then
#     cd $TMPDIR
# else
#     echo "WARNING: file ${TMPDIR} was not created, used a tmpdir as the name"
#     TMPDIR=tmpdir
#     mkdir $TMPDIR
#     cd $TMPDIR
#     pwd
# fi
pwd
# other arguments
TRAP_FLOOR=$3
TRAP_WALL=$4
SPACECHARGE_MIN=$5
SPACECHARGE_MAX=$6
FRAGMENTS_DATA_FILE=$7

analysis_path="/afs/cern.ch/work/j/jzielins/public/tof_simulation"
analysis_script=calculate_tof.py

# for file in ${analysis_script} modules
# do
#     cp -r ${analysis_path}/$file $TMPDIR/
# done

if [ ! -f ${analysis_script} ]; then
    echo "ERROR: file ${analysis_script} wasn't copied correctly." >&2
    exit 1
fi
# create subdirectories for plots and data
mkdir data plots

# check if all the files were copied
ls -A

# activate python venv
# source ${analysis_path}/venv/bin/activate
. /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh
export PYTHONPATH=./site-packages/:$PYTHONPATH

# run simulation code
python calculate_tof.py --trap_wall_V ${TRAP_WALL}  --trap_floor_V ${TRAP_FLOOR} --spacecharge_min_V ${SPACECHARGE_MIN} --savedata_mask 0x1 --showfig False --savefig True --fig_format png --verbose_level 3 tof_fragments --fragments_data_path ${FRAGMENTS_DATA_FILE}

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
else
    echo "WARNING: no plots created" >&2
fi

# cd $INIT_PATH
# if [[ -d $TMPDIR && $TMPDIR != "/afs/cern.ch/user/j/jzielins" ]];then
#     echo "I'm in $( pwd ) trying to rm ${TMPDIR}"
#     rm -r $TMPDIR
# fi
