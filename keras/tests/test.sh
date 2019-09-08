#!/bin/bash
#PBS -N vmmr_test
#PBS -l select=1:ncpus=1:mem=1gb:scratch_local=1gb:ompthreads=1 
#PBS -l walltime=1:00:00

trap 'clean_scratch' TERM EXIT # setup SCRATCH cleaning in case of an error


# data source directory
DATADIR="/storage/plzen1/home/radekj/vmmr"
DATASET="test"

# fix: error while loading shared libraries: libpython3.6m.so.1.0: cannot open shared object file
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/software/python-3.6.2/gcc/lib

source "$DATADIR/run_venv_cpu.sh"

echo "PATH: $PATH"
echo "------"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "------"
echo "VE: $VIRTUAL_ENV"
pip freeze

# checks if scratch directory is created
if [ ! -d "$SCRATCHDIR" ] ; then
  echo "Scratch directory is not created!" 1>&2; exit 1;
fi

# enters user's scratch directory
cd $SCRATCHDIR || exit 1
pwd

cp $DATADIR/tasks/test.py $SCRATCHDIR/test.py

echo "============= start python"
python3 "$SCRATCHDIR/test.py" > "$DATADIR/results/$DATASET/result_test.out"
echo "============= end python"

cd $SCRATCHDIR # moves to your scratch diretory

# export CLEAN_SCRATCH=false
