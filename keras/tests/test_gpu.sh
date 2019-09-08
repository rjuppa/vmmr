#!/bin/bash
#PBS -N vmmr_test
#PBS -l select=1:ncpus=4:ngpus=1:gpu_cap=cuda35:mem=16gb:scratch_local=10gb:ompthreads=2 
#PBS -l walltime=24:00:00
#PBS -q gpu

trap 'clean_scratch' TERM EXIT # setup SCRATCH cleaning in case of an error


# data source directory
DATADIR="/storage/plzen1/home/radekj/vmmr"
DATASET="sample6"

# fix: error while loading shared libraries: libpython3.6m.so.1.0: cannot open shared object file
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/software/python-3.6.2/gcc/lib


module add tensorflow-1.7.1-gpu-python3

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/software/python-3.6.2/gcc/lib
source "/storage/plzen1/home/radekj/vmmr/venv_g/bin/activate"

# source "$DATADIR/run_venv_gpu.sh"

echo "PATH: $PATH"
echo "------"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "------"
echo "VE: $VIRTUAL_ENV"
pip3 freeze

# checks if scratch directory is created
if [ ! -d "$SCRATCHDIR" ] ; then
  echo "Scratch directory is not created!" 1>&2; exit 1;
fi

# enters user's scratch directory
cd $SCRATCHDIR || exit 1
pwd

# cp $DATADIR/tasks/train_resnet60.py $SCRATCHDIR/train_resnet60.py

echo "============= start python"
echo $(/storage/plzen1/home/radekj/vmmr/venv_g/bin/python3 $DATADIR/tasks/train_resnet60.py "$DATADIR/datasets/$DATASET/") > "$DATADIR/results/$DATASET/result_test.out"
echo "============= end python"

cd $SCRATCHDIR # moves to your scratch diretory

# export CLEAN_SCRATCH=false
