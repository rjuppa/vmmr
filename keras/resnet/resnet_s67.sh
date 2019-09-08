#!/bin/bash
#PBS -N models_s67
#PBS -l select=1:ncpus=8:ngpus=2:gpu_cap=cuda35:mem=8gb:scratch_local=6gb:ompthreads=2:mpiprocs=8 
#PBS -l walltime=24:00:00
#PBS -q gpu

trap 'clean_scratch' TERM EXIT # setup SCRATCH cleaning in case of an error

# data source directory
DATADIR="/storage/plzen1/home/radekj/vmmr"
DATASET="sample67"

module add tensorflow-1.13.1-gpu-python3

# virtual env.
source "/storage/plzen1/home/radekj/vmmr/venv_gpu/bin/activate"

echo $DATASET
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

  echo "============= start python"
  echo $(/storage/plzen1/home/radekj/vmmr/venv_gpu/bin/python3 $DATADIR/tasks/resnet_s67.py "$DATADIR/datasets/$DATASET/Skoda") > "$DATADIR/results/$DATASET/result_test.out"
  echo "============= end python"

#  cp -R $SCRATCHDIR/tasks/. $DATADIR/results/$DATASET
  cd $SCRATCHDIR # moves to your scratch diretory

  # export CLEAN_SCRATCH=false