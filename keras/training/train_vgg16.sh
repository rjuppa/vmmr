#!/bin/bash
#PBS -N vmmr_vgg16
#PBS -l select=1:ncpus=16:mem=16gb:scratch_local=10gb:ompthreads=6
#PBS -l walltime=48:00:00

trap 'clean_scratch' TERM EXIT # setup SCRATCH cleaning in case of an error

# data source directory
DATADIR="/storage/plzen1/home/radekj/vmmr"
DATASET="vgg16"

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

echo $(python3 $DATADIR/tasks/train_vgg16.py "$DATADIR/datasets/$DATASET/")
# python3 < "$DATADIR/tasks/train_vgg16.py" > "$DATADIR/results/$DATASET/results.out"

# cp results_$DATASET.out $DATADIR
cd $SCRATCHDIR # moves to your scratch diretory

export CLEAN_SCRATCH=true
