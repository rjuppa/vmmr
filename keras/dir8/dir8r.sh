#!/bin/bash
#PBS -N vmmr_test
#PBS -l select=1:ncpus=8:ngpus=1:mem=8gb:scratch_local=10gb:ompthreads=2
#PBS -l walltime=24:00:00

trap 'clean_scratch' TERM EXIT # setup SCRATCH cleaning in case of an error

# data source directory
DATADIR="/storage/plzen1/home/radekj/vmmr"
DATASET="dir8r"

module add tensorflow-1.7.1-gpu-python3

# virtual env.
source "/storage/plzen1/home/radekj/vmmr/venv_g/bin/activate"

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
echo $(/storage/plzen1/home/radekj/vmmr/venv_g/bin/python3 $DATADIR/tasks/train_dir8r.py "$DATADIR/datasets/$DATASET/") > "$DATADIR/results/$DATASET/result_test.out"
echo "============= end python"

cp -R $SCRATCHDIR/tasks/. $DATADIR/results/$DATASET
cd $SCRATCHDIR # moves to your scratch diretory

export CLEAN_SCRATCH=true

