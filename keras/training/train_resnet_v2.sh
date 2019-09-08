#!/bin/bash
#PBS -N vmmr_resnet_v2
#PBS -l select=1:ncpus=16:mem=16gb:scratch_local=10gb:ompthreads=6
#PBS -l walltime=48:00:00

trap 'clean_scratch' TERM EXIT # setup SCRATCH cleaning in case of an error

module add tensorflow-1.7.1-cpu-python3

pip3 install --upgrade --user setuptools pip
pip3 install Pillow --user
pip3 install numpy --user
pip3 install scipy
pip3 install matplotlib
pip3 install Keras --user
pip3 install keras-vis --user
pip3 install scikit-learn --user
pip3 install unicode --user
pip3 install Unidecode --user
pip3 install python-dateutil --user

# data source directory
DATADIR="/storage/plzen1/home/radekj/vmmr"
DATASET="resnet_v2"

# checks if scratch directory is created
if [ ! -d "$SCRATCHDIR" ] ; then
  echo "Scratch directory is not created!" 1>&2; exit 1;
fi

# enters user's scratch directory
cd $SCRATCHDIR || exit 1
pwd

# create target directory
mkdir -p $DATASET
cp -R $DATADIR/datasets/$DATASET/. $SCRATCHDIR/$DATASET/ # gets job's input data


mkdir tasks
cp -R $DATADIR/tasks/. tasks
cd tasks

echo $(python3 train_resnet_v2.py "$SCRATCHDIR/$DATASET/")
# python3 < svm_classifier.py > results_$DATASET.out #

cp -R $SCRATCHDIR/tasks/. $DATADIR/results/$DATASET
# cp results_$DATASET.out $DATADIR
cd $SCRATCHDIR # moves to your scratch diretory

export CLEAN_SCRATCH=true
