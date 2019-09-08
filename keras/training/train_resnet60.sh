#!/bin/bash
#PBS -N vmmr_sample6
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:gpu_cap=cuda35:scratch_local=10gb
#PBS -l walltime=24:00:00

trap 'clean_scratch' TERM EXIT # setup SCRATCH cleaning in case of an error

module add cuda-7.0 
module add cudnn-7.0
# module add python-3.6.2-gcc
module add tensorflow-1.7.1-gpu-python3

echo $PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH

pip3 install --upgrade pip
pip3 install Pillow
pip3 install numpy
pip3 install tensorflow-gpu
pip3 install scipy
pip3 install scikit-image
pip3 install matplotlib
pip3 install Keras 
pip3 install keras-vis 
pip3 install scikit-learn 

pip3 list

echo "---------"
ls -l $SCRATCHDIR/cuda
echo "========"
find / -type d -name cuda 2>/dev/null

# data source directory
DATADIR="/storage/plzen1/home/radekj/vmmr"
DATASET="sample6"

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

echo $(python3 $DATADIR/tasks/train_resnet60.py "$DATADIR/datasets/$DATASET/")
# echo $(python3 train_resnet60.py "$SCRATCHDIR/$DATASET/")
# python3 < svm_classifier.py > results_$DATASET.out #

cp -R $SCRATCHDIR/tasks/. $DATADIR/results/$DATASET
# cp results_$DATASET.out $DATADIR
cd $SCRATCHDIR # moves to your scratch diretory

export CLEAN_SCRATCH=true
