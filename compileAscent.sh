#/bin/bash

# Begin LSF Directives

#BSUB -P TRN008

#BSUB -W 1:00

#BSUB -nnodes 4

#BSUB -alloc_flags gpumps

#BSUB -J COMPILE

#BSUB -o COMPILE.%J

#BSUB -e COMPILE.%J

rm -frd build
mkdir build
cd build
module purge
module load cmake
module load gcc/11.1.0
module load boost/1.77.0
module load intel-tbb/2020.3
module load cuda/11.4.2
cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
make -j4
