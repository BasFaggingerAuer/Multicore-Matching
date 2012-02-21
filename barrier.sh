#!/bin/sh
#
# Script which records the time, size, and weight of a matching as a function of the relative size of the group of blue vertices.
#
# Usage: ./barrier.sh foo.mtx
#
barrierfile="results/barrier.dat";
device=1;
repeats=32;

rm -f $barrierfile

for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
do
	build/bin/match -r -a $repeats -d $device -b $i -m "4 6 8" --gnuplot $barrierfile $1;
done

