#!/bin/sh
#
# Script which generates all benchmark data for random and weighted matching.
# Uses the graphs listed by stats.sh.
# 
# Usage: ./benchmark.sh
#
weightfile="results/weighted.dat";
randomfile="results/random.dat";
device=1;
repeats=32;

rm -f $weightfile
rm -f $randomfile

cat results/graphs.txt | sort -n | cut -f 2 | grep .mtx.bz2 | xargs -n 1 build/bin/match -r -d $device -a $repeats --gnuplot $weightfile -m "4 5 8 9 11"
cat results/graphs.txt | sort -n | cut -f 2 | xargs -n 1 build/bin/match -r -d $device -a $repeats --gnuplot $randomfile -m "0 6 7 10"
