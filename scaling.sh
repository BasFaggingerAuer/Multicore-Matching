#!/bin/sh
#
# Script which measures performance scaling as the number of threads increases.
#
# Usage: ./scaling.sh /dimacs/data/matrix
#
device=1;
repeats=4;

rm -f results/*.sdat;

for f in `find $1 -type f -printf '%s %p\n' | sort -n | cut --delimiter=' ' -f 2 | grep .mtx.bz2`
do
	fshort=`echo $f | sed 's/.*\///g' | sed 's/\.bz2//g'`;
	echo $f $fshort;
	build/bin/match -r -a $repeats -d $device --scaledata --gnuplot results/$fshort.sdat --threads "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16" -m "10" $f;
done

