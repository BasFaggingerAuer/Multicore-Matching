#!/bin/sh
#
# Script which generates the saturation as a function of the number of matching iterations.
#
# Usage: ./saturate.sh /dimacs/data/matrix
#
# It is necessary for the software to be compiled with the MATCH_INTERMEDIATE_COUNT define for this to work.
#
device=1;

rm -f results/*.rdat;
rm -f results/*.wdat;

for f in `find $1 -type f -printf '%s %p\n' | sort -n | cut --delimiter=' ' -f 2 | grep .mtx.bz2`
do
	fshort=`echo $f | sed 's/.*\///g' | sed 's/\.bz2//g'`;
	echo $f $fshort;
	build/bin/match -r -d $device -m "6" $f > results/$fshort.rdat;
	build/bin/match -r -d $device -m "8" $f > results/$fshort.wdat;
done

