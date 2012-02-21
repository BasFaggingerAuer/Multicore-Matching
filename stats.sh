#!/bin/sh
#
# Script which generates a list of all graphs and matrices to be matched, together with their number of vertices.
#
# Usage: ./stats.sh /dimacs/data
#

rm -f results/graphs.txt

find $1 -type f -printf '%s\t%p\n' | sort -n | cut -f 2 | grep -e .graph.bz2 -e .mtx.bz2 | xargs -n 1 build/bin/graphstat >> results/graphs.txt
