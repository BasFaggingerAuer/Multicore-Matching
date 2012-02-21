#!/bin/sh

cat results/graphs.txt | sort -n | cut -f 2 | xargs -n 1 build/bin/match -d 1 -t -m "0 1 2 3 4 5 6 7 8 9 10 11" 
