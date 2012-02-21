set autoscale
unset label
unset grid
unset log x
unset log y
unset y2label
set xtic nomirror auto
set ytic auto
unset y2tic
set title "Influence of relative blue/red group size"
set xlabel "Fraction of vertices that are blue (%)"
set ylabel "Fraction of matched vertices (%)"
set yr [0:100]
set key center top
plot "roundmatch.dat" using (100*$1):(100*$2) title 'Observed' with linespoints, \
"roundmatch.dat" using (100*$1):(200*(1 - $1)*(1 - exp(-$1/(1.01 - $1)))) title 'Equation (2)' with linespoints
