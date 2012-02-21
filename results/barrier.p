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
set ylabel "Fraction of maximum value (%)"
set yr [0:100]
set key center
plot "barrier.dat" using (100*$12):((100*$24)/4.42e+06) title 'Matching weight' with linespoints, \
"barrier.dat" using (100*$12):((100*$13)/913231) title 'Matching size' with linespoints, \
"barrier.dat" using (100*$12):((100*$19)/106.3) title 'Matching time' with linespoints
