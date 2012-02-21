set autoscale
unset label
unset grid
set log x
set format x "10^{%L}"
unset log y
unset y2label
unset y2tic
set xtic nomirror auto
set ytic mirror auto
set title "Speedup for weighted parallel matching (vs. Alg. 1)"
set xlabel "Number of graph edges"
set ylabel "Speedup rel. to Alg. 1"
set key top left
plot "weighted.dat" using 3:($13/$43):($14/$43) title 'CUDA' with yerrorbars, \
"weighted.dat" using 3:($13/$53):($14/$53) title 'TBB' with yerrorbars
