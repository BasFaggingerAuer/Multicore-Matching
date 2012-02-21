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
set title "Speedup for weighted parallel matching (vs. Alg. 2)"
set xlabel "Number of graph edges"
set ylabel "Speedup rel. to Alg. 2"
set key top left
plot "weighted.dat" using 3:($23/$43):($24/$43) title 'CUDA' with yerrorbars, \
"weighted.dat" using 3:($23/$53):($24/$53) title 'TBB' with yerrorbars
