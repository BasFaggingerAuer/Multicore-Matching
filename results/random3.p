set autoscale
unset label
unset grid
set log x
unset log y
unset y2label
unset y2tic
set xtic nomirror auto
set ytic mirror auto
set title "Speedup for random parallel matching (with atomics)"
set xlabel "Number of graph edges"
set ylabel "GPU/CPU speedup"
set yr [0:3.5]
unset key
plot "random.dat" using 3:($10/$28):($11/$28) title 'atomics' with yerrorbars
