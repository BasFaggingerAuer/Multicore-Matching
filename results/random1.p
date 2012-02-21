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
set grid ytics noxtics
set title "Matching size for random parallel matching (vs. Alg. 1)"
set xlabel "Number of graph edges"
set ylabel "Matching size rel. to Alg. 1 (%)"
set yr [80:120]
set key top left
plot "random.dat" using 3:((100*$27)/$7):((100*$28)/$7) title 'CUDA' with yerrorbars, \
"random.dat" using 3:((100*$37)/$7):((100*$38)/$7) title 'TBB' with yerrorbars
