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
set title "Matching weight for weighted parallel matching (vs. Alg. 2)"
set xlabel "Number of graph edges"
set ylabel "Matching weight rel. to Alg. 2 (%)"
set yr [0:250]
set key left top
plot "weighted.dat" using 3:((100*$39)/$19):((100*$40)/$19) title 'CUDA' with yerrorbars, \
"weighted.dat" using 3:((100*$49)/$19):((100*$50)/$19) title 'TBB' with yerrorbars
