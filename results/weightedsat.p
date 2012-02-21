set autoscale
unset log
unset label
unset grid
set xtic nomirror auto
set ytic nomirror auto
set title "Saturation of matching weight"
set xlabel "Number of iterations"
set ylabel "Matching weight/attained weight (%)"
set xr [0:24]
set yr [0:100]
set key right bottom
plot "ecology2.mtx.wdat" using 1:((100*$2)/1962826) title 'ecology2' with linespoints, \
"ecology1.mtx.wdat" using 1:((100*$2)/1962828) title 'ecology1' with linespoints, \
"G3_circuit.mtx.wdat" using 1:((100*$2)/6260805) title 'G3_circuit' with linespoints, \
"af_shell9.mtx.wdat" using 1:((100*$2)/110019673) title 'af_shell9' with linespoints, \
"thermal2.mtx.wdat" using 1:((100*$2)/922792) title 'thermal2' with linespoints, \
"kkt_power.mtx.wdat" using 1:((100*$2)/2881307789) title 'kkt_power' with linespoints, \
"nlpkkt120.mtx.wdat" using 1:((100*$2)/25920000) title 'nlpkkt120' with linespoints, \
"af_shell10.mtx.wdat" using 1:((100*$3)/1508064) title 'af_shell10' with linespoints, \
"ldoor.mtx.wdat" using 1:((100*$2)/232087380) title 'ldoor' with linespoints, \
"cage15.mtx.wdat" using 1:((100*$3)/5154858) title 'cage15' with linespoints, \
"audikw1.mtx.wdat" using 1:((100*$2)/324437250) title 'audikw1' with linespoints
