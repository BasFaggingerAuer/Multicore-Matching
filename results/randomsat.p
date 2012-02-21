set autoscale
unset log
unset label
unset grid
set xtic nomirror auto
set ytic nomirror auto
set title "Saturation of matching size"
set xlabel "Number of iterations"
set ylabel "Matched vertices/total nr. of vertices (%)"
set xr [0:24]
set yr [0:100]
set grid noytics noxtics
set key right bottom
plot "ecology2.mtx.rdat" using 1:((100*$3)/999999) title 'ecology2 (1,997,996)' with linespoints, \
"ecology1.mtx.rdat" using 1:((100*$3)/1000000) title 'ecology1 (1,998,000)' with linespoints, \
"G3_circuit.mtx.rdat" using 1:((100*$3)/1585478) title 'G3\_circuit (3,037,674)' with linespoints, \
"thermal2.mtx.rdat" using 1:((100*$3)/1228045) title 'thermal2 (3,676,134)' with linespoints, \
"kkt_power.mtx.rdat" using 1:((100*$3)/2063494) title 'kkt\_power (6,482,320)' with linespoints, \
"af_shell9.mtx.rdat" using 1:((100*$3)/504855) title 'af\_shell9 (8,542,010)' with linespoints, \
"ldoor.mtx.rdat" using 1:((100*$3)/952203) title 'ldoor (22,785,136)' with linespoints, \
"af_shell10.mtx.rdat" using 1:((100*$3)/1508065) title 'af\_shell10 (25,582,130)' with linespoints, \
"audikw1.mtx.rdat" using 1:((100*$3)/943695) title 'audikw1 (38,354,076)' with linespoints, \
"nlpkkt120.mtx.rdat" using 1:((100*$3)/3456000) title 'nlpkkt120 (46,651,696)' with linespoints, \
"cage15.mtx.rdat" using 1:((100*$3)/5154859) title 'cage15 (47,022,346)' with linespoints
