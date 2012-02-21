set autoscale
set log y
set log x
unset label
unset grid
set xtic nomirror auto
set ytic nomirror auto
set title "Matching time scaling"
set xlabel "Number of CPU threads"
set ylabel "Relative matching time (%)"
set key right top
set xtics 2
set xr [1:16]
set ytics (10,20,30,40,50,60,70,80,90,100)
set grid ytics noxtics
set yr [10:100]
plot "ecology2.mtx.sdat" using 1:(100*$8/496) title 'ecology2 (1,997,996)' with linespoints, \
"ecology1.mtx.sdat" using 1:(100*$8/497) title 'ecology1 (1,998,000)' with linespoints, \
"G3_circuit.mtx.sdat" using 1:(100*$8/823) title 'G3\_circuit (3,037,674)' with linespoints, \
"thermal2.mtx.sdat" using 1:(100*$8/738) title 'thermal2 (3,676,134)' with linespoints, \
"kkt_power.mtx.sdat" using 1:(100*$8/1546) title 'kkt\_power (6,482,320)' with linespoints, \
"af_shell9.mtx.sdat" using 1:(100*$8/670) title 'af\_shell9 (8,542,010)' with linespoints, \
"ldoor.mtx.sdat" using 1:(100*$8/3599) title 'ldoor (22,785,136)' with linespoints, \
"af_shell10.mtx.sdat" using 1:(100*$8/2609) title 'af\_shell10 (25,582,130)' with linespoints, \
"audikw1.mtx.sdat" using 1:(100*$8/4754) title 'audikw1 (38,354,076)' with linespoints, \
"nlpkkt120.mtx.sdat" using 1:(100*$8/3620) title 'nlpkkt120 (46,651,696)' with linespoints, \
"cage15.mtx.sdat" using 1:(100*$8/5357) title 'cage15 (47,022,346)' with linespoints, \
"ecology1.mtx.sdat" using 1:(100/$1) title 'ideal scaling' with lines
