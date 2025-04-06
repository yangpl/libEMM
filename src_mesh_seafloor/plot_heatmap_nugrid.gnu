set terminal pngcairo size 1200,1200
set output 'mesh_resistivity.png'
#set output '| display png:-'

set multiplot layout 2,1


#----------------------------------------------
set yrange [0:5000] reverse
set xlabel 'Distance X[m]'
set ylabel 'Depth Z[m]'
set grid back

set pm3d map corners2color c3
set autoscale fix
splot 'grid_rho11' using 2:1:3 notitle
show grid

#----------------------------------------------
set yrange [5000:0] 
set grid back
set xlabel 'Distance X [m]'
set ylabel 'Depth Z [m]'
#splot 'table_rho11' using 2:1:3 with lines notitle
plot 'grid_rho11' using 2:1 with lines lc 'red' notitle, \
'topo.txt' using 1:2 with lines lc 'black' notitle, \
'acquisition.txt' using 1:3 with points pt 13 notitle, \
"<echo '0 300'" with points pt 2 pointsize 5 notitle
show grid

