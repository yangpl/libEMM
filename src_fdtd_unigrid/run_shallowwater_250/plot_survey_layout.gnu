set terminal pngcairo size 800,800
# set xtics font "Helvetica,15"
# set ytics font "Helvetica,15"
# set key font "Helvetica,15"
# set title font 'Helvetica,20'
set output 'survey.png'
#set output '| display png:-'


#set multiplot layout 2,1
set grid back
# set xrange [-3000:3000]
# set yrange [-3000:3000]
set title "Acquisition geometry"
set xlabel 'X[m]'
set ylabel 'Y[m]'

plot "receivers.txt" using 1:2 with points ps 1 pt 8 title 'Receivers', \
"sources.txt" using 1:2 with points ps 2 pt 7 title 'Transmitters' 

# plot "receivers.txt" using 1:2 with lp title 'Receivers', \
# "sources.txt" using 1:2 with points ps 3 pt 7 title 'Transmitters' 

