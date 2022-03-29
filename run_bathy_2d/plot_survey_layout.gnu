set terminal pngcairo size 800,800

set output 'survey.png'
#set output '| display png:-'

set key autotitle columnhead

#set multiplot layout 2,1
set grid back
set xrange [-10000:10000]
set yrange [-10000:10000]
set title "Acquisition geometry"
set xlabel 'X[m]'
set ylabel 'Y[m]'

plot "receivers.txt" using 1:2 with points ps 1 pt 8 title 'Receivers', \
"sources.txt" using 1:2 with points ps 2 pt 7 title 'Transmitters' 

