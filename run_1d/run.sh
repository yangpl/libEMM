#!/usr/bin/bash

n1=101
n2=101
n3=101
d1=200 #100m
d2=200 #100m
d3=50 #100m


export OMP_NUM_THREADS=8
#time mpirun -n 1 ../bin/fdtd \
#nvprof --log-file profiling.txt
../bin/fdtd \
       mode=0 \
       fsrc=sources.txt \
       frec=receivers.txt \
       fsrcrec=src_rec_table.txt \
       frho11=rho11 \
       frho22=rho22 \
       frho33=rho33 \
       chsrc=Ex \
       chrec=Ex \
       x1min=-10000 x1max=10000 \
       x2min=-10000 x2max=10000 \
       x3min=0 x3max=5000 \
       n1=$n1 n2=$n2 n3=$n3 \
       d1=$d1 d2=$d2 d3=$d3 \
       nb=12 ne=6 \
       freqs=0.25,0.75,1.25 \
       rd=2 

       
