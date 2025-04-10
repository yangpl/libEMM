#!/usr/bin/bash

n1=81
n2=81
n3=101
d1=100 #100m
d2=100 #100m
d3=50 #100m
nb=18


export OMP_NUM_THREADS=8
time mpirun -np 1 ../fdtd \
       mode=0 \
       fsrc=sources.txt \
       frec=receivers.txt \
       fsrcrec=src_rec_table.txt \
       frho11=rho11 \
       frho22=rho22 \
       frho33=rho33 \
       chsrc=Ex \
       chrec=Ex \
       x1min=-12000 x1max=12000 \
       x2min=-12000 x2max=12000 \
       x3min=0 x3max=5000 \
       n1=$n1 n2=$n2 n3=$n3 \
       d1=$d1 d2=$d2 d3=$d3 \
       nb=$nb \
       freqs=0.25,0.75,1.25 \
       rd1=3 \
       rd2=3 \
       rd3=3 \
       nugrid=1 \
       fx1nu=x1nu \
       fx2nu=x2nu \
       fx3nu=x3nu \

