#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import re
import glob
import sys
import os

#----------------------------------------------------
mu0 = 4.*np.pi*1e-7


#-----------------------------------------------
iTx = []
iRx = []
ichrec = []
ifreq = []
dat = []
f =open('broadside_emf_0001.txt', 'r')
header = f.readline()
for line in f:
    line = line.strip()
    columns = line.split()
    iTx.append(int(columns[0]))
    iRx.append(int(columns[1]))
    ichrec.append(columns[2])
    ifreq.append(int(columns[3]))
    dat.append(float(columns[4]) +float(columns[5])*1j)
iTx = np.array(iTx)
iRx = np.array(iRx)
ichrec = np.array(ichrec)
ifreq = np.array(ifreq)
dat = np.array(dat)

amp = np.abs(dat) #nsrc*nfreq*nrec
pha = np.angle(dat, deg=True) #phase


plt.figure()
plt.subplot(211)
idx = (ifreq==1) & (ichrec=='Ex')
plt.plot(iRx[idx], amp[idx], 'r', label='FDTD Ex-0.25 Hz')
idx = (ifreq==2) & (ichrec=='Ex')  
plt.plot(iRx[idx], amp[idx], 'g', label='FDTD Ex-0.75 Hz')
idx = (ifreq==3) & (ichrec=='Ex')  
plt.plot(iRx[idx], amp[idx], 'b', label='FDTD Ex-1.25 Hz')


plt.yscale('log')
plt.ylabel('Amplitude [V/Am$^2$]')
plt.xlabel('# iRx')
plt.title('(a) Amplitude')
plt.legend()


plt.subplot(212)
idx = (ifreq==1) & (ichrec=='Ex')
plt.plot(iRx[idx], pha[idx], 'r', label='FDTD Ex-0.25 Hz')
idx = (ifreq==2) & (ichrec=='Ex')
plt.plot(iRx[idx], pha[idx], 'g', label='FDTD Ex-0.75 Hz')
idx = (ifreq==3) & (ichrec=='Ex')
plt.plot(iRx[idx], pha[idx], 'b', label='FDTD Ex-1.25 Hz')

plt.ylabel('Degree [$^o$]')
plt.xlabel('# iRx')
plt.title('(b) Phase')
plt.legend()


plt.tight_layout(pad=1)
plt.savefig('broadside_amp_phase.png')
plt.show()

