#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import re
import glob
import sys
import os

#----------------------------------------------------
mu0 = 4.*np.pi*1e-7

tmpfile = 'mare2dem.txt'
freq = []
TxNav = np.zeros((0,5))
RxNav = np.zeros((0,5))
Txids = []
Rxids = []

f = open(tmpfile,'r')
for line in f:
    line = line.lower()

    if line.find(':')!=-1:
        head = line.split(':')[0]
        sValue = line.split(':')[1]
        print(head + ': ' + sValue)
        
        if head == 'format':
            file_format = sValue.strip()
        elif head == 'utm grid, n, e of origin':
            pass
        elif head == 'lat & long of origin':
            pass
        elif head == 'strike of x':
            pass
        elif head == 'phase convention':
            pass
        elif head == '# csem frequencies':
            #------------------------------------------------
            if len(freq) == 0: #have not record any freq yet
                nfreq = int(sValue)
                ifreq = 0
                while ifreq < nfreq:
                    nextline = next(f)
                    freq.append(float(nextline))
                    print('frequency[%d]: %f Hz'%(ifreq, freq[ifreq]))
                    ifreq += 1
        elif head == '# transmitters': 
            nTx = int(sValue)
            y_Tx = np.zeros((nTx, 1))
            z_Tx = np.zeros((nTx, 1))

            iTx = 0
            while iTx < nTx:
                nextline = next(f)
                slist = re.findall(r"\S+", nextline)                            

                if(slist[0]!='!' and slist[0]!='#'):
                    print(slist[8])
                    Txids.append(slist[8])
                    x = float(slist[0])
                    y = float(slist[1])
                    z = float(slist[2])
                    hd = float(slist[3])
                    pitch = float(slist[4])
                    TxNav = np.vstack( (TxNav, [x,y,z,hd,pitch]))

                    y_Tx[iTx] = float(slist[1])
                    z_Tx[iTx] = float(slist[2])
                    iTx += 1
                    
        elif head == '# csem receivers': 
            nRx = int(sValue)
            y_Rx = np.zeros((nRx, 1))
            z_Rx = np.zeros((nRx, 1))
            iRx = 0
            while iRx < nRx:
                nextline = next(f)
                slist = re.findall(r"\S+", nextline)                            

                if(slist[0]!='!' and slist[0]!='#'):
                    print(slist[8])
                    Rxids.append(slist[8])
                    x = float(slist[0])
                    y = float(slist[1])
                    z = float(slist[2])
                    hd = float(slist[3])
                    pitch = float(slist[4])
                    RxNav = np.vstack( (RxNav, [x,y,z,hd,pitch]))

                    y_Rx[iRx] = float(slist[1])
                    z_Rx[iRx] = float(slist[2])
                    iRx += 1
                    
        elif head == '# data' or head == '#data':
            ndp = int(sValue)
            #print(nRx, nTx, nfreq)
            emf = np.zeros((nRx, nTx, nfreq, 2))

            ndp /= 2
            idp = 0
            while idp < ndp:
                nextline = next(f)
                slist = re.findall(r"\S+", nextline)                            

                if(slist[0]=='23'):
                    ifreq = int(slist[1])-1
                    iTx = int(slist[2])-1
                    iRx = int(slist[3])-1
                    #print(ifreq, iTx, iRx)
                    emf[iRx, iTx, ifreq, 0] = float(slist[6])
                if(slist[0]=='24'):
                    ifreq = int(slist[1])-1
                    iTx = int(slist[2])-1
                    iRx = int(slist[3])-1
                    emf[iRx, iTx, ifreq, 1] = float(slist[6])
                    idp += 1
f.close()


x= np.loadtxt('receivers.txt', skiprows=1, unpack=True) #skip 0 lines
offset = x[0,:]





#-----------------------------------------------
iTx = []
iRx = []
ichrec = []
ifreq = []
dat = []
f =open('emf_0001.txt', 'r')
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


nsrc = int(np.max(iTx))
nrec = int(np.max(iRx))
nfreq = int(np.max(ifreq))
print("nsrc=%d"%nsrc)
print("nrec=%d"%nrec)
print("nfreq=%d"%nfreq)



plt.figure(figsize=(10,8))
plt.subplot(221)
idx = (ifreq==1) & (ichrec=='Ex')
plt.plot(offset, amp[idx], 'r', label='FDTD Ex-0.25 Hz')
idx = (ifreq==2) & (ichrec=='Ex')  
plt.plot(offset, amp[idx], 'g', label='FDTD Ex-0.75 Hz')
idx = (ifreq==3) & (ichrec=='Ex')  
plt.plot(offset, amp[idx], 'b', label='FDTD Ex-1.25 Hz')

plt.plot(y_Rx, emf[:, 0, 0, 0], 'k--', label='MARE2DEM 0.25 Hz')
plt.plot(y_Rx, emf[:, 0, 1, 0], 'g--', label='MARE2DEM 0.75 Hz')
plt.plot(y_Rx, emf[:, 0, 2, 0], 'm--', label='MARE2DEM 1.25 Hz')


plt.yscale('log')
plt.ylabel('Amplitude [V/Am$^2$]')
plt.title('(a) Amplitude')
plt.legend()


plt.subplot(222)
idx = (ifreq==1) & (ichrec=='Ex')
plt.plot(offset, pha[idx], 'r', label='FDTD Ex-0.25 Hz')
idx = (ifreq==2) & (ichrec=='Ex')
plt.plot(offset, pha[idx], 'g', label='FDTD Ex-0.75 Hz')
idx = (ifreq==3) & (ichrec=='Ex')
plt.plot(offset, pha[idx], 'b', label='FDTD Ex-1.25 Hz')

plt.plot(y_Rx, emf[:, 0, 0, 1],'k--', label='MARE2DEM 0.25 Hz')
plt.plot(y_Rx, emf[:, 0, 1, 1],'g--', label='MARE2DEM 0.75 Hz')
plt.plot(y_Rx, emf[:, 0, 2, 1],'m--', label='MARE2DEM 1.25 Hz')

plt.ylabel('Degree [$^o$]')
plt.title('(b) Phase')
plt.legend()


plt.subplot(223)
idx = (ifreq==1) & (ichrec=='Ex')
ratio = amp[idx]/emf[:, 0, 0, 0]-1
#ratio[0]=ratio[1]
plt.plot(offset, ratio, 'r', label='Ex-0.25 Hz')
idx = (ifreq==2) & (ichrec=='Ex')
ratio = amp[idx]/emf[:, 0, 1, 0]-1
#ratio[0]=ratio[1]
plt.plot(offset, ratio, 'g', label='Ex-0.75 Hz')
idx = (ifreq==3) & (ichrec=='Ex')
ratio = amp[idx]/emf[:, 0, 2, 0]-1
#ratio[0]=ratio[1]
plt.plot(offset, ratio, 'b', label='Ex-1.25 Hz')

plt.xlabel('Offset [m]')
plt.ylabel('$|E_x|_{FD}/|E_x|_{ref}-1$')
plt.ylim([-0.1,0.1])
plt.title('(c) Amplitude difference')
plt.legend()


plt.subplot(224)
idx = (ifreq==1) & (ichrec=='Ex')
phaerr = pha[idx]-emf[:, 0, 0, 1]
plt.plot(offset, phaerr, 'r', label='Ex-0.25 Hz')
idx = (ifreq==2) & (ichrec=='Ex')
phaerr = pha[idx]-emf[:, 0, 1, 1]
plt.plot(offset, phaerr, 'g', label='Ex-0.75 Hz')
idx = (ifreq==3) & (ichrec=='Ex')
phaerr = pha[idx]-emf[:, 0, 2, 1]
plt.plot(offset, phaerr, 'b', label='Ex-1.25 Hz')

plt.xlabel('Offset [m]')
plt.ylabel('Degree [$^o$]')
plt.ylim([-5,5])
plt.title('(d) Phase difference')
plt.legend()

plt.tight_layout(pad=0)
plt.savefig('bathy_comparison.png')
plt.show()

