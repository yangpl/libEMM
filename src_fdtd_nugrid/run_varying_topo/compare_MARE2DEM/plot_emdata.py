#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import re
import glob
import sys
import os

mu0 = 4.*np.pi*1e-7

tmpfile = 'output.0.resp'
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


filename ="emf_0000"
f = open(filename, mode="rb")
dat = np.fromfile(f, dtype=np.float32)

cdat = dat[0::2]+ dat[1::2]*1j

emffd = np.reshape(cdat,(641,3), order="F") #index[i1,i2,i3]=(fastest,faster,slow)
amp = np.abs(emffd)
pha = np.angle(emffd, deg=True) #phase
offset = np.linspace(-8000.,8000.,num=641)

plt.figure(figsize=(16,10))
plt.subplot(221)

plt.plot(offset, amp[:,0],'k', label='FDTD 0.25 Hz')
plt.plot(offset, amp[:,1],'g', label='FDTD 0.75 Hz')
plt.plot(offset, amp[:,2],'m', label='FDTD 1.25 Hz')

plt.plot(y_Rx, emf[:, 0, 0, 0], 'k--', label='MARE2DEM 0.25 Hz')
plt.plot(y_Rx, emf[:, 0, 1, 0], 'g--', label='MARE2DEM 0.75 Hz')
plt.plot(y_Rx, emf[:, 0, 2, 0], 'm--', label='MARE2DEM 1.25 Hz')


plt.legend()
plt.yscale('log')
plt.xlabel('offset [m]')
plt.ylabel('Amplitude [V/Am^2]')
plt.title('(a) Amplitude')
plt.grid(color='r', linestyle='-')


plt.subplot(222)

plt.plot(offset, pha[:,0],'k', label='FDTD 0.25 Hz')
plt.plot(offset, pha[:,1],'g', label='FDTD 0.75 Hz')
plt.plot(offset, pha[:,2],'m', label='FDTD 1.25 Hz')

plt.plot(y_Rx, emf[:, 0, 0, 1],'k--', label='MARE2DEM 0.25 Hz')
plt.plot(y_Rx, emf[:, 0, 1, 1],'g--', label='MARE2DEM 0.75 Hz')
plt.plot(y_Rx, emf[:, 0, 2, 1],'m--', label='MARE2DEM 1.25 Hz')


plt.legend()
plt.xlabel('Offset [m]')
plt.ylabel('Degree')
plt.title('(b) Phase')
plt.grid(color='r', linestyle='-')



plt.subplot(223)
ratio1 = emf[:, 0, 0, 0]/amp[:,0]-1
ratio2 = emf[:, 0, 1, 0]/amp[:,1]-1
ratio3 = emf[:, 0, 2, 0]/amp[:,2]-1
plt.plot(offset, ratio1, 'k', label='0.25 Hz')
plt.plot(offset, ratio2, 'g', label='0.75 Hz')
plt.plot(offset, ratio3, 'm', label='1.25 Hz')
plt.ylim([-0.05,0.05])

plt.legend()
plt.grid(color='r', linestyle='-')
plt.title('(c) Amplitude error')

plt.subplot(224)
phaerr1 = pha[:,0] - emf[:, 0, 0, 1]
phaerr2 = pha[:,1] - emf[:, 0, 1, 1]
phaerr3 = pha[:,2] - emf[:, 0, 2, 1]

plt.plot(offset,phaerr1,'k', label=' 0.25 Hz')
plt.plot(offset,phaerr2,'g',label=' 0.75 Hz')
plt.plot(offset,phaerr3,'m', label=' 1.25 Hz')

plt.legend()
plt.xlabel('Offset [m]')
plt.ylabel('Degree')
plt.title('(d) Phase error')
plt.ylim([-2,2])
plt.grid(color='r', linestyle='-')

plt.savefig('bathy_comparison.png')
plt.show()
















