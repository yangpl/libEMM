import matplotlib.pyplot as plt
import numpy as np
import cmath
import os

filename ="emf_0000"
f = open(filename, mode="rb")
dat = np.fromfile(f, dtype=np.float32)

#tmp = np.reshape(dat,(-1,2)) #-1 means infer the dimension, 2=ncols
# amp = np.sqrt(tmp[:,0]**2 +tmp[:,1]**2)


cdat = dat[0::2]+ dat[1::2]*1j

nfreq = 3
nrec = cdat.size/nfreq
print('nfreq=%d'%nfreq)
print('nrec=%d'%nrec)

                 
emf = np.reshape(cdat,(641,3), order="F") #index[i1,i2,i3]=(fastest,faster,slow)
amp = np.abs(emf)
pha = np.angle(emf, deg=True) #phase
offset = np.linspace(-8000.,8000.,num=641)


plt.figure()
# for ic in range(nch):
#     for ifreq in range(nfreq):
#         plt.plot(offset,amp[:,ic,ifreq],'k',label='ch-%d freq-%d Hz'%(ic,ifreq))
plt.plot(offset,amp[:,0],'k', label='Ex 0.25 Hz')
plt.plot(offset,amp[:,1],'g',label='Ex 0.75 Hz')
plt.plot(offset,amp[:,2],'m', label='Ex 1.25 Hz')

plt.legend()
plt.yscale('log')
plt.xlabel('offset [m]')
plt.ylabel('Amplitude [V/Am^2]')
plt.title('Amplitude')
plt.grid(color='r', linestyle='-')
plt.savefig('Amplitude.png')
plt.show()

plt.figure()
plt.plot(offset,pha[:,0],'k', label='Ex 0.25 Hz')
plt.plot(offset,pha[:,1],'g',label='Ex 0.75 Hz')
plt.plot(offset,pha[:,2],'m', label='Ex 1.25 Hz')
plt.legend()
plt.xlabel('offset [m]')
plt.ylabel('Degree')
plt.title('Phase')
plt.grid(color='r', linestyle='-')
plt.savefig('Phase.png')
plt.show()


