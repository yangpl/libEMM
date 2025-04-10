import empymod
import matplotlib.pyplot as plt
import numpy as np

x, y, z, azimuth, tilt, irec= np.loadtxt('receivers.txt', skiprows=1, unpack=True) #skip 0 lines

iTx,iRx,ichsrc,ichrec,ifreq,emf_real, emf_imag= np.loadtxt('emf_0001.txt', skiprows=1, unpack=True) #skip 1 line
dat = emf_real + emf_imag*1j
amp = np.abs(dat) #nsrc*nfreq*nrec
pha = np.angle(dat, deg=True) #phase

ref1d = empymod.dipole(src=[0, 0, 200],
                       rec=[x, x*0, 250],
                       depth=[0, 275, 1025, 1525],
                       freqtime=[0.25,0.75,1.25],
                       res=[1e8, 0.3, 1.0, 2.0, 4.],
                       verb=1)
ref1d = np.conj(ref1d)
amp0 = np.abs(ref1d)
pha0 = np.angle(ref1d, deg=True) #phase

plt.figure(figsize=(12,8))
plt.subplot(221)
idx = (ifreq==1) & (ichrec==1)
plt.plot(x, amp[idx], 'r', label='$E_x^{FD}$-0.25 Hz')
idx = (ifreq==2) & (ichrec==1)
plt.plot(x, amp[idx], 'g', label='$E_x^{FD}$-0.75 Hz')
idx = (ifreq==3) & (ichrec==1)
plt.plot(x, amp[idx], 'b', label='$E_x^{FD}$-1.25 Hz')


plt.plot(x, amp0[0,:], 'r--', label='$E_x^{ref}$-0.25 Hz')
plt.plot(x, amp0[1,:], 'g--', label='$E_x^{ref}$-0.75 Hz')
plt.plot(x, amp0[2,:], 'b--', label='$E_x^{ref}$-1.25 Hz')

plt.xlabel('Offset (m)')
plt.yscale('log')
plt.ylabel('Amplitude (V/Am$^2$)')
plt.grid(color='k', linestyle='-')
plt.legend()
plt.title('(a)')

plt.subplot(222)
idx = (ifreq==1) & (ichrec==1)
plt.plot(x, pha[idx], 'r', label='$E_x^{FD}$-0.25 Hz')
idx = (ifreq==2) & (ichrec==1)
plt.plot(x, pha[idx], 'g', label='$E_x^{FD}$-0.75 Hz')
idx = (ifreq==3) & (ichrec==1)
plt.plot(x, pha[idx], 'b', label='$E_x^{FD}$-1.25 Hz')


plt.plot(x, pha0[0,:], 'r--', label='$E_x^{ref}$-0.25 Hz')
plt.plot(x, pha0[1,:], 'g--', label='$E_x^{ref}$-0.75 Hz')
plt.plot(x, pha0[2,:], 'b--', label='$E_x^{ref}$-1.25 Hz')

plt.xlabel('Offset (m)')
plt.ylabel('Degree ($^o$)')
plt.grid(color='k', linestyle='-')
plt.legend()
plt.title('(b)')


plt.subplot(223)
idx = (ifreq==1) & (ichrec==1)
ratio = amp[idx]/amp0[0,:]-1
plt.plot(x, ratio, 'r', label='Ex-0.25 Hz')
idx = (ifreq==2) & (ichrec==1)
ratio = amp[idx]/amp0[1,:]-1
plt.plot(x, ratio, 'g', label='Ex-0.75 Hz')
idx = (ifreq==3) & (ichrec==1)
ratio = amp[idx]/amp0[2,:]-1
plt.plot(x, ratio, 'b', label='Ex-1.25 Hz')

plt.xlabel('Offset (m)')
plt.ylabel('$|E_x|_{FD}/|E_x|_{Analytic}-1$')
plt.ylim([-0.05,0.05])
plt.grid(color='k', linestyle='-')
plt.legend()
plt.title('(c)')


plt.subplot(224)
idx = (ifreq==1) & (ichrec==1)
phaerr = pha[idx]-pha0[0,:]
plt.plot(x, phaerr, 'r', label='Ex-0.25 Hz')
idx = (ifreq==2) & (ichrec==1)
phaerr = pha[idx]-pha0[1,:]
plt.plot(x, phaerr, 'g', label='Ex-0.75 Hz')
idx = (ifreq==3) & (ichrec==1)
phaerr = pha[idx]-pha0[2,:]
plt.plot(x, phaerr, 'b', label='Ex-1.25 Hz')

plt.xlabel('Offset (m)')
plt.ylabel('Degree [$^o$]')
plt.ylim([-2,2])
plt.grid(color='k', linestyle='-')
plt.legend()
plt.title('(d)')


plt.tight_layout()
plt.savefig('comparison.png')
plt.show()

