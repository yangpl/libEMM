import matplotlib.pyplot as plt
import numpy as np

filename ="emf"
f = open(filename, mode="rb")
dat = np.fromfile(f, dtype=np.float32)
cdat = dat[0::2]+ dat[1::2]*1j
emf = np.reshape(cdat,(3,201), order="C") #3 columns (1st dim), 201 rows (2nd dim)


x= np.loadtxt('receivers.txt', skiprows=1, unpack=True) #skip 0 lines
offset = x[0,:]

#-----------------------------------------------
emf= np.loadtxt('emf_0001.txt', skiprows=1, unpack=True) #skip 1 line
iTx = emf[0,:]
iRx = emf[1,:]
ichsrc = emf[2,:]
ichrec = emf[3,:]
ifreq = emf[4,:]
emf_real = emf[5,:]
emf_imag = emf[6,:]
dat = emf_real + emf_imag*1j

nsrc = int(np.max(iTx))
nrec = int(np.max(iRx))
nfreq = int(np.max(ifreq))
print("nsrc=%d"%nsrc)
print("nrec=%d"%nrec)
print("nfreq=%d"%nfreq)


plt.figure(figsize=(12,4))

plt.subplot(121)
myfreq = 0
idx = (ifreq==myfreq+1) & (ichrec==1)
idx0 = range(myfreq*nrec,(myfreq+1)*nrec)
ratio = np.abs(dat[idx])/np.abs(cdat[idx0])-1
plt.plot(offset, ratio, 'r', label='Ex-0.25 Hz')
myfreq = 1
idx = (ifreq==myfreq+1) & (ichrec==1)
idx0 = range(myfreq*nrec,(myfreq+1)*nrec)
ratio = np.abs(dat[idx])/np.abs(cdat[idx0])-1
plt.plot(offset, ratio, 'g', label='Ex-0.75 Hz')
myfreq = 2
idx = (ifreq==myfreq+1) & (ichrec==1)
idx0 = range(myfreq*nrec,(myfreq+1)*nrec)
ratio = np.abs(dat[idx])/np.abs(cdat[idx0])-1
plt.plot(offset, ratio, 'b', label='Ex-1.25 Hz')

plt.xlabel('Offset (m)')
plt.ylabel('$|E_x^{FD}|/|E_x^{ref}|-1$')
plt.ylim([-0.1,0.1])
plt.title('(a) Amplitude difference')
plt.legend()


plt.subplot(122)
myfreq = 0
idx = (ifreq==myfreq+1) & (ichrec==1)
idx0 = range(myfreq*nrec,(myfreq+1)*nrec)
phaerr = np.angle(dat[idx]/cdat[idx0], deg=True)
plt.plot(offset, phaerr, 'r', label='Ex-0.25 Hz')
myfreq = 1
idx = (ifreq==myfreq+1) & (ichrec==1)
idx0 = range(myfreq*nrec,(myfreq+1)*nrec)
phaerr = np.angle(dat[idx]/cdat[idx0], deg=True)
plt.plot(offset, phaerr, 'g', label='Ex-0.75 Hz')
myfreq = 2
idx = (ifreq==myfreq+1) & (ichrec==1)
idx0 = range(myfreq*nrec,(myfreq+1)*nrec)
phaerr = np.angle(dat[idx]/cdat[idx0], deg=True)
plt.plot(offset, phaerr, 'b', label='Ex-1.25 Hz')

plt.ylim([-5,5])
plt.xlabel('Offset (m)')
plt.ylabel(r'$\angle E_x^{FD}-\angle E_x^{ref} $ [$^o$]')
plt.title('(b) Phase difference')
plt.legend()

plt.tight_layout()
plt.savefig('comparison.png')
plt.show()



