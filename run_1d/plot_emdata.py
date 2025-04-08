import matplotlib.pyplot as plt
import numpy as np


x, y, z, azimuth, dip, iRx= np.loadtxt('receivers.txt', skiprows=1, unpack=True) #skip 0 lines

# filename ="emf"
# f = open(filename, mode="rb")
# dat = np.fromfile(f, dtype=np.float32)
# cdat = dat[0::2]+ dat[1::2]*1j

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

nsrc = int(np.max(iTx))
nrec = int(np.max(iRx))
nfreq = int(np.max(ifreq))
print("nsrc=%d"%nsrc)
print("nrec=%d"%nrec)
print("nfreq=%d"%nfreq)

#----------------------------------------------------------
ref = np.loadtxt('emf_ref.txt', skiprows=1, unpack=True) #skip 0 lines
cdat = ref[0,:] + ref[1,:]*1j

#----------------------------------------------------------
plt.figure(figsize=(12,9))
plt.subplot(221)
idx = range(nrec)
plt.plot(x, np.abs(dat[idx]), 'r', label='FDTD Ex-0.25 Hz')
plt.plot(x, np.abs(cdat[idx]), 'r--', label='Analytic Ex-0.25 Hz')
idx = range(nrec,2*nrec)
plt.plot(x, np.abs(dat[idx]), 'g', label='FDTD Ex-0.75 Hz')
plt.plot(x, np.abs(cdat[idx]), 'g--', label='Analytic Ex-0.75 Hz')
idx = range(2*nrec,3*nrec)
plt.plot(x, np.abs(dat[idx]), 'b', label='FDTD Ex-1.25 Hz')
plt.plot(x, np.abs(cdat[idx]), 'b--', label='Analytic Ex-1.25 Hz')

plt.yscale('log')
plt.ylabel('Amplitude [V/Am$^2$]')
plt.title('(a) Amplitude')
plt.legend()


plt.subplot(222)
idx = range(nrec)
plt.plot(x, np.angle(dat[idx],deg=True), 'r', label='FDTD Ex-0.25 Hz')
plt.plot(x, np.angle(cdat[idx],deg=True), 'r--', label='Analytic Ex-0.25 Hz')
idx = range(nrec,2*nrec)
plt.plot(x, np.angle(dat[idx],deg=True), 'g', label='FDTD Ex-0.75 Hz')
plt.plot(x, np.angle(cdat[idx],deg=True), 'g--', label='Analytic Ex-0.75 Hz')
idx = range(2*nrec,3*nrec)
plt.plot(x, np.angle(dat[idx],deg=True), 'b', label='FDTD Ex-1.25 Hz')
plt.plot(x, np.angle(cdat[idx],deg=True), 'b--', label='Analytic Ex-1.25 Hz')

plt.ylabel('Degree [$^o$]')
plt.title('(b) Phase')
plt.legend()


plt.subplot(223)
idx = range(nrec)
ratio = np.abs(dat[idx]/cdat[idx])-1
plt.plot(x, ratio, 'r', label='Ex-0.25 Hz')
idx = range(nrec,2*nrec)
ratio = np.abs(dat[idx]/cdat[idx])-1
plt.plot(x, ratio, 'g', label='Ex-0.75 Hz')
idx = range(2*nrec,3*nrec)
ratio = np.abs(dat[idx]/cdat[idx])-1
plt.plot(x, ratio, 'b', label='Ex-1.25 Hz')


plt.xlabel('X [m]')
plt.ylabel('$|E_x|_{FD}/|E_x|_{Analytic}-1$')
plt.ylim([-0.03,0.03])
plt.title('(c) Amplitude difference')
plt.legend()



plt.subplot(224)
idx = range(nrec)
phaerr = np.angle(dat[idx]/cdat[idx], deg=True) #phase
plt.plot(x, phaerr, 'r', label='Ex-0.25 Hz')
idx = range(nrec,2*nrec)
phaerr = np.angle(dat[idx]/cdat[idx], deg=True) #phase
plt.plot(x, phaerr, 'g', label='Ex-0.75 Hz')
idx = range(2*nrec,3*nrec)
phaerr = np.angle(dat[idx]/cdat[idx], deg=True) #phase
plt.plot(x, phaerr, 'b', label='Ex-1.25 Hz')


plt.xlabel('X [m]')
plt.ylabel('Degree [$^o$]')
plt.ylim([-5,5])
plt.title('(d) Phase difference')
plt.legend()

plt.tight_layout(pad=2)
plt.savefig('comparison.png')
plt.show()



