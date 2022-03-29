#this requires installing pyvtk via: pip install pyvtk
import numpy as np
import sys
sys.path = ['..']+sys.path

from pyvtk import *


f = open('x1nu', mode="rb")
x1 = np.fromfile(f, dtype=np.float32)

f = open('x2nu', mode="rb")
x2 = np.fromfile(f, dtype=np.float32)

f = open('x3nu', mode="rb")
x3 = np.fromfile(f, dtype=np.float32)


f = open('rho11', mode="rb")
dat = np.fromfile(f, dtype=np.float32)

n1 = x1.size
n2 = x2.size
n3 = x3.size

print("n1=%d"%n1)
print("n2=%d"%n2)
print("n3=%d"%n3)

vtk = VtkData(RectilinearGrid(x1,x2,-x3),
              PointData(Scalars(dat)))

vtk.tofile('rho11_rectilinear')

