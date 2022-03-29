# libEMM
A C + CUDA library for 3D CSEM modelling

Author: Pengliang Yang, Harbin Institute of Technology, China

Email: ypl.2100@gmail.com


Programming language: C, CUDA, Fortran, Shell

Operating System: Linux

Software dependencies: MPI [1], FFTW [2], CUDA [3]

Nature of problem: Land/Marine Controlled-source electromagnetics (CSEM)

Solution method: High-order finite-dfference time-domain (FDTD) on staggered non-uniform grid by fictious wave domain transformation

Governing equation: 1st order diffusive Maxwell equation



Acknowledgement: The initiative to start this ficititious wave domain modelling project begins when I was a scientist working in Electromagnetic Geoservices ASA (EMGS). I developed some modeling codes in my free time, but they never work correctly.  After I left EMGS in 2020, I restarted everything from scratch using all things I learned from Madagascar open software development. It took me more than one year to make it work correctly: the solution now matches the semi-analytic one. During the development, I benefit from the discussion with Rune Mittet, in order to understand his method.
