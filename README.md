# libEMM
A C + CUDA library for 3D CSEM modelling

Author: Pengliang Yang, Harbin Institute of Technology, China

Email: ypl.2100@gmail.com


Programming language: C, CUDA, Fortran, Shell

Operating System: Linux

Software dependencies: MPI, FFTW, CUDA (optional)

Nature of problem: Land/Marine Controlled-source electromagnetics (CSEM)

Solution method: High-order finite-dfference time-domain (FDTD) on staggered non-uniform grid by fictious wave domain transformation

Governing equation: 1st order diffusive Maxwell equation

Code structure:
===============

* src: the source code in .c and .cu/.cuh. Code applies uniform gridding along x and y axes, but nonuniform gridding along z axis.

* include: the header files in .h

* doc: the mannual for libEMM

* bin: the folderto store executable after compilation

* run_1d: Application example in 3D model of 1D layered sturture

* run_bathy_2d: Appliation example in 3D model of 2D structure with seafloor bathymetry

* src_fdtd_nugrid: independent folder to run FDTD on pure nonuniform grid in x, y and z axes

* src_fdtd_unigrid: independent folder to run FDTD on pure uniform grid in x, y and z axes
  It pads nb layered on each side of x, y, z axes.
  
* src_fdtd_unigrid_v2: independent folder to run FDTD on pure uniform grid in x, y and z axes
 The difference with src_fdtd_unigrid is: It removes nb layers above sea surface.

Instructions to run
===================

1. Compile the code:

   cd src;

   make (to compile with mpicc)

   or

   make GPU=1 (to comipile with mpicc and CUDA)

2. Running reproducible examples

    Example 1: run_1d

    cd run_1d;

    bash run.sh
    

    To plot the result, use:
    python3 plot_emdata.py
    
    Example 2: run_bathy_2d

    cd run_bathy_2d

    bash run.sh
    

    To plot the result，use:

    python3 plot_cmp_libEMM_mare2dem.py

    python3 plot_emdata.py
    
NB: the input files may be generated prior to running. The resistivity files and nonuniform grid in binary format will be generated in src_nugrid:

    cd src_nugrid;

    make
    
    ./main
    
    One may need to copy rho11, rho22, rho33 and x3nu into /run_1d and /run_bathy_2d.
    


Acknowledgement:
================
The initiative to start this ficititious wave domain modelling project begins when I was a scientist working in Electromagnetic Geoservices ASA (EMGS). I developed some modeling codes in my free time, but they never work correctly.  After I left EMGS in 2020, I restarted everything from scratch using all things I learned from Madagascar open software development. It took me more than one year to make it work correctly: the solution now matches the semi-analytic one. During the development, I benefit from the discussion with Rune Mittet, in order to understand his method.

Please give a credit to the following publication if any component of libEMM is used in your research:

1. Pengliang Yang, libEMM: A fictious wave domain 3D CSEM modelling library bridging sequential and parallel GPU implementation, 2023 Computer Physics Communications, Vol. 288 p. 108745 [doi:10.1016/j.cpc.2023.108745](https://doi.org/10.1016/j.cpc.2023.108745)

2. Pengliang Yang and Rune Mittet, Controlled-source electromagnetics modelling using high order finite-difference time-domain method on a nonuniform grid 2023 Geophysics , Vol. 88, No. 2 Society of Exploration Geophysicists p. E53-E67
[doi:10.1190/geo2022-0134.1](https://doi.org/10.1190/geo2022-0134.1)

