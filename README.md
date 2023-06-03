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

* src: the source code in .c and .cu/.cuh.

* include: the header files in .h

* doc: the mannual for libEMM

* bin: the folderto store executable after compilation

* run_1d: Application example in 3D model of 1D layered sturture

* run_bathy_2d: Appliation example in 3D model of 2D structure with seafloor bathymetry

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

@Article{Yang_2023_libEMM,
author = {Pengliang Yang},
title = {{libEMM: A fictious wave domain 3D CSEM modelling library bridging sequential and parallel GPU implementation}},
journal = {Computer Physics Communications},
volume = {288},
pages = {108745},
year = {2023},
issn = {0010-4655},
doi = {https://doi.org/10.1016/j.cpc.2023.108745},
}
