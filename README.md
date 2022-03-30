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

Code structure:
.
├── bin
├── doc
│   └── libEMM_mannual.pdf
├── include
│   ├── acqui.h
│   ├── constants.h
│   ├── constants.h~
│   ├── cstd.h
│   ├── emf.h
│   ├── emf.h~
│   ├── interp.h
│   └── mpi_info.h
├── LICENSE
├── README.md
├── run_1d
│   ├── comparison.png
│   ├── create_acquisition_oneline.f90
│   ├── emf_0001.txt
│   ├── emf_ref.txt
│   ├── plot_emdata.py
│   ├── receivers.txt
│   ├── rho11
│   ├── rho22
│   ├── rho33
│   ├── run.sh
│   ├── sources.txt
│   ├── src_rec_table.txt
│   ├── time_info_gpu.txt
│   ├── time_info_omp1.txt
│   ├── time_info_omp2.txt
│   ├── time_info_omp4.txt
│   ├── time_info_omp8.txt
│   ├── time_info.txt
│   ├── x1nu
│   ├── x2nu
│   └── x3nu
├── run_bathy_2d
│   ├── bathy_comparison.png
│   ├── broadside_amp_phase.png
│   ├── broadside_emf_0001.txt
│   ├── create_acquisition_broadside.f90
│   ├── create_acquisition_inline.f90
│   ├── create_acquisition_oneline.f90
│   ├── emf_0001.txt
│   ├── mare2dem.txt
│   ├── mesh.png
│   ├── plot_cmp_libEMM_mare2dem.py
│   ├── plot_emdata.py
│   ├── plot_survey_layout.gnu
│   ├── receivers.txt
│   ├── rho11
│   ├── rho22
│   ├── rho33
│   ├── rho3d.png
│   ├── run.sh
│   ├── sources.txt
│   ├── src_rec_table.txt
│   ├── survey.png
│   ├── time_info.txt
│   ├── x1nu
│   ├── x2nu
│   └── x3nu
├── src
│   ├── acqui_init_close.c
│   ├── airwave_bc.c
│   ├── check_convergence.c
│   ├── compute_green_function.c
│   ├── cpml.c
│   ├── cstd.c
│   ├── cuda_fdtd.cuh
│   ├── cuda_modeling.cu
│   ├── do_modeling.c
│   ├── dtft_emf.c
│   ├── emf_init_close.c
│   ├── extend_model.c
│   ├── extract_emf.c
│   ├── fdtd.c
│   ├── inject_src_fwd.c
│   ├── interpolation.c
│   ├── main.c
│   ├── Makefile
│   ├── nugrid_init_close.c
│   ├── sanity_check.c
│   ├── vandemonde.c
│   └── write_data.c
└── src_nugrid
    ├── create_nugrid.c
    ├── cstd.c
    ├── cstd.h
    ├── main.c
    ├── Makefile
    ├── output_model_vtk.py
    ├── plot_3d_faces.m
    ├── plot_mesh.m
    ├── plot_slices.m
    └── readme.txt


Acknowledgement: The initiative to start this ficititious wave domain modelling project begins when I was a scientist working in Electromagnetic Geoservices ASA (EMGS). I developed some modeling codes in my free time, but they never work correctly.  After I left EMGS in 2020, I restarted everything from scratch using all things I learned from Madagascar open software development. It took me more than one year to make it work correctly: the solution now matches the semi-analytic one. During the development, I benefit from the discussion with Rune Mittet, in order to understand his method.
