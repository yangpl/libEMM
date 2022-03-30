/* 3D CSEM modeling using FDTD method
 * Reference:
 *    [1] Pengliang Yang and Rune Mittet, 2022, Controlled-source 
 *        electromagnetics modelling using high order finite-difference 
 *        time-domain method on a nonuniform grid, Geophysics
 *--------------------------------------------------------------------
 *
 *   Copyright (c) 2020-2022, Harbin Institute of Technology, China
 *   Author: Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *--------------------------------------------------------------------*/
#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cstd.h"
#include "emf.h"
#include "acqui.h"
#include "interp.h"
#include "constants.h"

#include "mpi_info.h"

int iproc, nproc, ierr;

void emf_init(emf_t *emf);
void emf_close(emf_t *emf);

void acqui_init(acqui_t *acqui, emf_t * emf);
void acqui_close(acqui_t *acqui);

void sanity_check(emf_t *emf);

void interpolation_init(acqui_t *acqui, emf_t *emf, interp_t *interp_rg, interp_t *interp_sg);
void interpolation_close(emf_t *emf, interp_t *interp_rg, interp_t *interp_sg);
void interpolation_weights(acqui_t *acqui, emf_t *emf, interp_t *interp_rg, interp_t *interp_sg);

void extend_model_init(emf_t *emf);
void extend_model_close(emf_t *emf);

void cpml_init(emf_t *emf);
void cpml_close(emf_t *emf);

void dtft_emf_init(emf_t *emf);
void dtft_emf_close(emf_t *emf);


void airwave_bc_init(emf_t *emf);
void airwave_bc_close(emf_t *emf);

void compute_green_function(emf_t *emf);
void extract_emf(acqui_t *acqui, emf_t *emf, interp_t *interp_rg, interp_t *interp_sg);
void write_data(acqui_t *acqui, emf_t *emf);

#ifdef GPU
void cuda_modeling(acqui_t *acqui, emf_t *emf, interp_t *interp_rg, interp_t *interp_sg);
#else
void do_modeling(acqui_t *acqui, emf_t *emf, interp_t *interp_rg, interp_t *interp_sg);
#endif

/*-----------------------------------------------------------------*/
int main(int argc, char* argv[])
{
  emf_t *emf;
  acqui_t *acqui;
  interp_t *interp_rg;/* interpolation for regular grid */
  interp_t *interp_sg;/* interpolation for staggered grid */

  int it, ifreq;
  float _Complex omegap;

  MPI_Init(&argc, &argv);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  
  initargs(argc,argv);

  printf("=====================================================\n");
  printf("    Welcome to 3D CSEM modeling using FDTD           \n");
  printf("            Author: Pengliang Yang                   \n");
  printf("            E-mail: ypl.2100@gmail.com               \n");
  printf("  Copyright (c) 2020-2022. All rights reserved. \n");
  printf("=====================================================\n");

  emf = (emf_t *)malloc(sizeof(emf_t));
  acqui = (acqui_t *)malloc(sizeof(acqui_t));
  emf_init(emf);
  acqui_init(acqui, emf);

  emf->verb = (iproc==0)?1:0;//print verbose information only on master process
  sanity_check(emf);//check CFL stability condition

  interp_rg = (interp_t *) malloc(sizeof(interp_t));
  interp_sg = (interp_t *) malloc(sizeof(interp_t));
  interpolation_init(acqui, emf, interp_rg, interp_sg);
  interpolation_weights(acqui, emf, interp_rg, interp_sg);
  

  /* construct a source time function in fictious wave domain */
  emf->stf = alloc1float(emf->nt);
  memset(emf->stf, 0, emf->nt*sizeof(float));
  emf->stf[0] = 1.;//Dirac delta source makes the output of EM fields=Green's function

  emf->expfactor = alloc2complexf(emf->nfreq, emf->nt);
  for(it=0; it<emf->nt; ++it){
    for(ifreq=0; ifreq<emf->nfreq; ifreq++) {
      omegap = (1.0+I)*sqrt(emf->omega0*emf->omegas[ifreq]);/* omega' in fictitous wave domain */
      emf->expfactor[it][ifreq] = cexp(I*omegap*(it+0.5)*emf->dt);//note: time at it+1 for field
    }
  }

  emf->dcal_fd = alloc3complexf(acqui->nrec, emf->nfreq, emf->nchrec);
  memset(&emf->dcal_fd[0][0][0], 0, acqui->nrec*emf->nchrec*emf->nfreq*sizeof(float _Complex));

  if(emf->verb) printf("-------------- forward modeling --------------\n");
  extend_model_init(emf);
  cpml_init(emf);
  dtft_emf_init(emf);
  if(emf->airwave) airwave_bc_init(emf);


  /*------------------------------------------------------------*/
#ifdef GPU
  cuda_modeling(acqui, emf, interp_rg, interp_sg); /* mode=0 */  
#else
  do_modeling(acqui, emf, interp_rg, interp_sg); /* mode=0 */  
#endif
  
  /*------------------------------------------------------------*/
  compute_green_function(emf);
  extract_emf(acqui, emf, interp_rg, interp_sg);
  write_data(acqui, emf);


  /*------------------------------------------------------------*/
  if(emf->airwave) airwave_bc_close(emf);
  dtft_emf_close(emf);
  extend_model_close(emf);
  cpml_close(emf);


  interpolation_close(emf, interp_rg, interp_sg);
  free(interp_rg);
  free(interp_sg);

  free(emf->stf);
  free2complexf(emf->expfactor);
  free3complexf(emf->dcal_fd);
  
  acqui_close(acqui);
  emf_close(emf);
  free(emf);
  free(acqui);
    
  MPI_Finalize();

  return 0;
}
