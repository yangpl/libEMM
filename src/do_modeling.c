/* CSEM forward modeling
 *--------------------------------------------------------------------
 *
 *   Copyright (c) 2020-2022, Harbin Institute of Technology, China
 *   Author: Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *--------------------------------------------------------------------*/
#include <mpi.h>

#include "cstd.h"
#include "acqui.h"
#include "emf.h"
#include "interp.h"
#include "constants.h"

void fdtd_init(emf_t *emf);
void fdtd_null(emf_t *emf);
void fdtd_close(emf_t *emf);
void fdtd_curlH(emf_t *emf, int it);
void fdtd_update_E(emf_t *emf, int it);
void fdtd_curlE(emf_t *emf, int it);
void fdtd_update_H(emf_t *emf, int it);

void inject_electric_src_fwd(acqui_t *acqui, emf_t *emf, interp_t *interp_rg, interp_t *interp_sg, int it);
void inject_magnetic_src_fwd(acqui_t *acqui, emf_t *emf, interp_t *interp_rg, interp_t *interp_sg, int it);

void airwave_bc_update_H(emf_t *emf);
void airwave_bc_update_E(emf_t *emf);

void dtft_emf(emf_t *emf, int it, float ***E1, float _Complex ****fwd_E1);
int check_convergence(emf_t *emf);


void do_modeling(acqui_t *acqui, emf_t *emf, interp_t *interp_rg, interp_t *interp_sg)
{
  int it, ic;
  double t_updateE,t_updateH,t_injectE,t_injectH,t_curlE, t_curlH,t_dtft,t_conv, t_airwave, t0;
  
  fdtd_init(emf);
  if(emf->verb){
    t0 = 0;
    t_curlE = 0.;
    t_injectH = 0.;
    t_updateH = 0.;
    t_curlH = 0.;
    t_injectE = 0.;
    t_updateE = 0.;
    t_airwave = 0;
    t_dtft= 0.;
    t_conv=0.;
  }


  for(it=0; it<emf->nt; it++){
    if(emf->verb && it%50==0) printf("it---- %d\n", it);

    /*--------------------------------------------------------------*/
    if(emf->verb) t0 = MPI_Wtime();
    fdtd_curlH(emf, it);
    if(emf->verb) t_curlH += MPI_Wtime()-t0;

    if(emf->verb) t0 = MPI_Wtime();
    inject_electric_src_fwd(acqui, emf, interp_rg, interp_sg, it);
    if(emf->verb) t_injectE += MPI_Wtime()-t0;


    if(emf->verb) t0 = MPI_Wtime();
    fdtd_update_E(emf, it);
    if(emf->verb) t_updateE += MPI_Wtime()-t0;

    if(emf->verb) t0 = MPI_Wtime();
    if(emf->airwave) airwave_bc_update_E(emf);
    if(emf->verb) t_airwave += MPI_Wtime()-t0;


    /*--------------------------------------------------------------*/
    if(emf->verb) t0 = MPI_Wtime();
    fdtd_curlE(emf, it); 
    if(emf->verb) t_curlE += MPI_Wtime()-t0;

    if(emf->verb) t0 = MPI_Wtime();
    inject_magnetic_src_fwd(acqui, emf, interp_rg, interp_sg, it);
    if(emf->verb) t_injectH += MPI_Wtime()-t0;

    
    if(emf->verb) t0 = MPI_Wtime();
    fdtd_update_H(emf, it); 
    if(emf->verb) t_updateH += MPI_Wtime()-t0;

    if(emf->verb) t0 = MPI_Wtime();
    if(emf->airwave) airwave_bc_update_H(emf);
    if(emf->verb) t_airwave += MPI_Wtime()-t0;
	
    /*--------------------------------------------------------------*/
    if(emf->verb) t0 = MPI_Wtime();
    for(ic=0; ic<emf->nchrec; ++ic) {
      if     (strcmp(emf->chrec[ic],"Ex")==0) dtft_emf(emf, it, emf->E1, emf->fwd_E1);
      else if(strcmp(emf->chrec[ic],"Ey")==0) dtft_emf(emf, it, emf->E2, emf->fwd_E2);
      else if(strcmp(emf->chrec[ic],"Ez")==0) dtft_emf(emf, it, emf->E3, emf->fwd_E3);
      else if(strcmp(emf->chrec[ic],"Hx")==0) dtft_emf(emf, it, emf->H1, emf->fwd_H1);
      else if(strcmp(emf->chrec[ic],"Hy")==0) dtft_emf(emf, it, emf->H2, emf->fwd_H2);
      else if(strcmp(emf->chrec[ic],"Hz")==0) dtft_emf(emf, it, emf->H3, emf->fwd_H3);
    }
    if(emf->verb) t_dtft += MPI_Wtime()-t0;

    /*--------------------------------------------------------------*/
    if(emf->verb) t0 = MPI_Wtime();
    if(it%100==0){/* convergence check */
      emf->ncorner = check_convergence(emf);
      if(emf->verb) printf("%d corners of the cube converged!\n", emf->ncorner);
      if(emf->ncorner==8) break;/* all 8 corners converged, exit now */
    }
    if(emf->verb) t_conv += MPI_Wtime()-t0;
  }


  if(emf->verb) {
    t0 = t_curlH + t_injectE + t_updateE
      + t_curlE + t_injectH + t_updateH
      + t_airwave + t_dtft + t_conv;
    FILE *fp = fopen("time_info.txt", "w");
    fprintf(fp, "curlE   \t %e\n", t_curlE);
    fprintf(fp, "injectH \t %e\n", t_injectH);
    fprintf(fp, "udpateH \t %e\n", t_updateH);
    fprintf(fp, "curlH   \t %e\n", t_curlH);
    fprintf(fp, "injectE \t %e\n", t_injectE);
    fprintf(fp, "udpateE \t %e\n", t_updateE);
    fprintf(fp, "airwave \t %e\n", t_airwave);
    fprintf(fp, "dtft    \t %e\n", t_dtft);
    fprintf(fp, "conv    \t %e\n", t_conv);
    fprintf(fp, "total   \t %e\n", t0);    
    fclose(fp);
    
    printf("-------------- elapsed time --------------------\n");
    printf(" compute curlE:           %e s\n", t_curlE);
    printf(" inject magnetic source:  %e s\n", t_injectH);
    printf(" update magnetic field:   %e s\n", t_updateH);

    printf(" compute curlH:           %e s\n", t_curlH);
    printf(" inject electric source:  %e s\n", t_injectE);
    printf(" update electric field:   %e s\n", t_updateE);

    printf(" Airwave computation:     %e s\n", t_airwave);
    printf(" DTFT EM field:           %e s\n", t_dtft);
    printf(" convergence check:       %e s\n", t_conv);
    printf(" Total modeling time:     %e s\n", t0);
    printf("------------------------------------------------\n");
  }
  
  fdtd_close(emf); 
  
}
