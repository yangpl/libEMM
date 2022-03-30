/* compute the Green's function of the diffusive Maxwell equation
 *--------------------------------------------------------------------
 *
 *   Copyright (c) 2020-2022, Harbin Institute of Technology, China
 *   Author: Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *--------------------------------------------------------------------*/
#include "cstd.h"
#include "emf.h"

void compute_green_function(emf_t *emf)
{
  int ifreq,it,i1,i2,i3;
  float _Complex omegap, src_fd, s;

  int i1min=emf->nb;
  int i2min=emf->nb;
  int i3min=emf->nb;
  int i1max=emf->n1pad-1-emf->nb;
  int i2max=emf->n2pad-1-emf->nb;
  int i3max=emf->n3pad-1-emf->nb;

  for(ifreq=0; ifreq < emf->nfreq; ++ifreq){
  /* omega' in fictitous wave domain */
    omegap = (1.0+I)*sqrtf(emf->omega0*emf->omegas[ifreq]);
    src_fd = 0.;
    for(it=0; it<emf->nt; ++it)  src_fd += emf->stf[it]*cexp(I*omegap*it*emf->dt);//J'
    s = csqrt(-I*0.5*emf->omegas[ifreq]/emf->omega0);

#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3)							\
  shared(i1min, i1max, i2min, i2max, i3min, i3max, ifreq, src_fd, s, emf)
#endif
    for(i3=i3min; i3<i3max; ++i3){
      for(i2=i2min; i2<i2max; ++i2){
	for(i1=i1min; i1<i1max; ++i1){
	  emf->fwd_E1[ifreq][i3][i2][i1] /= src_fd;
	  emf->fwd_E2[ifreq][i3][i2][i1] /= src_fd;
	  emf->fwd_E3[ifreq][i3][i2][i1] /= src_fd;
	  emf->fwd_H1[ifreq][i3][i2][i1] /= src_fd;
	  emf->fwd_H2[ifreq][i3][i2][i1] /= src_fd;
	  emf->fwd_H3[ifreq][i3][i2][i1] /= src_fd;
	  emf->fwd_E1[ifreq][i3][i2][i1] *= s;
	  emf->fwd_E2[ifreq][i3][i2][i1] *= s;
	  emf->fwd_E3[ifreq][i3][i2][i1] *= s;
	}
      }
    }
  }
  
}
