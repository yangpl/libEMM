/* DTFT on the fly during time integration
 *--------------------------------------------------------------------
 *
 *   Copyright (c) 2020, Harbin Institute of Technology, China
 *   Author: Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *--------------------------------------------------------------------*/
#ifdef _OPENMP
#include <omp.h>
#endif

#include "cstd.h"
#include "emf.h"

void dtft_emf_init(emf_t *emf)
{
  emf->fwd_E1 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
  emf->fwd_E2 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
  emf->fwd_E3 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
  emf->fwd_H1 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
  emf->fwd_H2 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
  emf->fwd_H3 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);

  memset(&emf->fwd_E1[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
  memset(&emf->fwd_E2[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
  memset(&emf->fwd_E3[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
  memset(&emf->fwd_H1[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
  memset(&emf->fwd_H2[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
  memset(&emf->fwd_H3[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
}

void dtft_emf_close(emf_t *emf)
{
  free4complexf(emf->fwd_E1);
  free4complexf(emf->fwd_E2);
  free4complexf(emf->fwd_E3);
  free4complexf(emf->fwd_H1);
  free4complexf(emf->fwd_H2);
  free4complexf(emf->fwd_H3);

}


void dtft_emf(emf_t *emf, int it, float ***E1, float _Complex ****fwd_E1)
{
  int i1,i2,i3,ifreq;
  float _Complex factor;
  
  int i1min=emf->nb;//lower bound for index i1
  int i2min=emf->nb;//lower bound for index i2
  int i3min=emf->nb;//lower bound for index i3
  int i1max=emf->n1pad-1-emf->nb;//upper bound for index i1
  int i2max=emf->n2pad-1-emf->nb;//upper bound for index i2
  int i3max=emf->n3pad-1-emf->nb;//upper bound for index i3

  for(ifreq=0; ifreq<emf->nfreq; ++ifreq){
    factor = emf->expfactor[it][ifreq];

#ifdef _OPENMP
#pragma omp parallel for default(none)				\
  schedule(static)						\
  private(i1,i2,i3)						\
  shared(i1min,i1max,i2min,i2max,i3min,i3max,factor,ifreq,E1,fwd_E1)
#endif
    for(i3=i3min; i3<i3max; ++i3){
      for(i2=i2min; i2<i2max; ++i2){
	for(i1=i1min; i1<i1max; ++i1){
	  fwd_E1[ifreq][i3][i2][i1] += E1[i3][i2][i1]*factor;
	}
      }
    }

  }/* end for ifreq */
}

