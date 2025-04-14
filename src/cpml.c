/* convolutional perfectly matched layers (CPML)
 *--------------------------------------------------------------------
 *
 *   Copyright (c) 2020-2022, Harbin Institute of Technology, China
 *   Author: Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *--------------------------------------------------------------------*/
#include "cstd.h"
#include "emf.h"
#include "constants.h"

void cpml_init(emf_t *emf)
/*< initialize PML abosorbing coefficients >*/
{
  float x, damp0, damp;
  int i;

  /* by default, we choose: kappa=1, alpha=PI*f0 for CPML */
  //float f0 = 1;
  float alpha=PI*emf->f0; /* alpha>0 makes CPML effectively attenuates evanescent waves */
  //const float Rc = 1e-5; /* theoretic reflection coefficient for PML */
  //float vmax = sqrt(emf->vmin*emf->vmax);
  
  emf->apml = alloc1float(emf->nb);
  emf->bpml = alloc1float(emf->nb);

  //damp0=-3.*vmax*logf(Rc)/(2.*L);
  damp0 = 341.9;
  for(i=0; i<emf->nb; ++i)    {
    x=(float)(emf->nb-i)/(float)emf->nb;
    damp = damp0*x*x; /* damping profile in direction 1, sigma/epsilon0 */
    emf->bpml[i] = expf(-(damp+alpha)*emf->dt);
    emf->apml[i] = damp*(emf->bpml[i]-1.0)/(damp+alpha);
  }

}


void cpml_close(emf_t *emf)
{
  free(emf->apml);
  free(emf->bpml);
}

