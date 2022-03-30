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
  float x, L, damp0, damp;
  int i1, i2, i3;

  /* by default, we choose: kappa=1, alpha=PI*f0 for CPML */
  //float f0 = 1;
  float alpha=PI*emf->f0; /* alpha>0 makes CPML effectively attenuates evanescent waves */
  //const float Rc = 1e-5; /* theoretic reflection coefficient for PML */
  //float vmax = sqrt(emf->vmin*emf->vmax);
  
  emf->a1 = alloc1float(emf->nb);
  emf->a2 = alloc1float(emf->nb);
  emf->a3 = alloc1float(emf->nb);
  emf->b1 = alloc1float(emf->nb);
  emf->b2 = alloc1float(emf->nb);
  emf->b3 = alloc1float(emf->nb);

  L=emf->nb*emf->d1;    
  //damp0=-3.*vmax*logf(Rc)/(2.*L);
  damp0 = 341.9;
  for(i1=0; i1<emf->nb; ++i1)    {
    x=(emf->nb-i1)*emf->d1;
    x = x/L;
    damp = damp0*x*x; /* damping profile in direction 1, sigma/epsilon0 */
    // damp = damp0*(1.0-cos(0.5*PI*x));
    emf->b1[i1] = expf(-(damp+alpha)*emf->dt);
    emf->a1[i1] = damp*(emf->b1[i1]-1.0)/(damp+alpha);
    //printf("i=%d a1=%g b1=%g\n", i1, emf->a1[i1], emf->b1[i1]);
  }

  L=emf->nb*emf->d2;    
  //damp0=-3.*vmax*logf(Rc)/(2.*L);
  for(i2=0; i2<emf->nb; ++i2)    {
    x=(emf->nb-i2)*emf->d2;
    x = x/L;
    damp = damp0*x*x;/* damping profile in direction 2, sigma/epsilon0 */
    //damp = damp0*(1.0-cos(0.5*PI*x));
    emf->b2[i2] = expf(-(damp+alpha)*emf->dt);
    emf->a2[i2] = damp*(emf->b2[i2]-1.0)/(damp+alpha);
    //printf("i=%d a2=%g b2=%g\n", i2, emf->a2[i2], emf->b2[i2]);
  }

  L=emf->nb*emf->d3;    
  //damp0=-3.*vmax*logf(Rc)/(2.*L);
  for(i3=0; i3<emf->nb; ++i3)    {
    x=(emf->nb-i3)*emf->d3;
    x = x/L;
    damp = damp0*x*x;/* damping profile in direction 3, sigma/epsilon0 */
    //damp = damp0*(1.0-cos(0.5*PI*x));
    emf->b3[i3] = expf(-(damp+alpha)*emf->dt);
    emf->a3[i3] = damp*(emf->b3[i3]-1.0)/(damp+alpha);
    //printf("i=%d a3=%g b3=%g\n", i3, emf->a3[i3], emf->b3[i3]);
  }
}


void cpml_close(emf_t *emf)
{
  free(emf->a1);
  free(emf->a2);
  free(emf->a3);
  free(emf->b1);
  free(emf->b2);
  free(emf->b3);
}

