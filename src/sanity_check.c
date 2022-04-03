/* Sanity check for stability condition, dispersion requirement, and 
 * determine the optimal temporal sampling dt and the required number of time steps nt
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

void sanity_check(emf_t *emf)
{
  int i, i1, i2, i3, i3_;
  float tmp, tmp1,tmp2,eta,Rmax,cfl;
  float D1, D2, D3, s3, t3;

  /*----------------------------------------------------------------------------------*/
  /* Stage 1: find minimum and maximum velocity for stability conditon and dispersion */
  /*    emf->vmin: important for minimum number of points per wavelength              */
  /*    emf->vmax: important for CFL condition and fdtd computing box                 */
  /*----------------------------------------------------------------------------------*/
  emf->rhomax = MAX( MAX(emf->rho11[0][0][0], emf->rho22[0][0][0]), emf->rho33[0][0][0]);
  emf->rhomin = MIN( MIN(emf->rho11[0][0][0], emf->rho22[0][0][0]), emf->rho33[0][0][0]);
  /* sigma=2*omega0*eps --> inveps=2*omega0*rho */
  emf->vmin = sqrt(2.*emf->omega0*emf->rhomin*invmu0);
  emf->vmax = sqrt(2.*emf->omega0*emf->rhomax*invmu0);

  /* set the values assuming uniform grid */
  D1 = 0;
  D2 = 0;
  D3 = 0;
  if(emf->rd==1){
    D1 = 1.;
    D2 = 1.;
    D3 = 1.;
  }else if(emf->rd==2){
    D1 = (fabs(1.125) + fabs(-0.04167));
    D2 = (fabs(1.125) + fabs(-0.04167));
    D3 = (fabs(1.125) + fabs(-0.04167));
  }else if(emf->rd==3){
    D1 = (fabs(1.17188) + fabs(-0.06510) + fabs(0.00469));
    D2 = (fabs(1.17188) + fabs(-0.06510) + fabs(0.00469));
    D3 = (fabs(1.17188) + fabs(-0.06510) + fabs(0.00469));
  }
  D1 *= 2;
  D2 *= 2;
  D3 *= 2;
  D1 /= emf->d1;
  D2 /= emf->d2;
  D3 /= emf->d3;
  
  eta = 0;
  for(i3=0; i3<emf->n3; ++i3){
    i3_ = i3+emf->nbe;
    if(emf->nugrid){//adapt the value in case of non-uniform grid
      s3 = 0;
      t3 = 0;
      for(i=0; i<2*emf->rd; i++) {
	s3 += fabs(emf->v3[i3_][i]);
	t3 += fabs(emf->v3s[i3_][i]);
      }
      D3 = MAX(s3, t3);
    }

    for(i2=0; i2<emf->n2; ++i2){
      for(i1=0; i1<emf->n1; ++i1){
	tmp1 = MIN( MIN(emf->rho11[i3][i2][i1], emf->rho22[i3][i2][i1]), emf->rho33[i3][i2][i1]);
	tmp2 = MAX( MAX(emf->rho11[i3][i2][i1], emf->rho22[i3][i2][i1]), emf->rho33[i3][i2][i1]);
	if(emf->airwave==1 && i3==0){//sigma_[11,22]=0.5*sigma_water, rho_interface=2*rho_water
	  tmp1 *= 2;
	  tmp2 *= 2;
	}
	if(emf->rhomin>tmp1)  emf->rhomin = tmp1;
	if(emf->rhomax<tmp2)  emf->rhomax = tmp2;
	emf->vmin = sqrt(2.*emf->omega0*emf->rhomin*invmu0);
	emf->vmax = sqrt(2.*emf->omega0*emf->rhomax*invmu0);

	tmp = 0.5*sqrt(D1*D1 + D2*D2 + D3*D3);
	tmp *= emf->vmax;
	if(tmp>eta) eta = tmp;
      }
    }
  }


  /*------------------------------------------------------------------------*/
  /* Stage 2: determine the optimal dt and nt automatically                 */
  /*------------------------------------------------------------------------*/
  if(!getparfloat("dt", &emf->dt)) emf->dt = 0.99/eta;
  /* temporal sampling, determine dt by stability condition if not provided */
  cfl = emf->dt*eta;
  if(emf->verb) printf("cfl=%g\n", cfl); 
  if(cfl > 1.0) err("CFL condition not satisfied!");
  emf->freqmax = emf->vmin/(emf->Glim*MIN(MIN(emf->d1,emf->d2),emf->d3));

  if(!getparint("nt", &emf->nt)){
    Rmax = MAX((emf->n1-1)*emf->d1, (emf->n2-1)*emf->d2);
    emf->nt = (int)(1.5*Rmax/(emf->vmin*emf->dt));
  }/* automatically determine nt using maximum offset if not provided */
  if(emf->verb){
    printf("[rhomin, rhomax]=[%g, %g] Ohm-m\n", emf->rhomin, emf->rhomax);
    printf("[vmin, vmax]=[%g, %g] m/s\n", emf->vmin, emf->vmax);
    printf("freqmax=%g Hz\n", emf->freqmax);
    printf("dt=%g s\n",  emf->dt);
    printf("nt=%d\n",  emf->nt);
  }
}
