/* polynomial interpolation by inverting Vandermonde matrix
 *--------------------------------------------------------------------
 *
 *   Copyright (c) 2020-2022, Harbin Institute of Technology, China
 *   Author: Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *--------------------------------------------------------------------*/
#include "cstd.h"
#include "acqui.h"
#include "emf.h"
#include "interp.h"
#include "constants.h"

void vandermonde(int n, float *x, float *a, float *f);

void interpolation_init(acqui_t *acqui, emf_t *emf,
			interp_t *interp_rg, interp_t *interp_sg)
{
  if(acqui->nsubsrc%2==0) err("nsubsrc must be odd number!");
  if(acqui->nsubrec%2==0) err("nsubrec must be odd number!");

  interp_rg->rec_i1 = alloc2int(acqui->nsubrec, acqui->nrec);
  interp_rg->rec_i2 = alloc2int(acqui->nsubrec, acqui->nrec);
  interp_rg->rec_i3 = alloc2int(acqui->nsubrec, acqui->nrec);
  interp_rg->rec_w1 = alloc3float(2*emf->rd, acqui->nsubrec, acqui->nrec);
  interp_rg->rec_w2 = alloc3float(2*emf->rd, acqui->nsubrec, acqui->nrec);
  interp_rg->rec_w3 = alloc3float(2*emf->rd, acqui->nsubrec, acqui->nrec);
  interp_rg->rec_v1 = alloc3float(2*emf->rd, acqui->nsubrec, acqui->nrec);
  interp_rg->rec_v2 = alloc3float(2*emf->rd, acqui->nsubrec, acqui->nrec);
  interp_rg->rec_v3 = alloc3float(2*emf->rd, acqui->nsubrec, acqui->nrec);

  interp_rg->src_i1 = alloc2int(acqui->nsubsrc, acqui->nsrc);
  interp_rg->src_i2 = alloc2int(acqui->nsubsrc, acqui->nsrc);
  interp_rg->src_i3 = alloc2int(acqui->nsubsrc, acqui->nsrc);
  interp_rg->src_w1 = alloc3float(2*emf->rd, acqui->nsubsrc, acqui->nsrc);
  interp_rg->src_w2 = alloc3float(2*emf->rd, acqui->nsubsrc, acqui->nsrc);
  interp_rg->src_w3 = alloc3float(2*emf->rd, acqui->nsubsrc, acqui->nsrc);

  interp_sg->rec_i1 = alloc2int(acqui->nsubrec,acqui->nrec);
  interp_sg->rec_i2 = alloc2int(acqui->nsubrec,acqui->nrec);
  interp_sg->rec_i3 = alloc2int(acqui->nsubrec,acqui->nrec);
  interp_sg->rec_w1 = alloc3float(2*emf->rd, acqui->nsubrec, acqui->nrec);
  interp_sg->rec_w2 = alloc3float(2*emf->rd, acqui->nsubrec, acqui->nrec);
  interp_sg->rec_w3 = alloc3float(2*emf->rd, acqui->nsubrec, acqui->nrec);
  interp_sg->rec_v1 = alloc3float(2*emf->rd, acqui->nsubrec, acqui->nrec);
  interp_sg->rec_v2 = alloc3float(2*emf->rd, acqui->nsubrec, acqui->nrec);
  interp_sg->rec_v3 = alloc3float(2*emf->rd, acqui->nsubrec, acqui->nrec);

  interp_sg->src_i1 = alloc2int(acqui->nsubsrc, acqui->nsrc);
  interp_sg->src_i2 = alloc2int(acqui->nsubsrc, acqui->nsrc);
  interp_sg->src_i3 = alloc2int(acqui->nsubsrc, acqui->nsrc);
  interp_sg->src_w1 = alloc3float(2*emf->rd, acqui->nsubsrc, acqui->nsrc);
  interp_sg->src_w2 = alloc3float(2*emf->rd, acqui->nsubsrc, acqui->nsrc);
  interp_sg->src_w3 = alloc3float(2*emf->rd, acqui->nsubsrc, acqui->nsrc);

}

void interpolation_close(emf_t *emf, interp_t *interp_rg, interp_t *interp_sg)
{
  free2int(interp_rg->rec_i1);
  free2int(interp_rg->rec_i2);
  free2int(interp_rg->rec_i3);
  free3float(interp_rg->rec_w1);
  free3float(interp_rg->rec_w2);
  free3float(interp_rg->rec_w3);
  free3float(interp_rg->rec_v1);
  free3float(interp_rg->rec_v2);
  free3float(interp_rg->rec_v3);

  free2int(interp_rg->src_i1);
  free2int(interp_rg->src_i2);
  free2int(interp_rg->src_i3);
  free3float(interp_rg->src_w1);
  free3float(interp_rg->src_w2);
  free3float(interp_rg->src_w3);

  free2int(interp_sg->rec_i1);
  free2int(interp_sg->rec_i2);
  free2int(interp_sg->rec_i3);
  free3float(interp_sg->rec_w1);
  free3float(interp_sg->rec_w2);
  free3float(interp_sg->rec_w3);
  free3float(interp_sg->rec_v1);
  free3float(interp_sg->rec_v2);
  free3float(interp_sg->rec_v3);

  free2int(interp_sg->src_i1);
  free2int(interp_sg->src_i2);
  free2int(interp_sg->src_i3);
  free3float(interp_sg->src_w1);
  free3float(interp_sg->src_w2);
  free3float(interp_sg->src_w3);
  
}

/*------------------------------------------------------------------------------- 
 * ( f(x0))  (1  x0-x (x0-x)^2 ... (x0-x)^n) ( f(x)      )
 * ( f(x1)) =(1  x1-x (x1-x)^2 ... (x1-x)^n) ( f^1(x)    )
 * ( ...  )  (...                          ) ( ...       )
 * ( f(xn))  (1  xn-x (xn-x)^2 ... (xn-x)^n) ( f^n(x)/n! )
 *           -------------------------------
 *            V^T (V=Vandermonde matrix)
 * Given the vector f=(f(x0),f(x1),...,f(xn))^T and Vandermonde matrix 
 * V(x0,x1,...,xn), the solution of Vandermonde matrix inversion V^T a =f
 * gives:
 *  (a0)  (f (x)    )  (w0 w1 ... wn) ( f(x0) )
 *  (a1)= (f'(x)    ) =(            ) ( f(x1) )
 *  (.)     ...        (            ) ( ...   )
 *  (an)  (f^n(x)/n!)  (            ) ( f(xn) )
 *              --------------
 *                 V^{-1}
 * we use the first row of inverse Vandermonde matrix as the interpolation weights:
 *  a0 = f(x) = \sum_{i=0}^{i=n} wi * f(xi)
 * From the above expression, we know all weights wi (i=0,...,n) can be obtained by
 * setting f(xi)=1 and f(xj)=0 (for all j\neq i).
 *-------------------------------------------------------------------------------*/
void interpolation_weights(acqui_t *acqui, emf_t *emf, interp_t *interp_rg, interp_t *interp_sg)
{
  const float eps=1e-5;

  float o1, o2, o3;
  float x1j,x2j,x3j;
  int j1,j2,j3;
  int i,j,isub,loop,m,isrc,irec;
  bool skip1,skip2,skip3;
  float *x1,*x2,*x3,*aa,*ff;
  float tmp, dlen1, dlen2, dlen3;
  interp_t *interp=NULL;
  
  x1 = alloc1float(2*emf->rd);
  x2 = alloc1float(2*emf->rd);
  x3 = alloc1float(2*emf->rd);
  aa = alloc1float(2*emf->rd);
  ff = alloc1float(2*emf->rd);

  /*---------- setup interpolation weights for receivers ----------*/
  for(irec=0; irec<acqui->nrec; ++irec){
    tmp = acqui->lenrec/acqui->nsubrec;
    dlen1 = tmp*cos(acqui->rec_dip[irec]*PI/180.)*cos(acqui->rec_azimuth[irec]*PI/180.);
    dlen2 = tmp*cos(acqui->rec_dip[irec]*PI/180.)*sin(acqui->rec_azimuth[irec]*PI/180.);
    dlen3 = tmp*sin(acqui->rec_dip[irec]*PI/180.);
    for(isub=0; isub<acqui->nsubrec; ++isub){
      j = isub - acqui->nsubrec/2;
      x1j = acqui->rec_x1[irec]+j*dlen1;
      x2j = acqui->rec_x2[irec]+j*dlen2;
      x3j = acqui->rec_x3[irec]+j*dlen3;
      
      for(loop=0; loop<2; loop++){
	if(loop==0){/* regular grid */
	  interp = interp_rg;
	  o1 = acqui->x1min;
	  o2 = acqui->x2min;
	  o3 = acqui->x3min;
	}else{/* staggered grid */
	  interp = interp_sg;
	  o1 = acqui->x1min+0.5*emf->d1;
	  o2 = acqui->x2min+0.5*emf->d2;
	  o3 = acqui->x3min+0.5*emf->d3;
	}
	j1 = (int)((x1j-o1)/emf->d1);/* integer part */
	j2 = (int)((x2j-o2)/emf->d2);/* integer part */
	j3 = (int)((x3j-o3)/emf->d3);/* integer part */

	interp->rec_i1[irec][isub] = j1 + emf->nbe;
	interp->rec_i2[irec][isub] = j2 + emf->nbe;
	interp->rec_i3[irec][isub] = j3 + emf->nbe;

	skip1 = false;
	skip2 = false;
	skip3 = false;
	if(fabs(x1j-o1-j1*emf->d1)< eps){
	  memset(interp->rec_w1[irec][isub], 0, 2*emf->rd*sizeof(float));
	  interp->rec_w1[irec][isub][emf->rd-1] = 1.;
	  memset(interp->rec_v1[irec][isub], 0, 2*emf->rd*sizeof(float));
	  interp->rec_v1[irec][isub][emf->rd-2] = -0.5/emf->d1;
	  interp->rec_v1[irec][isub][emf->rd] = 0.5/emf->d1;
	  skip1 = true;
	}
	if(fabs(x2j-o2-j2*emf->d2)< eps) {
	  memset(interp->rec_w2[irec][isub], 0, 2*emf->rd*sizeof(float));
	  interp->rec_w2[irec][isub][emf->rd-1] = 1.;
	  memset(interp->rec_v2[irec][isub], 0, 2*emf->rd*sizeof(float));
	  interp->rec_v2[irec][isub][emf->rd-2] = -0.5/emf->d2;
	  interp->rec_v2[irec][isub][emf->rd] = 0.5/emf->d2;
	  skip2 = true;
	}
	if(fabs(x3j-o3-j3*emf->d3)< eps) {
	  memset(interp->rec_w3[irec][isub], 0, 2*emf->rd*sizeof(float));
	  interp->rec_w3[irec][isub][emf->rd-1] = 1.;
	  memset(interp->rec_v3[irec][isub], 0, 2*emf->rd*sizeof(float));
	  interp->rec_v3[irec][isub][emf->rd-2] = -0.5/emf->d3;
	  interp->rec_v3[irec][isub][emf->rd] = 0.5/emf->d3;
	  skip3 = true;
	}

	  //if(j3+m==i3_bottomwater) ka = m;
	for(i=0; i<2*emf->rd; i++){/* construct vector x1[], x2[], x3[] */
	  m = i-emf->rd+1;/* m=offset/shift between [-emf->rd+1,emf->rd] */
	  if(!skip1) x1[i] = o1+(j1+m)*emf->d1-x1j;
	  if(!skip2) x2[i] = o2+(j2+m)*emf->d2-x2j;
	  if(!skip3) x3[i] = o3+(j3+m)*emf->d3-x3j;
	}/* end for i */
	
	for(i=0; i<2*emf->rd; i++){
	  memset(ff, 0, 2*emf->rd*sizeof(float));
	  ff[i] =  1.;
	  if(!skip1){
	    vandermonde(2*emf->rd-1, x1, aa, ff);
	    interp->rec_w1[irec][isub][i] = aa[0];
	    interp->rec_v1[irec][isub][i] = aa[1];
	  }
	  if(!skip2){
	    vandermonde(2*emf->rd-1, x2, aa, ff);
	    interp->rec_w2[irec][isub][i] = aa[0];
	    interp->rec_v2[irec][isub][i] = aa[1];
	  }
	  if(!skip3){
	    vandermonde(2*emf->rd-1, x3, aa, ff);
	    interp->rec_w3[irec][isub][i] = aa[0];
	    interp->rec_v3[irec][isub][i] = aa[1];
	  }
	}
	
      }/* end for loop */
    }/*end for j */
  }/* end for irec */

  /* ------------ setup interpolation weights for sources ------------*/
  for(isrc=0; isrc< acqui->nsrc; ++isrc){
    tmp = acqui->lensrc/acqui->nsubsrc;
    dlen1 = tmp*cos(acqui->src_dip[isrc]*PI/180.)*cos(acqui->src_azimuth[isrc]*PI/180.);
    dlen2 = tmp*cos(acqui->src_dip[isrc]*PI/180.)*sin(acqui->src_azimuth[isrc]*PI/180.);
    dlen3 = tmp*sin(acqui->src_dip[isrc]*PI/180.);
    for(isub=0; isub<acqui->nsubsrc; ++isub){
      j = isub - acqui->nsubsrc/2;
      x1j = acqui->src_x1[isrc]+j*dlen1;
      x2j = acqui->src_x2[isrc]+j*dlen2;
      x3j = acqui->src_x3[isrc]+j*dlen3;

      for(loop=0; loop<2; loop++){
	if(loop==0){/* regular grid */
	  interp = interp_rg;
	  o1 = acqui->x1min;
	  o2 = acqui->x2min;
	  o3 = acqui->x3min;
	}else{/* staggered grid */
	  interp = interp_sg;
	  o1 = acqui->x1min+0.5*emf->d1;
	  o2 = acqui->x2min+0.5*emf->d2;
	  o3 = acqui->x3min+0.5*emf->d3;
	}
	j1 = (int)((x1j-o1)/emf->d1);/* integer part */
	j2 = (int)((x2j-o2)/emf->d2);/* integer part */
	j3 = (int)((x3j-o3)/emf->d3);/* integer part */

	interp->src_i1[isrc][isub] = j1 + emf->nbe;
	interp->src_i2[isrc][isub] = j2 + emf->nbe;
	interp->src_i3[isrc][isub] = j3 + emf->nbe;

	skip1 = false;
	skip2 = false;
	skip3 = false;
	if(fabs(x1j-o1-j1*emf->d1)< eps){
	  memset(interp->src_w1[isrc][isub], 0, 2*emf->rd*sizeof(float));
	  interp->src_w1[isrc][isub][emf->rd-1] = 1.;
	  skip1 = true;
	}
	if(fabs(x2j-o2-j2*emf->d2)< eps) {
	  memset(interp->src_w2[isrc][isub], 0, 2*emf->rd*sizeof(float));
	  interp->src_w2[isrc][isub][emf->rd-1] = 1.;
	  skip2 = true;
	}
	if(fabs(x3j-o3-j3*emf->d3)< eps) {
	  memset(interp->src_w3[isrc][isub], 0, 2*emf->rd*sizeof(float));
	  interp->src_w3[isrc][isub][emf->rd-1] = 1.;
	  skip3 = true;
	}
	for(i=0; i<2*emf->rd; i++){/* construct vector x1[], x2[], x3[] */
	  m = i-emf->rd+1;/* m=offset/shift between [-emf->rd+1,emf->rd] */
	  if(!skip1) x1[i] = o1+(j1+m)*emf->d1-x1j;
	  if(!skip2) x2[i] = o2+(j2+m)*emf->d2-x2j;
	  if(!skip3) x3[i] = o3+(j3+m)*emf->d3-x3j;
	}

	for(i=0; i<2*emf->rd; i++){
	  memset(ff, 0, 2*emf->rd*sizeof(float));
	  ff[i] =  1.;
	  if(!skip1){
	    vandermonde(2*emf->rd-1, x1, aa, ff);
	    interp->src_w1[isrc][isub][i] = aa[0];
	  }
	  if(!skip2){
	    vandermonde(2*emf->rd-1, x2, aa, ff);
	    interp->src_w2[isrc][isub][i] = aa[0];
	  }
	  if(!skip3){
	    vandermonde(2*emf->rd-1, x3, aa, ff);
	    interp->src_w3[isrc][isub][i] = aa[0];
	  }
	}
      }/* end for loop */

    }/*end for j */
  }/* end for isrc */

  free(x1);
  free(x2);
  free(x3);
  free(aa);
  free(ff);
}
