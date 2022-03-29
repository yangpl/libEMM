/* Nonuniform (NU) grid extended to the whole computing domain
 *--------------------------------------------------------------------
 *
 *   Copyright (c) 2020, Harbin Institute of Technology, China
 *   Author: Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *--------------------------------------------------------------------*/
#include "cstd.h"
#include "emf.h"

void vandermonde(int n, float *x, float *a, float *f);

void nugrid_init(emf_t *emf)
{
  int i3, i, m, loop, istat;
  float *xx, *aa, *ff;
  float dx3_start, dx3_end;
  char *fx3nu;
  FILE *fp;
  
  if(!(getparstring("fx3nu", &fx3nu))) err("Need fx3nu= ");

  emf->x3nu = alloc1float(emf->n3);
  emf->x3n = alloc1float(emf->n3pad);
  emf->x3s = alloc1float(emf->n3pad);

  fp=fopen(fx3nu, "rb");
  if(fp==NULL) err("cannot open fx3nu=%s", fx3nu);
  istat = fread(emf->x3nu, sizeof(float), emf->n3, fp);
  if(istat != emf->n3) err("size parameter does not match the file!");
  fclose(fp);

  emf->d3 = emf->x3nu[1] - emf->x3nu[0];//reset d3
  dx3_start  = emf->x3nu[1]    - emf->x3nu[0];
  dx3_end    = emf->x3nu[emf->n3-1] - emf->x3nu[emf->n3-2];

  for(i3=0; i3<emf->n3; i3++) emf->x3n[i3+emf->nbe] = emf->x3nu[i3];
  for(i3=0; i3<emf->nbe; i3++){
    emf->x3n[i3]      = emf->x3n[emf->nbe]      - (emf->nbe-i3)*dx3_start;
    emf->x3n[emf->nbe+emf->n3+i3] = emf->x3n[emf->nbe+emf->n3-1] + (i3+1)*dx3_end;
  }

  for(i3=0; i3<emf->n3pad-1; i3++) emf->x3s[i3] = 0.5*(emf->x3n[i3] + emf->x3n[i3+1]);
  emf->x3s[emf->n3pad-1] = emf->x3s[emf->n3pad-2] + dx3_end;
  
  emf->u3s = alloc2float(2*emf->rd, emf->n3pad);
  emf->v3 = alloc2float(2*emf->rd, emf->n3pad);
  emf->v3s = alloc2float(2*emf->rd, emf->n3pad);
  memset(&emf->u3s[0][0], 0, 2*emf->rd*emf->n3pad*sizeof(float));
  memset(&emf->v3[0][0], 0, 2*emf->rd*emf->n3pad*sizeof(float));
  memset(&emf->v3s[0][0], 0, 2*emf->rd*emf->n3pad*sizeof(float));

  xx = alloc1float(2*emf->rd);
  aa = alloc1float(2*emf->rd);
  ff = alloc1float(2*emf->rd);

  for(loop=0; loop<2; loop++){
    /*------------------------------------------------------------------*/
    for(i3=emf->rd; i3<emf->n3pad-emf->rd; i3++){
      for(i=0; i<2*emf->rd; i++){/* construct vector x1[], x2[], x3[] */
	m = i-emf->rd;/* m=offset/shift between [-emf->rd+1, emf->rd] */
	if(loop==0) xx[i] = emf->x3s[i3+m]-emf->x3n[i3];//non-shifted point as the center
	else        xx[i] = emf->x3n[i3+1+m]-emf->x3s[i3];//non-shifted point as the center
      }
      for(i=0; i<2*emf->rd; i++){
	//prepare a different rhs for Vandemonde matrix
	memset(ff, 0, 2*emf->rd*sizeof(float));
	ff[i] =  1.;
	vandermonde(2*emf->rd-1, xx, aa, ff);//invert vandemonde matrix system
	if(loop==0) emf->v3[i3][i] = aa[1]; //take coefficients to extract derivatives f'(x)
	else{
	  emf->u3s[i3][i] = aa[0];
	  emf->v3s[i3][i] = aa[1];
	}
      }
    }
  }
  
  free1float(xx);
  free1float(aa);
  free1float(ff);
}

void nugrid_close(emf_t *emf)
{
  free1float(emf->x3nu);
  free1float(emf->x3n);
  free1float(emf->x3s);

  free2float(emf->u3s);
  free2float(emf->v3);
  free2float(emf->v3s);
}


