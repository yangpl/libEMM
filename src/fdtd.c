/* EM modeling using FDTD method
 *------------------------------------------------------------------------
 *
 * Copyright (c) 2021 Harbin Institute of Technology. All rights reserved.
 * Anothr: Pengliang Yang 
 * Email: ypl.2100@gmail.com
 * Homepage: https://yangpl.wordpress.com
 *-----------------------------------------------------------------------*/
#include "cstd.h"
#include "emf.h"
#include "constants.h"

/*----------------------------------------------------------------*/
void fdtd_init(emf_t *emf)
{
  emf->E1 = alloc3float(emf->n1pad, emf->n2pad, emf->n3pad);
  emf->E2 = alloc3float(emf->n1pad, emf->n2pad, emf->n3pad);
  emf->E3 = alloc3float(emf->n1pad, emf->n2pad, emf->n3pad);
  emf->H1 = alloc3float(emf->n1pad, emf->n2pad, emf->n3pad);
  emf->H2 = alloc3float(emf->n1pad, emf->n2pad, emf->n3pad);
  emf->H3 = alloc3float(emf->n1pad, emf->n2pad, emf->n3pad);
  emf->curlE1 = alloc3float(emf->n1pad, emf->n2pad, emf->n3pad);
  emf->curlE2 = alloc3float(emf->n1pad, emf->n2pad, emf->n3pad);
  emf->curlE3 = alloc3float(emf->n1pad, emf->n2pad, emf->n3pad);
  emf->curlH1 = alloc3float(emf->n1pad, emf->n2pad, emf->n3pad);
  emf->curlH2 = alloc3float(emf->n1pad, emf->n2pad, emf->n3pad);
  emf->curlH3 = alloc3float(emf->n1pad, emf->n2pad, emf->n3pad);
  emf->memD2E1 = alloc3float(emf->n1pad, 2*emf->nb, emf->n3pad);
  emf->memD3E1 = alloc3float(emf->n1pad, emf->n2pad, 2*emf->nb);
  emf->memD1E2 = alloc3float(2*emf->nb, emf->n2pad, emf->n3pad);
  emf->memD3E2 = alloc3float(emf->n1pad, emf->n2pad, 2*emf->nb);
  emf->memD1E3 = alloc3float(2*emf->nb, emf->n2pad, emf->n3pad);
  emf->memD2E3 = alloc3float(emf->n1pad, 2*emf->nb, emf->n3pad);
  emf->memD2H1 = alloc3float(emf->n1pad, 2*emf->nb, emf->n3pad);
  emf->memD3H1 = alloc3float(emf->n1pad, emf->n2pad, 2*emf->nb);
  emf->memD1H2 = alloc3float(2*emf->nb, emf->n2pad, emf->n3pad);
  emf->memD3H2 = alloc3float(emf->n1pad, emf->n2pad, 2*emf->nb);
  emf->memD1H3 = alloc3float(2*emf->nb, emf->n2pad, emf->n3pad);
  emf->memD2H3 = alloc3float(emf->n1pad, 2*emf->nb, emf->n3pad);
  
  memset(emf->E1[0][0], 0, emf->n123pad*sizeof(float));
  memset(emf->E2[0][0], 0, emf->n123pad*sizeof(float));
  memset(emf->E3[0][0], 0, emf->n123pad*sizeof(float));
  memset(emf->H1[0][0], 0, emf->n123pad*sizeof(float));
  memset(emf->H2[0][0], 0, emf->n123pad*sizeof(float));
  memset(emf->H3[0][0], 0, emf->n123pad*sizeof(float));
  memset(emf->curlE1[0][0], 0, emf->n123pad*sizeof(float));
  memset(emf->curlE2[0][0], 0, emf->n123pad*sizeof(float));
  memset(emf->curlE3[0][0], 0, emf->n123pad*sizeof(float));
  memset(emf->curlH1[0][0], 0, emf->n123pad*sizeof(float));
  memset(emf->curlH2[0][0], 0, emf->n123pad*sizeof(float));
  memset(emf->curlH3[0][0], 0, emf->n123pad*sizeof(float));
  memset(emf->memD2E1[0][0], 0, 2*emf->nb*emf->n1pad*emf->n3pad*sizeof(float));
  memset(emf->memD3E1[0][0], 0, 2*emf->nb*emf->n1pad*emf->n2pad*sizeof(float));
  memset(emf->memD1E2[0][0], 0, 2*emf->nb*emf->n2pad*emf->n3pad*sizeof(float));
  memset(emf->memD3E2[0][0], 0, 2*emf->nb*emf->n1pad*emf->n2pad*sizeof(float));
  memset(emf->memD1E3[0][0], 0, 2*emf->nb*emf->n2pad*emf->n3pad*sizeof(float));
  memset(emf->memD2E3[0][0], 0, 2*emf->nb*emf->n1pad*emf->n3pad*sizeof(float));
  memset(emf->memD2H1[0][0], 0, 2*emf->nb*emf->n1pad*emf->n3pad*sizeof(float));
  memset(emf->memD3H1[0][0], 0, 2*emf->nb*emf->n1pad*emf->n2pad*sizeof(float));
  memset(emf->memD1H2[0][0], 0, 2*emf->nb*emf->n2pad*emf->n3pad*sizeof(float));
  memset(emf->memD3H2[0][0], 0, 2*emf->nb*emf->n1pad*emf->n2pad*sizeof(float));
  memset(emf->memD1H3[0][0], 0, 2*emf->nb*emf->n2pad*emf->n3pad*sizeof(float));
  memset(emf->memD2H3[0][0], 0, 2*emf->nb*emf->n1pad*emf->n3pad*sizeof(float));

}

void fdtd_close(emf_t *emf)
{
  
  free3float(emf->E1);
  free3float(emf->E2);
  free3float(emf->E3);
  free3float(emf->H1);
  free3float(emf->H2);
  free3float(emf->H3);
  free3float(emf->curlE1);
  free3float(emf->curlE2);
  free3float(emf->curlE3);
  free3float(emf->curlH1);
  free3float(emf->curlH2);
  free3float(emf->curlH3);
  free3float(emf->memD2E1);
  free3float(emf->memD3E1);
  free3float(emf->memD1E2);
  free3float(emf->memD3E2);
  free3float(emf->memD1E3);
  free3float(emf->memD2E3);
  free3float(emf->memD2H1);
  free3float(emf->memD3H1);
  free3float(emf->memD1H2);
  free3float(emf->memD3H2);
  free3float(emf->memD1H3);
  free3float(emf->memD2H3);
}

void fdtd_curlH(emf_t *emf, int it, int adj)
{
  int i1, i2, i3, j1, j2, j3, k1, k2, k3;
  float D2H3, D3H2, D3H1, D1H3, D1H2, D2H1;
  float c11, c21, c31, c12, c22, c32, c13, c23, c33;

  int i1min = emf->rd;
  int i2min = emf->rd;
  int i3min = emf->airwave?emf->nbe:emf->rd;
  int i1max = emf->n1pad-emf->rd;
  int i2max = emf->n2pad-emf->rd;
  int i3max = emf->n3pad-emf->rd;

  c11 = 0;
  c21 = 0;
  c31 = 0;
  c12 = 0;
  c22 = 0;
  c32 = 0;
  c13 = 0;
  c23 = 0;
  c33 = 0;
  if(emf->rd==1){
    c11 = 1./emf->d1;
    c12 = 1./emf->d2;
    c13 = 1./emf->d3;
  }else if(emf->rd==2){
    c11 = 1.125/emf->d1;
    c21 = -0.041666666666666664/emf->d1;
    c12 = 1.125/emf->d2;
    c22 = -0.041666666666666664/emf->d2;
    c13 = 1.125/emf->d3;
    c23 = -0.041666666666666664/emf->d3;
  }else if(emf->rd==3){
    c11 = 1.171875/emf->d1;
    c21 = -0.065104166666667/emf->d1;
    c31 = 0.0046875/emf->d1;
    c12 = 1.171875/emf->d2;
    c22 = -0.065104166666667/emf->d2;
    c32 = 0.0046875/emf->d2;
    c13 = 1.171875/emf->d3;
    c23 = -0.065104166666667/emf->d3;
    c33 = 0.0046875/emf->d3;
  }

#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3, D2H3, D3H2, D3H1, D1H3, D1H2, D2H1)	\
  shared(i1min, i1max, i2min, i2max, i3min, i3max, c11, c21, c31, c12, c22, c32, c13, c23, c33, emf)
#endif
  for(i3=i3min; i3<=i3max; ++i3){
    for(i2=i2min; i2<=i2max; ++i2){
      for(i1=i1min; i1<=i1max; ++i1){
	D1H3 = 0;
	D1H2 = 0;
	D2H3 = 0;
	D2H1 = 0;
	D3H2 = 0;
	D3H1 = 0;

	if(emf->rd==1){
	  D1H3 = c11*(emf->H3[i3][i2][i1]-emf->H3[i3][i2][i1-1]);
	  D1H2 = c11*(emf->H2[i3][i2][i1]-emf->H2[i3][i2][i1-1]);
	}else if(emf->rd==2){
	  D1H3 = c11*(emf->H3[i3][i2][i1]-emf->H3[i3][i2][i1-1])
	    + c21*(emf->H3[i3][i2][i1+1]-emf->H3[i3][i2][i1-2]);
	  D1H2 = c11*(emf->H2[i3][i2][i1]-emf->H2[i3][i2][i1-1])
	    + c21*(emf->H2[i3][i2][i1+1]-emf->H2[i3][i2][i1-2]);
	}else if(emf->rd==3){
	  D1H3 = c11*(emf->H3[i3][i2][i1]-emf->H3[i3][i2][i1-1])
	    + c21*(emf->H3[i3][i2][i1+1]-emf->H3[i3][i2][i1-2])
	    + c31*(emf->H3[i3][i2][i1+2]-emf->H3[i3][i2][i1-3]);
	  D1H2 = c11*(emf->H2[i3][i2][i1]-emf->H2[i3][i2][i1-1])
	    + c21*(emf->H2[i3][i2][i1+1]-emf->H2[i3][i2][i1-2])
	    + c31*(emf->H2[i3][i2][i1+2]-emf->H2[i3][i2][i1-3]);
	}

	if(emf->rd==1){
	  D2H3 = c12*(emf->H3[i3][i2][i1]-emf->H3[i3][i2-1][i1]);
	  D2H1 = c12*(emf->H1[i3][i2][i1]-emf->H1[i3][i2-1][i1]);
	}else if(emf->rd==2){
	  D2H3 = c12*(emf->H3[i3][i2][i1]-emf->H3[i3][i2-1][i1])
	    + c22*(emf->H3[i3][i2+1][i1]-emf->H3[i3][i2-2][i1]);
	  D2H1 = c12*(emf->H1[i3][i2][i1]-emf->H1[i3][i2-1][i1])
	    + c22*(emf->H1[i3][i2+1][i1]-emf->H1[i3][i2-2][i1]);
	}else if(emf->rd==3){
	  D2H3 = c12*(emf->H3[i3][i2][i1]-emf->H3[i3][i2-1][i1])
	    + c22*(emf->H3[i3][i2+1][i1]-emf->H3[i3][i2-2][i1])
	    + c32*(emf->H3[i3][i2+2][i1]-emf->H3[i3][i2-3][i1]);
	  D2H1 = c12*(emf->H1[i3][i2][i1]-emf->H1[i3][i2-1][i1])
	    + c22*(emf->H1[i3][i2+1][i1]-emf->H1[i3][i2-2][i1])
	    + c32*(emf->H1[i3][i2+2][i1]-emf->H1[i3][i2-3][i1]);
	}

	if(emf->nugrid){
	  if(emf->rd==1){
	    D3H2 = emf->v3s[i3][0]*emf->H2[i3-1][i2][i1]
	      + emf->v3s[i3][1]*emf->H2[i3][i2][i1];
	    D3H1 = emf->v3s[i3][0]*emf->H1[i3-1][i2][i1]
	      + emf->v3s[i3][1]*emf->H1[i3][i2][i1];
	  }else if(emf->rd==2){
	    D3H2 = emf->v3s[i3][0]*emf->H2[i3-2][i2][i1]
	      + emf->v3s[i3][1]*emf->H2[i3-1][i2][i1]
	      + emf->v3s[i3][2]*emf->H2[i3][i2][i1]
	      + emf->v3s[i3][3]*emf->H2[i3+1][i2][i1];
	    D3H1 = emf->v3s[i3][0]*emf->H1[i3-2][i2][i1]
	      + emf->v3s[i3][1]*emf->H1[i3-1][i2][i1]
	      + emf->v3s[i3][2]*emf->H1[i3][i2][i1]
	      + emf->v3s[i3][3]*emf->H1[i3+1][i2][i1];
	  }else if(emf->rd==3){
	    D3H2 = emf->v3s[i3][0]*emf->H2[i3-3][i2][i1]
	      + emf->v3s[i3][1]*emf->H2[i3-2][i2][i1]
	      + emf->v3s[i3][2]*emf->H2[i3-1][i2][i1]
	      + emf->v3s[i3][3]*emf->H2[i3][i2][i1]
	      + emf->v3s[i3][4]*emf->H2[i3+1][i2][i1]
	      + emf->v3s[i3][5]*emf->H2[i3+2][i2][i1];
	    D3H1 = emf->v3s[i3][0]*emf->H1[i3-3][i2][i1]
	      + emf->v3s[i3][1]*emf->H1[i3-2][i2][i1]
	      + emf->v3s[i3][2]*emf->H1[i3-1][i2][i1]
	      + emf->v3s[i3][3]*emf->H1[i3][i2][i1]
	      + emf->v3s[i3][4]*emf->H1[i3+1][i2][i1]
	      + emf->v3s[i3][5]*emf->H1[i3+2][i2][i1];
	  }

	}else{
	  if(emf->rd==1){
	    D3H2 = c13*(emf->H2[i3][i2][i1]-emf->H2[i3-1][i2][i1]);
	    D3H1 = c13*(emf->H1[i3][i2][i1]-emf->H1[i3-1][i2][i1]);
	  }else if(emf->rd==2){
	    D3H2 = c13*(emf->H2[i3][i2][i1]-emf->H2[i3-1][i2][i1])
	      + c23*(emf->H2[i3+1][i2][i1]-emf->H2[i3-2][i2][i1]);
	    D3H1 = c13*(emf->H1[i3][i2][i1]-emf->H1[i3-1][i2][i1])
	      + c23*(emf->H1[i3+1][i2][i1]-emf->H1[i3-2][i2][i1]);
	  }else if(emf->rd==3){
	    D3H2 = c13*(emf->H2[i3][i2][i1]-emf->H2[i3-1][i2][i1])
	      + c23*(emf->H2[i3+1][i2][i1]-emf->H2[i3-2][i2][i1])
	      + c33*(emf->H2[i3+2][i2][i1]-emf->H2[i3-3][i2][i1]);
	    D3H1 = c13*(emf->H1[i3][i2][i1]-emf->H1[i3-1][i2][i1])
	      + c23*(emf->H1[i3+1][i2][i1]-emf->H1[i3-2][i2][i1])
	      + c33*(emf->H1[i3+2][i2][i1]-emf->H1[i3-3][i2][i1]);
	  }
	}

	/* CPML: mem=memory variable */
	if(i1<emf->nb){
	  emf->memD1H3[i3][i2][i1] = emf->bpml[i1]*emf->memD1H3[i3][i2][i1] + emf->apml[i1]*D1H3;
	  emf->memD1H2[i3][i2][i1] = emf->bpml[i1]*emf->memD1H2[i3][i2][i1] + emf->apml[i1]*D1H2;
	  D1H3 += emf->memD1H3[i3][i2][i1];
	  D1H2 += emf->memD1H2[i3][i2][i1];
	}else if(i1>emf->n1pad-1-emf->nb){
	  j1 = emf->n1pad-1-i1;
	  k1 = j1+emf->nb;
	  emf->memD1H3[i3][i2][k1] = emf->bpml[j1]*emf->memD1H3[i3][i2][k1] + emf->apml[j1]*D1H3;
	  emf->memD1H2[i3][i2][k1] = emf->bpml[j1]*emf->memD1H2[i3][i2][k1] + emf->apml[j1]*D1H2;
	  D1H3 += emf->memD1H3[i3][i2][k1];
	  D1H2 += emf->memD1H2[i3][i2][k1];
	}
	if(i2<emf->nb){
	  emf->memD2H3[i3][i2][i1] = emf->bpml[i2]*emf->memD2H3[i3][i2][i1] + emf->apml[i2]*D2H3;
	  emf->memD2H1[i3][i2][i1] = emf->bpml[i2]*emf->memD2H1[i3][i2][i1] + emf->apml[i2]*D2H1;
	  D2H3 += emf->memD2H3[i3][i2][i1];
	  D2H1 += emf->memD2H1[i3][i2][i1];
	}else if(i2>emf->n2pad-1-emf->nb){
	  j2 = emf->n2pad-1-i2;
	  k2 = j2+emf->nb;
	  emf->memD2H3[i3][k2][i1] = emf->bpml[j2]*emf->memD2H3[i3][k2][i1] + emf->apml[j2]*D2H3;
	  emf->memD2H1[i3][k2][i1] = emf->bpml[j2]*emf->memD2H1[i3][k2][i1] + emf->apml[j2]*D2H1;
	  D2H3 += emf->memD2H3[i3][k2][i1];
	  D2H1 += emf->memD2H1[i3][k2][i1];
	}
	if(i3<emf->nb){
	  emf->memD3H2[i3][i2][i1] = emf->bpml[i3]*emf->memD3H2[i3][i2][i1] + emf->apml[i3]*D3H2;
	  emf->memD3H1[i3][i2][i1] = emf->bpml[i3]*emf->memD3H1[i3][i2][i1] + emf->apml[i3]*D3H1;
	  D3H2 += emf->memD3H2[i3][i2][i1];
	  D3H1 += emf->memD3H1[i3][i2][i1];
	}else if(i3>emf->n3pad-1-emf->nb){
	  j3 = emf->n3pad-1-i3;
	  k3 = j3+emf->nb;
	  emf->memD3H2[k3][i2][i1] = emf->bpml[j3]*emf->memD3H2[k3][i2][i1] + emf->apml[j3]*D3H2;
	  emf->memD3H1[k3][i2][i1] = emf->bpml[j3]*emf->memD3H1[k3][i2][i1] + emf->apml[j3]*D3H1;
	  D3H2 += emf->memD3H2[k3][i2][i1];
	  D3H1 += emf->memD3H1[k3][i2][i1];
	}

	emf->curlH1[i3][i2][i1] = D2H3-D3H2;
	emf->curlH2[i3][i2][i1] = D3H1-D1H3;
	emf->curlH3[i3][i2][i1] = D1H2-D2H1;
      }
    }
  }
  
}


void fdtd_update_E(emf_t *emf, int it, int adj)
{
  int i1, i2, i3;

  int i1min = 0;
  int i2min = 0;
  int i3min = emf->airwave?emf->nbe:0;
  int i1max = emf->n1pad-1;
  int i2max = emf->n2pad-1;
  int i3max = emf->n3pad-1;

#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3)							\
  shared(i1min, i1max, i2min, i2max, i3min, i3max, emf)
#endif
  for(i3=i3min; i3<=i3max; ++i3){
    for(i2=i2min; i2<=i2max; ++i2){
      for(i1=i1min; i1<=i1max; ++i1){
	/* E1=Ex[i1+0.5, i2, i3]; emf->inveps11=emf->inveps11[i1+0.5, i2, i3] */
	emf->E1[i3][i2][i1] += emf->dt*emf->inveps11[i3][i2][i1]* emf->curlH1[i3][i2][i1];
	/* emf->E2=emf->Ey[i1, i2+0.5, i3]; emf->inveps11=emf->inveps22[i1, i2+0.5, i3] */
	emf->E2[i3][i2][i1] += emf->dt*emf->inveps22[i3][i2][i1]* emf->curlH2[i3][i2][i1];
	/* emf->E3=emf->Ez[i1, i2, i3+0.5]; emf->inveps11=emf->inveps33[i1, i2, i3+0.5] */
	emf->E3[i3][i2][i1] += emf->dt*emf->inveps33[i3][i2][i1]* emf->curlH3[i3][i2][i1];
      }
    }
  }
}

void fdtd_curlE(emf_t *emf, int it, int adj)
{
  int i1, i2, i3, j1, j2, j3, k1, k2, k3;
  float D2E3, D3E2, D3E1, D1E3, D1E2, D2E1;
  float c11, c21, c31, c12, c22, c32, c13, c23, c33;

  int i1min = emf->rd-1;
  int i2min = emf->rd-1;
  int i3min = emf->airwave?emf->nbe:emf->rd-1;
  int i1max = emf->n1pad-1-emf->rd;
  int i2max = emf->n2pad-1-emf->rd;
  int i3max = emf->n3pad-1-emf->rd;
  
  c11 = 0;
  c21 = 0;
  c31 = 0;
  c12 = 0;
  c22 = 0;
  c32 = 0;
  c13 = 0;
  c23 = 0;
  c33 = 0;
  if(emf->rd==1){
    c11 = 1./emf->d1;
    c12 = 1./emf->d2;
    c13 = 1./emf->d3;
  }else if(emf->rd==2){
    c11 = 1.125/emf->d1;
    c21 = -0.041666666666666664/emf->d1;
    c12 = 1.125/emf->d2;
    c22 = -0.041666666666666664/emf->d2;
    c13 = 1.125/emf->d3;
    c23 = -0.041666666666666664/emf->d3;
  }else if(emf->rd==3){
    c11 = 1.171875/emf->d1;
    c21 = -0.065104166666667/emf->d1;
    c31 = 0.0046875/emf->d1;
    c12 = 1.171875/emf->d2;
    c22 = -0.065104166666667/emf->d2;
    c32 = 0.0046875/emf->d2;
    c13 = 1.171875/emf->d3;
    c23 = -0.065104166666667/emf->d3;
    c33 = 0.0046875/emf->d3;
  }

#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3, D2E3, D3E2, D3E1, D1E3, D1E2, D2E1)	\
  shared(i1min, i1max, i2min, i2max, i3min, i3max, c11, c21, c31, c12, c22, c32, c13, c23, c33, emf)
#endif
  for(i3=i3min; i3<=i3max; ++i3){
    for(i2=i2min; i2<=i2max; ++i2){
      for(i1=i1min; i1<=i1max; ++i1){
	D1E3 = 0;
	D1E2 = 0;
	D2E3 = 0;
	D2E1 = 0;
	D3E2 = 0;
	D3E1 = 0;

	if(emf->rd==1){
	  D1E3 = c11*(emf->E3[i3][i2][i1+1]-emf->E3[i3][i2][i1]);
	  D1E2 = c11*(emf->E2[i3][i2][i1+1]-emf->E2[i3][i2][i1]);
	}else if(emf->rd==2){
	  D1E3 = c11*(emf->E3[i3][i2][i1+1]-emf->E3[i3][i2][i1])
	    + c21*(emf->E3[i3][i2][i1+2]-emf->E3[i3][i2][i1-1]);
	  D1E2 = c11*(emf->E2[i3][i2][i1+1]-emf->E2[i3][i2][i1])
	    + c21*(emf->E2[i3][i2][i1+2]-emf->E2[i3][i2][i1-1]);
	}else if(emf->rd==3){
	  D1E3 = c11*(emf->E3[i3][i2][i1+1]-emf->E3[i3][i2][i1])
	    + c21*(emf->E3[i3][i2][i1+2]-emf->E3[i3][i2][i1-1])
	    + c31*(emf->E3[i3][i2][i1+3]-emf->E3[i3][i2][i1-2]);
	  D1E2 = c11*(emf->E2[i3][i2][i1+1]-emf->E2[i3][i2][i1])
	    + c21*(emf->E2[i3][i2][i1+2]-emf->E2[i3][i2][i1-1])
	    + c31*(emf->E2[i3][i2][i1+3]-emf->E2[i3][i2][i1-2]);
	}

	if(emf->rd==1){
	  D2E3 = c12*(emf->E3[i3][i2+1][i1]-emf->E3[i3][i2][i1]);
	  D2E1 = c12*(emf->E1[i3][i2+1][i1]-emf->E1[i3][i2][i1]);
	}else if(emf->rd==2){
	  D2E3 = c12*(emf->E3[i3][i2+1][i1]-emf->E3[i3][i2][i1])
	    + c22*(emf->E3[i3][i2+2][i1]-emf->E3[i3][i2-1][i1]);
	  D2E1 = c12*(emf->E1[i3][i2+1][i1]-emf->E1[i3][i2][i1])
	    + c22*(emf->E1[i3][i2+2][i1]-emf->E1[i3][i2-1][i1]);
	}else if(emf->rd==3){
	  D2E3 = c12*(emf->E3[i3][i2+1][i1]-emf->E3[i3][i2][i1])
	    + c22*(emf->E3[i3][i2+2][i1]-emf->E3[i3][i2-1][i1])
	    + c32*(emf->E3[i3][i2+3][i1]-emf->E3[i3][i2-2][i1]);
	  D2E1 = c12*(emf->E1[i3][i2+1][i1]-emf->E1[i3][i2][i1])
	    + c22*(emf->E1[i3][i2+2][i1]-emf->E1[i3][i2-1][i1])
	    + c32*(emf->E1[i3][i2+3][i1]-emf->E1[i3][i2-2][i1]);
	}

	if(emf->nugrid){
	  if(emf->rd==1){
	    D3E2 = emf->v3[i3][0]*emf->E2[i3][i2][i1]
	      + emf->v3[i3][1]*emf->E2[i3+1][i2][i1];
	    D3E1 = emf->v3[i3][0]*emf->E1[i3][i2][i1]
	      + emf->v3[i3][1]*emf->E1[i3+1][i2][i1];
	  }else if(emf->rd==2){
	    D3E2 = emf->v3[i3][0]*emf->E2[i3-1][i2][i1]
	      + emf->v3[i3][1]*emf->E2[i3][i2][i1]
	      + emf->v3[i3][2]*emf->E2[i3+1][i2][i1]
	      + emf->v3[i3][3]*emf->E2[i3+2][i2][i1];
	    D3E1 = emf->v3[i3][0]*emf->E1[i3-1][i2][i1]
	      + emf->v3[i3][1]*emf->E1[i3][i2][i1]
	      + emf->v3[i3][2]*emf->E1[i3+1][i2][i1]
	      + emf->v3[i3][3]*emf->E1[i3+2][i2][i1];
	  }else if(emf->rd==3){
	    D3E2 = emf->v3[i3][0]*emf->E2[i3-2][i2][i1]
	      + emf->v3[i3][1]*emf->E2[i3-1][i2][i1]
	      + emf->v3[i3][2]*emf->E2[i3][i2][i1]
	      + emf->v3[i3][3]*emf->E2[i3+1][i2][i1]
	      + emf->v3[i3][4]*emf->E2[i3+2][i2][i1]
	      + emf->v3[i3][5]*emf->E2[i3+3][i2][i1];
	    D3E1 = emf->v3[i3][0]*emf->E1[i3-2][i2][i1]
	      + emf->v3[i3][1]*emf->E1[i3-1][i2][i1]
	      + emf->v3[i3][2]*emf->E1[i3][i2][i1]
	      + emf->v3[i3][3]*emf->E1[i3+1][i2][i1]
	      + emf->v3[i3][4]*emf->E1[i3+2][i2][i1]
	      + emf->v3[i3][5]*emf->E1[i3+3][i2][i1];
	  }

	}else{
	  if(emf->rd==1){
	    D3E2 = c13*(emf->E2[i3+1][i2][i1]-emf->E2[i3][i2][i1]);
	    D3E1 = c13*(emf->E1[i3+1][i2][i1]-emf->E1[i3][i2][i1]);
	  }else if(emf->rd==2){
	    D3E2 = c13*(emf->E2[i3+1][i2][i1]-emf->E2[i3][i2][i1])
	      + c23*(emf->E2[i3+2][i2][i1]-emf->E2[i3-1][i2][i1]);
	    D3E1 = c13*(emf->E1[i3+1][i2][i1]-emf->E1[i3][i2][i1])
	      + c23*(emf->E1[i3+2][i2][i1]-emf->E1[i3-1][i2][i1]);
	  }else if(emf->rd==3){
	    D3E2 = c13*(emf->E2[i3+1][i2][i1]-emf->E2[i3][i2][i1])
	      + c23*(emf->E2[i3+2][i2][i1]-emf->E2[i3-1][i2][i1])
	      + c33*(emf->E2[i3+3][i2][i1]-emf->E2[i3-2][i2][i1]);
	    D3E1 = c13*(emf->E1[i3+1][i2][i1]-emf->E1[i3][i2][i1])
	      + c23*(emf->E1[i3+2][i2][i1]-emf->E1[i3-1][i2][i1])
	      + c33*(emf->E1[i3+3][i2][i1]-emf->E1[i3-2][i2][i1]);
	  }

	}

	/* CPML: mem=memory variable */
	if(i1<emf->nb){
	  emf->memD1E3[i3][i2][i1] = emf->bpml[i1]*emf->memD1E3[i3][i2][i1] + emf->apml[i1]*D1E3;
	  emf->memD1E2[i3][i2][i1] = emf->bpml[i1]*emf->memD1E2[i3][i2][i1] + emf->apml[i1]*D1E2;
	  D1E3 += emf->memD1E3[i3][i2][i1];
	  D1E2 += emf->memD1E2[i3][i2][i1];
	}else if(i1>emf->n1pad-1-emf->nb){
	  j1 = emf->n1pad-1-i1;
	  k1 = j1+emf->nb;
	  emf->memD1E3[i3][i2][k1] = emf->bpml[j1]*emf->memD1E3[i3][i2][k1] + emf->apml[j1]*D1E3;
	  emf->memD1E2[i3][i2][k1] = emf->bpml[j1]*emf->memD1E2[i3][i2][k1] + emf->apml[j1]*D1E2;
	  D1E3 += emf->memD1E3[i3][i2][k1];
	  D1E2 += emf->memD1E2[i3][i2][k1];
	}
	if(i2<emf->nb){
	  emf->memD2E3[i3][i2][i1] = emf->bpml[i2]*emf->memD2E3[i3][i2][i1] + emf->apml[i2]*D2E3;
	  emf->memD2E1[i3][i2][i1] = emf->bpml[i2]*emf->memD2E1[i3][i2][i1] + emf->apml[i2]*D2E1;
	  D2E3 += emf->memD2E3[i3][i2][i1];
	  D2E1 += emf->memD2E1[i3][i2][i1];
	}else if(i2>emf->n2pad-1-emf->nb){
	  j2 = emf->n2pad-1-i2;
	  k2 = j2+emf->nb;
	  emf->memD2E3[i3][k2][i1] = emf->bpml[j2]*emf->memD2E3[i3][k2][i1] + emf->apml[j2]*D2E3;
	  emf->memD2E1[i3][k2][i1] = emf->bpml[j2]*emf->memD2E1[i3][k2][i1] + emf->apml[j2]*D2E1;
	  D2E3 += emf->memD2E3[i3][k2][i1];
	  D2E1 += emf->memD2E1[i3][k2][i1];
	}
	if(i3<emf->nb){
	  emf->memD3E2[i3][i2][i1] = emf->bpml[i3]*emf->memD3E2[i3][i2][i1] + emf->apml[i3]*D3E2;
	  emf->memD3E1[i3][i2][i1] = emf->bpml[i3]*emf->memD3E1[i3][i2][i1] + emf->apml[i3]*D3E1;
	  D3E2 += emf->memD3E2[i3][i2][i1];
	  D3E1 += emf->memD3E1[i3][i2][i1];
	}else if(i3>emf->n3pad-1-emf->nb){
	  j3 = emf->n3pad-1-i3;
	  k3 = j3+emf->nb;
	  emf->memD3E2[k3][i2][i1] = emf->bpml[j3]*emf->memD3E2[k3][i2][i1] + emf->apml[j3]*D3E2;
	  emf->memD3E1[k3][i2][i1] = emf->bpml[j3]*emf->memD3E1[k3][i2][i1] + emf->apml[j3]*D3E1;
	  D3E2 += emf->memD3E2[k3][i2][i1];
	  D3E1 += emf->memD3E1[k3][i2][i1];
	}

	emf->curlE1[i3][i2][i1] = D2E3-D3E2;
	emf->curlE2[i3][i2][i1] = D3E1-D1E3;
	emf->curlE3[i3][i2][i1] = D1E2-D2E1;
      }
    }
  }
}



void fdtd_update_H(emf_t *emf, int it, int adj)
{
  int i1, i2, i3;

  int i1min=0;
  int i2min=0;
  int i3min=emf->airwave?emf->nbe:0;
  int i1max=emf->n1pad-1;
  int i2max=emf->n2pad-1;
  int i3max=emf->n3pad-1;
  
  float factor = emf->dt*invmu0;
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3)							\
  shared(i1min, i1max, i2min, i2max, i3min, i3max, emf, factor)
#endif
  for(i3=i3min; i3<=i3max; ++i3){
    for(i2=i2min; i2<=i2max; ++i2){
      for(i1=i1min; i1<=i1max; ++i1){
	emf->H1[i3][i2][i1] -= factor* emf->curlE1[i3][i2][i1];
	emf->H2[i3][i2][i1] -= factor* emf->curlE2[i3][i2][i1];
	emf->H3[i3][i2][i1] -= factor* emf->curlE3[i3][i2][i1];
      }
    }
  }
}

