/* implement boundary condition for air-water interface 
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

#include <fftw3.h>

fftw_complex *emf_kxky, *emf_kxkyz0;
fftw_plan fft_airwave, ifft_airwave;


int fft_next_fast_size(int n)
{
  int m,n2;
  /* m = 1; */
  /* while(m<=n) m *= 2; */
  /* return 2*m; */

  n2 = 4*n;
  while(1) {
    m=n2;
    while ( (m%2) == 0 ) m/=2;
    while ( (m%3) == 0 ) m/=3;
    while ( (m%5) == 0 ) m/=5;
    if (m<=1)
      break; /* n is completely factorable by twos, threes, and fives */
    n2++;
  }
  return n2;
}

/*--------------------------------------------------------------------------*/
void airwave_bc_init(emf_t *emf)
{
  double dkx, dky, tmp;
  int i1, i2, i3;
  float _Complex tmp1, tmp2;
  float *kx, *ky;
  float **sqrtkx2ky2;
  
  emf->n1fft = fft_next_fast_size(emf->n1pad);
  emf->n2fft = fft_next_fast_size(emf->n2pad);

  /* complex scaling factor for H1, H2, E1, E2 */
  emf->sH1kxky = alloc3complexf(emf->n1fft, emf->n2fft, emf->rd);
  emf->sH2kxky = alloc3complexf(emf->n1fft, emf->n2fft, emf->rd);
  if(emf->rd>1) emf->sE12kxky = alloc3float(emf->n1fft, emf->n2fft, emf->rd-1);
  
  /* FE3 is not necessary in the air because we do not compute derivates of Hx
   * and Hy in the air: Hx and Hy are derived directly by extrapolation from Hz. */
  emf_kxky=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*emf->n1fft*emf->n2fft);
  emf_kxkyz0=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*emf->n1fft*emf->n2fft);
  /* comapred with FFTW, we have opposite sign convention for time, same sign convetion for space */
  fft_airwave=fftw_plan_dft_2d(emf->n1fft, emf->n2fft, emf_kxky, emf_kxky, FFTW_FORWARD, FFTW_ESTIMATE);
  ifft_airwave=fftw_plan_dft_2d(emf->n1fft, emf->n2fft, emf_kxky, emf_kxky, FFTW_BACKWARD, FFTW_ESTIMATE);  
  
  kx = alloc1float(emf->n1fft);
  ky = alloc1float(emf->n2fft);
  sqrtkx2ky2 = alloc2float(emf->n1fft, emf->n2fft);
  /* pre-compute the discrete wavenumber - kx */
  dkx=2.0*PI/(emf->d1*emf->n1fft);
  kx[0]=0;
  for(i1=1; i1<(emf->n1fft+1)/2; i1++) {
    kx[i1]=i1*dkx;
    kx[emf->n1fft-i1]=-i1*dkx;
  }
  if(emf->n1fft%2==0) kx[emf->n1fft/2] = (emf->n1fft/2)*dkx;/* Nyquist freq*/

  /* pre-compute the discrete wavenumber - ky */
  dky=2.0*PI/(emf->d2*emf->n2fft);
  ky[0]=0;
  for(i2=1; i2<(emf->n2fft+1)/2; i2++) {
    ky[i2]=i2*dky;
    ky[emf->n2fft-i2]=-i2*dky;
  }
  if(emf->n2fft%2==0) ky[emf->n2fft/2] = (emf->n2fft/2)*dky;/* Nyquist freq*/
  
  for(i2=0; i2<emf->n2fft; i2++){
    for(i1=0; i1<emf->n1fft; i1++){
     sqrtkx2ky2[i2][i1] = sqrt(kx[i1]*kx[i1]+ky[i2]*ky[i2]);
    }
  }

  for(i3=0; i3<emf->rd; i3++){
    for(i2=0; i2<emf->n2fft; i2++){
      for(i1=0; i1<emf->n1fft; i1++){
	tmp1 = cexp(-I*kx[i1]*0.5*emf->d1)*I*kx[i1]/(sqrtkx2ky2[i2][i1]+1.e-15);
	tmp2 = cexp(-I*ky[i2]*0.5*emf->d2)*I*ky[i2]/(sqrtkx2ky2[i2][i1]+1.e-15);
	tmp = exp(-sqrtkx2ky2[i2][i1]*(i3+0.5)*emf->d3);
	emf->sH1kxky[i3][i2][i1] = tmp*tmp1;
	emf->sH2kxky[i3][i2][i1] = tmp*tmp2;
      }
    }
  }
  for(i3=0; i3<emf->rd-1; i3++){
    for(i2=0; i2<emf->n2fft; i2++){
      for(i1=0; i1<emf->n1fft; i1++){
	emf->sE12kxky[i3][i2][i1] =  exp(-sqrtkx2ky2[i2][i1]*(i3+1)*emf->d3);
      }
    }
  }
 
  free(kx);
  free(ky);
  free2float(sqrtkx2ky2);
 
}

void airwave_bc_close(emf_t *emf)
{
  free3complexf(emf->sH1kxky);
  free3complexf(emf->sH2kxky);
  if(emf->rd>1) free3float(emf->sE12kxky);

  fftw_free(emf_kxky);
  fftw_free(emf_kxkyz0);
  fftw_destroy_plan(fft_airwave);
  fftw_destroy_plan(ifft_airwave);  
}

void airwave_bc_update_H(emf_t *emf)
{
  int i1, i2, i3;
  
  for(i2=0; i2<emf->n2fft; i2++){
    for(i1=0; i1<emf->n1fft; i1++){
      if(i1<emf->n1pad && i2<emf->n2pad)
	emf_kxky[i1+emf->n1fft*i2] = emf->H3[emf->nbe][i2][i1] + I*0.; 
      else
	emf_kxky[i1+emf->n1fft*i2] = 0. + I*0.;
    }
  }
  fftw_execute(fft_airwave);/*Hz(x,y,z=0)-->Hz(kx,ky,z=0)*/
  memcpy(emf_kxkyz0, emf_kxky, emf->n1fft*emf->n2fft*sizeof(fftw_complex));

  /*----------------------------------- H1 -------------------------------------*/
  for(i3=0; i3<emf->rd; i3++){
    for(i2=0; i2<emf->n2fft; i2++){
      for(i1=0; i1<emf->n1fft; i1++){
	emf_kxky[i1+emf->n1fft*i2] = emf_kxkyz0[i1+emf->n1fft*i2]*emf->sH1kxky[i3][i2][i1];
      }
    }
    fftw_execute(ifft_airwave);
    for(i2=0; i2<emf->n2pad; i2++){
      for(i1=0; i1<emf->n1pad; i1++){
	emf->H1[emf->nbe-1-i3][i2][i1] = creal(emf_kxky[i1+emf->n1fft*i2]/(emf->n1fft*emf->n2fft));
      }
    }
    
  }

  /*----------------------------------- H2 -------------------------------------*/
  for(i3=0; i3<emf->rd; i3++){
    for(i2=0; i2<emf->n2fft; i2++){
      for(i1=0; i1<emf->n1fft; i1++){
	emf_kxky[i1+emf->n1fft*i2] = emf_kxkyz0[i1+emf->n1fft*i2]*emf->sH2kxky[i3][i2][i1];
      }
    }
    fftw_execute(ifft_airwave);
    for(i2=0; i2<emf->n2pad; i2++){
      for(i1=0; i1<emf->n1pad; i1++){
	emf->H2[emf->nbe-1-i3][i2][i1] = creal(emf_kxky[i1+emf->n1fft*i2]/(emf->n1fft*emf->n2fft));
      }
    }
  }
  
}

void airwave_bc_update_E(emf_t *emf)
{
  int i1, i2, i3;

  /*----------------------------------E1------------------------------------*/
  for(i2=0; i2<emf->n2fft; i2++){
    for(i1=0; i1<emf->n1fft; i1++){
      if(i1<emf->n1pad && i2<emf->n2pad)
	emf_kxky[i1+emf->n1fft*i2] = emf->E1[emf->nbe][i2][i1] + I*0.; 
      else
	emf_kxky[i1+emf->n1fft*i2] = 0. + I*0.;
    }
  }
  fftw_execute(fft_airwave);/* Ex(x,y,z=0)-->Hx(kx,ky,z=0) */
  memcpy(emf_kxkyz0, emf_kxky, emf->n1fft*emf->n2fft*sizeof(fftw_complex));
  
  for(i3=0; i3<emf->rd-1; i3++){
    for(i2=0; i2<emf->n2fft; i2++){
      for(i1=0; i1<emf->n1fft; i1++){
	emf_kxky[i1+emf->n1fft*i2] = emf_kxky[i1+emf->n1fft*i2]*emf->sE12kxky[i3][i2][i1];
      }
    }
    fftw_execute(ifft_airwave);
    for(i2=0; i2<emf->n2pad; i2++){
      for(i1=0; i1<emf->n1pad; i1++){
	emf->E1[emf->nbe-1-i3][i2][i1] = creal(emf_kxky[i1+emf->n1fft*i2]/(emf->n1fft*emf->n2fft));
      }
    }
  }

  /*-----------------------------E2---------------------------------------*/
  for(i2=0; i2<emf->n2fft; i2++){
    for(i1=0; i1<emf->n1fft; i1++){
      if(i1<emf->n1pad && i2<emf->n2pad)
	emf_kxky[i1+emf->n1fft*i2] = emf->E2[emf->nbe][i2][i1] + I*0.; 
      else
	emf_kxky[i1+emf->n1fft*i2] = 0. + I*0.;
    }
  }
  fftw_execute(fft_airwave);/* Ex(x,y,z=0)-->Hx(kx,ky,z=0) */
  memcpy(emf_kxkyz0, emf_kxky, emf->n1fft*emf->n2fft*sizeof(fftw_complex));
  
  for(i3=0; i3<emf->rd-1; i3++){
    for(i2=0; i2<emf->n2fft; i2++){
      for(i1=0; i1<emf->n1fft; i1++){
	emf_kxky[i1+emf->n1fft*i2] =	emf_kxky[i1+emf->n1fft*i2]*emf->sE12kxky[i3][i2][i1];
      }
    }
    fftw_execute(ifft_airwave);
    for(i2=0; i2<emf->n2pad; i2++){
      for(i1=0; i1<emf->n1pad; i1++){
	emf->E2[emf->nbe-1-i3][i2][i1] = creal(emf_kxky[i1+emf->n1fft*i2]/(emf->n1fft*emf->n2fft));
      }
    }
  }
}

