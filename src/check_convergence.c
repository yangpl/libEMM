/* check convergence of frequency domain EM field at lowest frequency 
 * NB: checking only in 8 corners of the cube
 *--------------------------------------------------------------------
 *
 *   Copyright (c) 2020-2022, Harbin Institute of Technology, China
 *   Author: Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *--------------------------------------------------------------------*/
#include "cstd.h"
#include "emf.h"

int check_convergence(emf_t *emf)
{
  static float _Complex vold[8], vnew[8];
  static int first = 1;
  int j, k;
  float _Complex ****E1 = emf->fwd_E1;
  
  k = 0;
  if(first){
    vold[0] = E1[0][emf->nbe][emf->nbe][emf->nbe];
    vold[1] = E1[0][emf->nbe][emf->nbe][emf->nbe+emf->n1-1];
    vold[2] = E1[0][emf->nbe][emf->nbe+emf->n2-1][emf->nbe];
    vold[3] = E1[0][emf->nbe][emf->nbe+emf->n2-1][emf->nbe+emf->n1-1];
    vold[4] = E1[0][emf->nbe+emf->n3-1][emf->nbe][emf->nbe];
    vold[5] = E1[0][emf->nbe+emf->n3-1][emf->nbe][emf->nbe+emf->n1-1];
    vold[6] = E1[0][emf->nbe+emf->n3-1][emf->nbe+emf->n2-1][emf->nbe];
    vold[7] = E1[0][emf->nbe+emf->n3-1][emf->nbe+emf->n2-1][emf->nbe+emf->n1-1];
    first = 0;
  }else{
    vnew[0] = E1[0][emf->nbe][emf->nbe][emf->nbe];
    vnew[1] = E1[0][emf->nbe][emf->nbe][emf->nbe+emf->n1-1];
    vnew[2] = E1[0][emf->nbe][emf->nbe+emf->n2-1][emf->nbe];
    vnew[3] = E1[0][emf->nbe][emf->nbe+emf->n2-1][emf->nbe+emf->n1-1];
    vnew[4] = E1[0][emf->nbe+emf->n3-1][emf->nbe][emf->nbe];
    vnew[5] = E1[0][emf->nbe+emf->n3-1][emf->nbe][emf->nbe+emf->n1-1];
    vnew[6] = E1[0][emf->nbe+emf->n3-1][emf->nbe+emf->n2-1][emf->nbe];
    vnew[7] = E1[0][emf->nbe+emf->n3-1][emf->nbe+emf->n2-1][emf->nbe+emf->n1-1];

    for(j=0; j<8; j++){
      if(cabs(vold[j])>0 && cabs(vnew[j]-vold[j])<1e-2*cabs(vold[j])) k++;
      vold[j] = vnew[j];
    }
  }
  return k;
}


