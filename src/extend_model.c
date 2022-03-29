/* Extend the interior domain with a number of absorbing layers on each side
 *--------------------------------------------------------------------
 *
 *   Copyright (c) 2020, Harbin Institute of Technology, China
 *   Author: Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *--------------------------------------------------------------------*/
#include "cstd.h"
#include "emf.h"


void extend_model_init(emf_t *emf)
{
  int i1, i2, i3, j1, j2, j3;
  float t;

  emf->inveps11 = alloc3float(emf->n1pad, emf->n2pad, emf->n3pad);
  emf->inveps22 = alloc3float(emf->n1pad, emf->n2pad, emf->n3pad);
  emf->inveps33 = alloc3float(emf->n1pad, emf->n2pad, emf->n3pad);

  /* copy the inner part */
  for(i3=0; i3<emf->n3; i3++){
    for(i2=0; i2<emf->n2; i2++){
      for(i1=0; i1<emf->n1; i1++){
  	t = 2.*emf->omega0;
  	emf->inveps11[i3+emf->nbe][i2+emf->nbe][i1+emf->nbe] = t*emf->rho11[i3][i2][i1];
  	emf->inveps22[i3+emf->nbe][i2+emf->nbe][i1+emf->nbe] = t*emf->rho22[i3][i2][i1];
  	emf->inveps33[i3+emf->nbe][i2+emf->nbe][i1+emf->nbe] = t*emf->rho33[i3][i2][i1];
      }
    }
  }
  /* pad the left and the right face */
  for(i3=0; i3<emf->n3pad; i3++) {
    for(i2=0; i2<emf->n2pad; i2++) {
      for(i1=0; i1<emf->nbe; i1++) {
  	j1 = emf->n1pad-1-i1;
  	emf->inveps11[i3][i2][i1] = emf->inveps11[i3][i2][emf->nbe        ];
  	emf->inveps22[i3][i2][i1] = emf->inveps22[i3][i2][emf->nbe        ];
  	emf->inveps33[i3][i2][i1] = emf->inveps33[i3][i2][emf->nbe        ];
  	emf->inveps11[i3][i2][j1] = emf->inveps11[i3][i2][emf->n1pad-emf->nbe-1];
  	emf->inveps22[i3][i2][j1] = emf->inveps22[i3][i2][emf->n1pad-emf->nbe-1];
  	emf->inveps33[i3][i2][j1] = emf->inveps33[i3][i2][emf->n1pad-emf->nbe-1];
      }
    }
  }
  /* pad the front and the rear face */
  for(i3=0; i3<emf->n3pad; i3++) {
    for(i2=0; i2<emf->nbe; i2++) {
      j2 = emf->n2pad-i2-1;
      for(i1=0; i1<emf->n1pad; i1++) {
  	emf->inveps11[i3][i2][i1] = emf->inveps11[i3][emf->nbe        ][i1];
  	emf->inveps22[i3][i2][i1] = emf->inveps22[i3][emf->nbe        ][i1];
  	emf->inveps33[i3][i2][i1] = emf->inveps33[i3][emf->nbe        ][i1];
  	emf->inveps11[i3][j2][i1] = emf->inveps11[i3][emf->n2pad-emf->nbe-1][i1];
  	emf->inveps22[i3][j2][i1] = emf->inveps22[i3][emf->n2pad-emf->nbe-1][i1];
  	emf->inveps33[i3][j2][i1] = emf->inveps33[i3][emf->n2pad-emf->nbe-1][i1];
      }
    }
  }
  /* pad the top and the bottom face */
  for(i3=0; i3<emf->nbe; i3++) {
    j3 = emf->n3pad-i3-1;
    for(i2=0; i2<emf->n2pad; i2++) {
      for(i1=0; i1<emf->n1pad; i1++) {
  	emf->inveps11[i3][i2][i1] = emf->inveps11[emf->nbe        ][i2][i1];
  	emf->inveps22[i3][i2][i1] = emf->inveps22[emf->nbe        ][i2][i1];
  	emf->inveps33[i3][i2][i1] = emf->inveps33[emf->nbe        ][i2][i1];
  	emf->inveps11[j3][i2][i1] = emf->inveps11[emf->n3pad-emf->nbe-1][i2][i1];
  	emf->inveps22[j3][i2][i1] = emf->inveps22[emf->n3pad-emf->nbe-1][i2][i1];
  	emf->inveps33[j3][i2][i1] = emf->inveps33[emf->n3pad-emf->nbe-1][i2][i1];
      }
    }
  }

  if(emf->airwave==1){
    /* air water interface: sigma = 0.5*(sigma_air+sigma_water)=0.5*sigma_water */
    i3=emf->nbe;
    for(i2=0; i2<emf->n2pad; i2++){
      for(i1=0; i1<emf->n1pad; i1++){
  	emf->inveps11[i3][i2][i1] *= 2;
  	emf->inveps22[i3][i2][i1] *= 2;
      }
    }
  }

}



void extend_model_close(emf_t *emf)
{
  free3float(emf->inveps11);
  free3float(emf->inveps22);
  free3float(emf->inveps33);
}
