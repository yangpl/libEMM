/* generate nonuniform grid using geometric progression 
 * determine the optimal ratio q by root finding using fixed point iteration 
 *--------------------------------------------------------------------
 *
 *   Copyright (c) 2020-2022, Harbin Institute of Technology, China
 *   Author: Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *--------------------------------------------------------------------*/
#include <math.h>

float create_nugrid(int n, float len, float dx, float *x)
{
  int i;
  float q, qq;
  float eps = 1e-15;

  if(fabs(n*dx-len)<eps) {
    for(i=0; i<=n; i++) x[i] = i*dx;
    return 1;
  }
  
  q = 1.1;
  qq = 1;
  while(1){
    qq = pow(len*(q-1.)/dx + 1., 1./n);
    if(fabs(qq-q)<eps) break;
    q = qq;
  }
  
  for(x[0]=0,i=1; i<=n; i++)
    x[i] = (pow(q,i) - 1.)*dx/(q-1.);

  return q;
}
