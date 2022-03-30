/* solve the vandermonde system V^T(x0,x1,...,xn) a = f
 * where the n+1 points are prescribed by vector x=[x0,x1,...,xn];
 * the solution is stored in vector a.
 * 
 * Reference:
 * [1] Golub and Loan, 1978, Matrix computation, 3rd ed., Algorithm 4.6.1
 *--------------------------------------------------------------------
 *
 *   Copyright (c) 2020-2022, Harbin Institute of Technology, China
 *   Author: Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *--------------------------------------------------------------------*/
void vandermonde(int n, float *x, float *a, float *f)
{
  int i,k;

  /* calculate Newton representation of the interpolating polynomial */
  for(i=0; i<=n; ++i) a[i] = f[i];

  for(k=0; k<n; k++){
    for(i=n; i>k; i--){
      a[i] = (a[i]-a[i-1])/(x[i]-x[i-k-1]);
    }
  }

  for(k=n-1; k>=0; k--){
    for(i=k; i<n; i++){
      a[i] -= a[i+1]*x[k];
    }
  }
}
