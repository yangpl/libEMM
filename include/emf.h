#ifndef emf_h
#define emf_h

typedef struct {
  int mode; //mode=0, modeling; mode=1, inversion, not done yet
  int verb;/* verbose display */
  
  float omega0;
  int n1, n2, n3, nb, ne, nbe, n123;
  int n1pad, n2pad, n3pad, n123pad;
  float d1, d2, d3, dt;

  int nchsrc, nchrec;//number of active src/rec channels for E and H
  char **chsrc, **chrec;

  int nfreq;//number of frequencies
  float *freqs, *omegas;//a list of frequencies
  float f0;//dominant frequency for the wavelet
  float freqmax;
  float Glim;

  float ***rho11, ***rho22, ***rho33;//normal and transverse resistivities
  float rhomin, rhomax; //mimum and maximum resistivity
  float vmin, vmax;//minimum and maximum velocities of the EM wave

  int airwave;//1=modeling with sea surface; 0=no sea surface
  int rd;/* half length/radius of the interpolation operator */
  int n1fft, n2fft;
  float _Complex ***sH1kxky, ***sH2kxky;
  float ***sE12kxky;

  int nt; //number of time steps in total
  float *stf;//source time function
  int ncorner;//number of corners converged

  float ***inveps11, ***inveps22, ***inveps33;//fictitous domain dielectric permittivity
  float *apml, *bpml;
  float ***E1, ***E2, ***E3, ***H1, ***H2, ***H3;
  float ***curlE1, ***curlE2, ***curlE3, ***curlH1, ***curlH2, ***curlH3;
  float ***memD2H3, ***memD3H2, ***memD3H1, ***memD1H3, ***memD1H2, ***memD2H1;
  float ***memD2E3, ***memD3E2, ***memD3E1, ***memD1E3, ***memD1E2, ***memD2E1;
  
  float _Complex ***dcal_fd;

  /* fields in size of the model size * nfreq */
  float _Complex ****fwd_E1,****fwd_E2, ****fwd_E3;
  float _Complex ****fwd_H1,****fwd_H2, ****fwd_H3;
  float _Complex **expfactor;

  int nugrid;//nonuniform grid in z-axis
  float dx3_start, dx3_end;
  float *x3nu;//gridding over x3
  float *x3n, *x3s; //padded x3 for nonstaggered and staggered coordinates
  float **v3, **v3s;//FD coefficients for computing 1st derivative
  float **u3, **u3s;//weights for computing field

} emf_t; /* type of electromagnetic field (emf)  */

#endif
