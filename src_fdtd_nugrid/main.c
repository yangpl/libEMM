/* 3D CSEM modeling using FDTD method in fictious wave domain
 *
 * Copyright (c) 2021 Harbin Institute of Technology. All rights reserved.
 * Anothr: Pengliang Yang 
 * Email: ypl.2100@gmail.com
 */
#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cstd.h"
#include <fftw3.h>

int iproc, nproc, ierr;

#define PI 3.14159265358979323846264
#define invmu0  (1./(4.*PI*1e-7))

typedef struct {
  int nsrc;/* number of sources on each processor, default=1 */
  int nrec;/* number of receivers on each processor */
  int nsubsrc;/* number of distributed subpoints for each source */
  int nsubrec;/* number of distributed subpoints for each receiver */
  float lensrc;/* length of source antenna */
  float lenrec; /* length of receiver antenna */
  int *shot_idx;
  int nsrc_total;
  int nrec_total;
  float *src_x1, *src_x2, *src_x3, *src_azimuth, *src_dip;
  float *rec_x1, *rec_x2, *rec_x3, *rec_azimuth, *rec_dip;
  float x1min, x1max, x2min, x2max, x3min, x3max;/* coordinate bounds */
} acqui_t;/* type of acquisition geometry */

typedef struct {
  int airwave;
  int n1, n2, n3, nb, nbe, n1pad, n2pad, n3pad;
  int n123, n123pad;
  float *x1nu, *x2nu, *x3nu;//non-staggered coordinates on non-uniform grid
  float dx1min, dx2min, dx3min;

  float *x1n, *x2n, *x3n; // extended coordinates for non-uniform grid
  float *x1s, *x2s, *x3s; // shift half grid for extended coordinates
  int rd1, rd2, rd3, rd;//half FD length
  int order;//FD order, order=2*radius

  int nch_src, nchrec;//number of active src/rec channels for E and H
  char **ch_src, **ch_rec;

  int nt;
  int nfreq;
  float dt;
  float *freqs, *omegas;//a list of frequencies
  float f0, omega0;//dominant frequency for the wavelet
  float freqmax;
  float sigma_water, sigma_formation;
  float rhomax, rhomin;
  float vmin, vmax;
  float Glim;

  float *stf;
  float waterdepth;
  int reciprocity;
  
  int nh;
  float *hx, *hz;
  
  float *kx, *ky;
  float *hw1, *hw2;//Hanning window
  float **sqrtkx2ky2;

  int n1fft, n2fft;
  float d1uni, d2uni, d3uni; //grid spacing of a uniform grid for air-water
  int n1uni, n2uni;//number of intervals for uniform grid 
  int *nu_i1, *nu_i2, *nu_i1_s, *nu_i2_s;
  float ***uni_H1, ***uni_H2, ***uni_E1, ***uni_E2;//interpolated fields on uniform grid
  
  float dx1_start, dx1_end, dx2_start, dx2_end, dx3_start, dx3_end;
  float *a1, *b1, *a2, *b2, *a3, *b3;
  
  float ***E1, ***E2, ***E3;
  float ***H1, ***H2, ***H3;
  float ***curlE1, ***curlE2, ***curlE3;
  float ***curlH1, ***curlH2, ***curlH3;
  float ***memD1E2, ***memD2E1, ***memD1E3, ***memD3E1, ***memD2E3, ***memD3E2;
  float ***memD1H2, ***memD2H1, ***memD1H3, ***memD3H1, ***memD2H3, ***memD3H2;
  float _Complex ****fwd_E1, ****fwd_E2, ****fwd_E3, ****fwd_H1, ****fwd_H2, ****fwd_H3, ****fwd_Jz;
  float _Complex ****adj_E1, ****adj_E2, ****adj_E3, ****adj_H1, ****adj_H2, ****adj_H3, ****adj_Jz;
  
  float ***rho11, ***rho22, ***rho33;
  float ***inveps11, ***inveps22, ***inveps33;
  float _Complex ***dcal_fd;
  
  float **v1, **v2, **v3;//FD coefficients
  float **v1s, **v2s, **v3s;//FD coefficients
} emf_t;


typedef struct {
  /*---------------------sources----------------------------*/
  int **src_i1, **src_i2, **src_i3;/* index on FD grid */
  float ***src_w1, ***src_w2, ***src_w3; /* weights on FD grid for f(x) */
  float ***src_v1, ***src_v2, ***src_v3; /* weights on FD grid for f'(x)*/

  /*---------------------receivers--------------------------*/
  int **rec_i1, **rec_i2, **rec_i3;/* index on FD grid */
  float ***rec_w1, ***rec_w2, ***rec_w3; /* weights on FD grid for f(x) */
  float ***rec_v1, ***rec_v2, ***rec_v3; /* weights on FD grid for f'(x) */
} interp_t;/* type of interpolation on regular and staggerred grid */

int cmpfunc(const void *a, const void *b) { return ( *(int*)a - *(int*)b ); }

void emf_init(emf_t *emf)
{
  char *frho11, *frho22, *frho33, *fx1nu, *fx2nu, *fx3nu;//, *fhx, *fhz;
  FILE *fp=NULL;
  int ifreq, ic, istat, i1, i2, i3;
  float tmp;

  if(!getparint("n1", &emf->n1)) emf->n1=101; 
  /* number of cells in axis-1, nx */
  if(!getparint("n2", &emf->n2)) emf->n2=101; 
  /* number of cells in axis-2, ny */
  if(!getparint("n3", &emf->n3)) emf->n3=51; 
  /* number of cells in axis-3, nz */
  if(!getparint("nb", &emf->nb)) emf->nb=14; 
  /* number of PML layers on each side */
  if(!getparint("rd1", &emf->rd1)) emf->rd1=2; 
  /* half length of FD stencil */
  if(!getparint("rd2", &emf->rd2)) emf->rd2=2; 
  /* half length of FD stencil */
  if(!getparint("rd3", &emf->rd3)) emf->rd3=2; 
  /* half length of FD stencil */
  emf->rd = MAX(MAX(emf->rd1, emf->rd2), emf->rd3);
  if(!getparint("airwave", &emf->airwave)) emf->airwave=1; 
  /* simulate airwave on top boundary */
  if(!getparfloat("f0", &emf->f0)) emf->f0=0.5;
  emf->omega0 = 2.*PI*emf->f0;
  /* reference frequency */
  if(!getparfloat("Glim", &emf->Glim)) emf->Glim=5;/* 5 points/wavelength */
  if(!(getparstring("frho11", &frho11))) err("Need frho11= ");
  if(!(getparstring("frho22", &frho22))) err("Need frho22= ");
  if(!(getparstring("frho33", &frho33))) err("Need frho33= ");

  if(!(emf->nfreq=countparval("freqs"))) err("Need freqs= vector");
  /* number of frequencies for electromagnetic emf->modeling */
  emf->freqs=alloc1float(emf->nfreq);
  emf->omegas=alloc1float(emf->nfreq);
  getparfloat("freqs", emf->freqs);/* a list of frequencies separated by comma */
  qsort(emf->freqs, emf->nfreq, sizeof(float), cmpfunc);/*sort frequencies in ascending order*/
  for(ifreq=0; ifreq<emf->nfreq; ++ifreq) {
    emf->omegas[ifreq]=2.*PI*emf->freqs[ifreq];
    if(iproc==0) printf("freq[%d]=%g ", ifreq+1, emf->freqs[ifreq]);
  }
  if(iproc==0) printf("\n");
  
  /* read active source channels */
  if((emf->nch_src=countparval("ch_src"))!=0) {
    emf->ch_src=(char**)alloc1(emf->nch_src, sizeof(void*));
    getparstringarray("ch_src", emf->ch_src);
    /* active source channels: Ex, Ey, Ez, Hx, Hy, Hz or their combinations */
  }else{
    emf->nch_src=1;
    emf->ch_src=(char**)alloc1(emf->nch_src, sizeof(void*));
    emf->ch_src[0]="Ex";
  }
  /* read active receiver channels */
  if((emf->nchrec=countparval("ch_rec"))!=0) {
    emf->ch_rec=(char**)alloc1(emf->nchrec, sizeof(void*));
    getparstringarray("ch_rec", emf->ch_rec);
    /* active receiver channels: Ex, Ey, Ez, Hx, Hy, Hz or their combinations */
  }else{
    emf->nchrec=4;
    emf->ch_rec=(char**)alloc1(emf->nchrec, sizeof(void*));
    emf->ch_rec[0]="Ex";
    emf->ch_rec[1]="Ey";
    emf->ch_rec[2]="Hx";
    emf->ch_rec[3]="Hy";
  }
  if(iproc==0){
    printf("Active source channels:");
    for(ic=0; ic<emf->nch_src; ++ic) printf(" %s", emf->ch_src[ic]);
    printf("\n");
    printf("Active recever channels:");
    for(ic=0; ic<emf->nchrec; ++ic) printf(" %s", emf->ch_rec[ic]);
    printf("\n");
  }

  if(!getparfloat("waterdepth", &emf->waterdepth)) emf->waterdepth=325;


  emf->n123 = emf->n1*emf->n2*emf->n3;
  emf->nbe = emf->nb + emf->rd;/* number of PML layers + extra 2 points due to 4-th order FD */
  emf->n1pad = emf->n1+2*emf->nbe;/* total number of grid points after padding PML+extra */
  emf->n2pad = emf->n2+2*emf->nbe;/* total number of grid points after padding PML+extra */
  emf->n3pad = emf->n3+2*emf->nbe;/* total number of grid points after padding PML+extra */
  emf->n123pad = emf->n1pad*emf->n2pad*emf->n3pad;
  if(iproc==0){
    printf("PML layers on each side: nb=%d\n", emf->nb);
    printf("Number of layers extended outside model: nbe=%d\n", emf->nbe);
    printf("[n1, n2, n3]=[%d, %d, %d]\n", emf->n1, emf->n2, emf->n3);
    printf("[n1pad, n2pad, n3pad]=[%d, %d, %d]\n", emf->n1pad, emf->n2pad, emf->n3pad);
    printf("[rd1, rd2, rd3]=[%d, %d, %d]\n", emf->rd1, emf->rd2, emf->rd3);
  }
  
  if(!(getparstring("fx1nu", &fx1nu))) err("Need fx1nu=");/* filename of x coordinates */
  if(!(getparstring("fx2nu", &fx2nu))) err("Need fx2nu=");/* filename of y coordinates */
  if(!(getparstring("fx3nu", &fx3nu))) err("Need fx3nu=");/* filename of z coordinates */

  emf->x1nu = alloc1float(emf->n1);
  emf->x2nu = alloc1float(emf->n2);
  emf->x3nu = alloc1float(emf->n3);

  fp=fopen(fx1nu, "rb");
  if(fp==NULL) err("cannot open fx1nu=%s", fx1nu);
  istat = fread(emf->x1nu, sizeof(float), emf->n1, fp);
  if(istat != emf->n1) err("size parameter does not match the file!");
  fclose(fp);

  fp=fopen(fx2nu, "rb");
  if(fp==NULL) err("cannot open fx2nu=%s", fx2nu);
  istat = fread(emf->x2nu, sizeof(float), emf->n2, fp);
  if(istat != emf->n2) err("size parameter does not match the file!");
  fclose(fp);

  fp=fopen(fx3nu, "rb");
  if(fp==NULL) err("cannot open fx3nu=%s", fx3nu);
  istat = fread(emf->x3nu, sizeof(float), emf->n3, fp);
  if(istat != emf->n3) err("size parameter does not match the file!");
  fclose(fp);

  
  emf->dx1min = emf->x1nu[1]- emf->x1nu[0];
  for(i1=0; i1<emf->n1-1; i1++) {
    tmp =  emf->x1nu[i1+1]- emf->x1nu[i1];
    if(tmp<emf->dx1min) emf->dx1min = tmp;
  }
  emf->dx2min = emf->x2nu[1]- emf->x2nu[0];
  for(i2=0; i2<emf->n2-1; i2++) {
    tmp =  emf->x2nu[i2+1]- emf->x2nu[i2];
    if(tmp<emf->dx2min) emf->dx2min = tmp;
  }
  emf->dx3min = emf->x3nu[1]- emf->x3nu[0];
  for(i3=0; i3<emf->n3-1; i3++) {
    tmp =  emf->x3nu[i3+1]- emf->x3nu[i3];
    if(tmp<emf->dx3min) emf->dx3min = tmp;
  }
  if(iproc==0){
    printf("[dx1min, dx2min, dx3min]=[%g, %g, %g]\n", emf->dx1min, emf->dx2min, emf->dx3min);
  }

  /*-------------------------------------------------------*/
  emf->rho11 = alloc3float(emf->n1, emf->n2, emf->n3);
  emf->rho22 = alloc3float(emf->n1, emf->n2, emf->n3);
  emf->rho33 = alloc3float(emf->n1, emf->n2, emf->n3);
  
  /* read  resistivity, assuming the homogenization has been done within the inputs */
  fp = fopen(frho11, "r");
  if(fp==NULL) err("cannot open file");
  istat = fread(emf->rho11[0][0], sizeof(float), emf->n123, fp);
  if(istat != emf->n123) err("size parameter does not match the file!");
  fclose(fp);

  fp = fopen(frho22, "r");
  if(fp==NULL) err("cannot open file");
  istat = fread(emf->rho22[0][0], sizeof(float), emf->n123, fp);
  if(istat != emf->n123) err("size parameter does not match the file!");
  fclose(fp);

  fp = fopen(frho33, "r");
  if(fp==NULL) err("cannot open file");
  istat = fread(emf->rho33[0][0], sizeof(float), emf->n123, fp);
  if(istat != emf->n123) err("size parameter does not match the file!");
  fclose(fp);
  
}

void emf_close(emf_t *emf)
{
  free1float(emf->freqs);
  free1float(emf->omegas);

  free1float(emf->x1nu);
  free1float(emf->x2nu);
  free1float(emf->x3nu);
  
  free3float(emf->rho11); 
  free3float(emf->rho22);
  free3float(emf->rho33);


}


void extend_model_init(emf_t *emf)
{
  int i1, i2, i3, j1, j2, j3;

  emf->inveps11 = alloc3float(emf->n1pad, emf->n2pad, emf->n3pad);
  emf->inveps22 = alloc3float(emf->n1pad, emf->n2pad, emf->n3pad);
  emf->inveps33 = alloc3float(emf->n1pad, emf->n2pad, emf->n3pad);

  /* copy the inner part */
  for(i3=0; i3<emf->n3; i3++){
    for(i2=0; i2<emf->n2; i2++){
      for(i1=0; i1<emf->n1; i1++){
  	float t = 2.*emf->omega0;
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



/* solve the vandermonde system V^T(x0, x1, ..., xn) a = f
 * where the n+1 points are prescribed by vector x=[x0, x1, ..., xn];
 * the solution is stored in vector a.
 * 
 * Reference:
 * [1] Golub and Loan, 1978, Matrix computation, 3rd ed., Algorithm 4.6.1
 *
 *  Copy_end (c) Pengliang Yang, 2020, Harbin Institute of Technology, China
 *  Copy_end (c) Pengliang Yang, 2018, University Grenoble Alpes, France
 *  Homepage: https://yangpl.wordpress.com
 *  E-mail: ypl.2100@gmail.com
 */
void vandermonde(int n, float *x, float *a, float *f)
{
  int i, k;

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

void acqui_init(acqui_t *acqui, emf_t * emf)
/*< read acquisition file to initialize survey geometry >*/
{
  static int nd = 5000;//maximum dimensions for the number of source and receiver
  float src_x1[nd], src_x2[nd], src_x3[nd], src_hd[nd], src_pit[nd];/* source receiver coordinates */
  float rec_x1[nd], rec_x2[nd], rec_x3[nd], rec_hd[nd], rec_pit[nd];/* source receiver coordinates */
  int rec_idx[nd];/* reciver index associated with current processor */
  float x, y, z, hd, pit;
  float x1min, x1max, x2min, x2max, x3min, x3max;
  int isrc, irec, iseof, idx, i, nsrc;
  char *fsrc, *frec, *fsrcrec;
  FILE *fp=NULL;

  if(!getparint("reciprocity", &emf->reciprocity)) emf->reciprocity=0;

  if(!(getparstring("fsrc", &fsrc))) err("Need fsrc= ");
  /* file to specify all possible source locations */
  if(!(getparstring("frec", &frec))) err("Need frec= ");
  /* file to specify all possible receiver locations */
  if(!(getparstring("fsrcrec", &fsrcrec))) err("Need fsrcrec= ");
  /* file to specify how source and receiver are combined */

  acqui->x1min = emf->x1nu[0];
  acqui->x1max = emf->x1nu[emf->n1-1];
  acqui->x2min = emf->x2nu[0];
  acqui->x2max = emf->x2nu[emf->n2-1];
  acqui->x3min = emf->x3nu[0];
  acqui->x3max = emf->x3nu[emf->n3-1];  
  if(!getparint("nsubsrc", &acqui->nsubsrc)) acqui->nsubsrc=1;
  /* number of subpoints to represent one source location */
  if(!getparint("nsubrec", &acqui->nsubrec)) acqui->nsubrec=1;
  /* number of subpoints to represent one receiver location */
  if(!getparfloat("lensrc", &acqui->lensrc)) acqui->lensrc=300.;
  /* length of the source antenna, default=1 m */
  if(!getparfloat("lenrec", &acqui->lenrec)) acqui->lenrec=8.;
  /* length of the receiver antenna, default=8 m */

  acqui->shot_idx = alloc1int(nproc);
  nsrc = countparval("shots");
  if(nsrc>0){
    if( nsrc<nproc) err("nproc > number of shot indices! ");
    getparint("shots", acqui->shot_idx);/* a list of source index separated by comma */
  }
  if(nsrc==0){
    for(i=0; i<nproc; i++) acqui->shot_idx[i] = i+1;//index starts from 1
  }
  idx = acqui->shot_idx[iproc];
  //find the receiver indices associated with source-idx, take reciprocity into account
  fp = fopen(fsrcrec,"r");
  if(fp==NULL) err("file does not exist!");
  fscanf(fp, "%*[^\n]\n");//skip a line at the beginning of the file
  i = 0;
  while(1){
    /* (northing,easting,depth)=(y,x,z);   azimuth = heading;  dip=pitch */
    iseof=fscanf(fp,"%d %d", &isrc, &irec);
    if(iseof==EOF)
      break;
    else{
      if(emf->reciprocity){
  	if(irec==idx) {//irec-th common receiver gather
  	  rec_idx[i] = isrc;//the global source index associated with current receiver
  	  i++;
  	}
      }else{
  	if(isrc==idx) {//isrc-th source gather
  	  rec_idx[i] = irec;//the global receiver index associated with current source
  	  i++;
  	}
      }
    }
  }
  acqui->nrec = i;
  fclose(fp);


  /*============================================*/
  /* step 2: read all possible source locations */
  /*============================================*/
  x1min = acqui->x1min;
  x1max = acqui->x1max;
  x2min = acqui->x2min;
  x2max = acqui->x2max;
  x3min = acqui->x3min;
  x3max = acqui->x3max;
  fp = fopen(fsrc,"r");
  if(fp==NULL) err("file does not exist!"); 
  fscanf(fp, "%*[^\n]\n");//skip a line at the beginning of the file
  isrc = 0;
  while(1){
    /* (northing,easting,depth)=(y,x,z);   azimuth = heading;  dip=pitch */
    iseof=fscanf(fp,"%f %f %f %f %f %d", &x, &y, &z, &hd, &pit, &idx);
    if(iseof==EOF)
      break;
    else{
      src_x1[isrc] = x;
      src_x2[isrc] = y;
      src_x3[isrc] = z;
      src_hd[isrc] = hd;
      src_pit[isrc] = pit;

      x1min = MIN(x1min, src_x1[isrc]);
      x1max = MAX(x1max, src_x1[isrc]);
      x2min = MIN(x2min, src_x2[isrc]);
      x2max = MAX(x2max, src_x2[isrc]);
      x3min = MIN(x3min, src_x3[isrc]);
      x3max = MAX(x3max, src_x3[isrc]);

      isrc++;
    }
  }
  acqui->nsrc_total = isrc;
  fclose(fp);
  if(x1min<acqui->x1min) err("source location: x<x1min");
  if(x2min<acqui->x2min) err("source location: y<x2min");
  if(x3min<acqui->x3min) err("source location: z<x3min");
  if(x1max>acqui->x1max) err("source location: x>x1max");
  if(x2max>acqui->x2max) err("source location: y>x2max");
  if(x3max>acqui->x3max) err("source location: z>x3max");
    
  /*==============================================*/
  /* step 1: read all possible receiver locations */
  /*==============================================*/
  fp = fopen(frec,"r");
  if(fp==NULL) err("file does not exist!"); 
  fscanf(fp, "%*[^\n]\n");//skip a line at the beginning of the file
  irec = 0;
  while(1){
    /* (northing,easting,depth)=(y,x,z);   azimuth = heading;  dip=pitch */
    iseof=fscanf(fp,"%f %f %f %f %f %d", &x, &y, &z, &hd, &pit, &idx);
    if(iseof==EOF)
      break;
    else{
      rec_x1[irec] = x;
      rec_x2[irec] = y;
      rec_x3[irec] = z;
      rec_hd[irec] = hd;
      rec_pit[irec] = pit;
      
      x1min = MIN(x1min, rec_x1[irec]);
      x1max = MAX(x1max, rec_x1[irec]);
      x2min = MIN(x2min, rec_x2[irec]);
      x2max = MAX(x2max, rec_x2[irec]);
      x3min = MIN(x3min, rec_x3[irec]);
      x3max = MAX(x3max, rec_x3[irec]);
      
      irec++;
    }
  }
  acqui->nrec_total = irec;
  fclose(fp);
  if(x1min<acqui->x1min) err("receiver location: x<x1min");
  if(x2min<acqui->x2min) err("receiver location: y<x2min");
  if(x3min<acqui->x3min) err("receiver location: z<x3min");
  if(x1max>acqui->x1max) err("receiver location: x>x1max");
  if(x2max>acqui->x2max) err("receiver location: y>x2max");
  if(x3max>acqui->x3max) err("receiver location: z>x3max");

  //-----------------------------------------------------------
  acqui->nsrc = 1; /* assume 1 source per process by default */
  acqui->src_x1 = alloc1float(acqui->nsrc);
  acqui->src_x2 = alloc1float(acqui->nsrc);
  acqui->src_x3 = alloc1float(acqui->nsrc);
  acqui->src_azimuth = alloc1float(acqui->nsrc);
  acqui->src_dip = alloc1float(acqui->nsrc);
  for(isrc=0; isrc< acqui->nsrc; ++isrc){
    i = acqui->shot_idx[iproc]-1;//index starts from 1
    if(emf->reciprocity){
      acqui->src_x1[isrc] = rec_x1[i];
      acqui->src_x2[isrc] = rec_x2[i];
      acqui->src_x3[isrc] = rec_x3[i];
      acqui->src_azimuth[isrc] = rec_hd[i];
      acqui->src_dip[isrc] = rec_pit[i];
    }else{
      acqui->src_x1[isrc] = src_x1[i];
      acqui->src_x2[isrc] = src_x2[i];
      acqui->src_x3[isrc] = src_x3[i];
      acqui->src_azimuth[isrc] = src_hd[i];
      acqui->src_dip[isrc] = src_pit[i];
    }
  }/* end for isrc */

  //-----------------------------------------------------------
  acqui->rec_x1 = alloc1float(acqui->nrec);
  acqui->rec_x2 = alloc1float(acqui->nrec);
  acqui->rec_x3 = alloc1float(acqui->nrec);
  acqui->rec_azimuth = alloc1float(acqui->nrec);
  acqui->rec_dip = alloc1float(acqui->nrec);
  for(irec=0; irec<acqui->nrec; ++irec){//we always have: acqui->nrec <= acqui->nrec_total
    //nrec < nrec_total if only inline data are used
    //idx=index of the receivers associated with current source or common receiver gather
    i = rec_idx[irec]-1; //index starts from 1
    if(emf->reciprocity){
      acqui->rec_x1[irec] = src_x1[i];
      acqui->rec_x2[irec] = src_x2[i];
      acqui->rec_x3[irec] = src_x3[i];
      acqui->rec_azimuth[irec] = src_hd[i];
      acqui->rec_dip[irec] = src_pit[i];
    }else{
      acqui->rec_x1[irec] = rec_x1[i];
      acqui->rec_x2[irec] = rec_x2[i];
      acqui->rec_x3[irec] = rec_x3[i];
      acqui->rec_azimuth[irec] = rec_hd[i];
      acqui->rec_dip[irec] = rec_pit[i];
    }
    
  }/* end for irec */
  if(iproc==0){
    printf("[x1min,x1max]=[%g, %g]\n", acqui->x1min, acqui->x1max);
    printf("[x2min,x2max]=[%g, %g]\n", acqui->x2min, acqui->x2max);
    printf("[x3min,x3max]=[%g, %g]\n", acqui->x3min, acqui->x3max);
    printf("nsrc_total=%d\n", acqui->nsrc_total);
    printf("nrec_total=%d\n", acqui->nrec_total);
  }
  printf("reciprocity=%d isrc=%d, nrec=%d (x,y,z)=(%.2f, %.2f, %.2f)\n",
	 emf->reciprocity, acqui->shot_idx[iproc], acqui->nrec, acqui->src_x1[0], acqui->src_x2[0], acqui->src_x3[0]);
}


void acqui_close(acqui_t *acqui)
/*< free the allocated variables for acquisition >*/
{
  free(acqui->src_x1);
  free(acqui->src_x2);
  free(acqui->src_x3);
  free(acqui->src_azimuth);
  free(acqui->src_dip);

  free(acqui->rec_x1);
  free(acqui->rec_x2);
  free(acqui->rec_x3);
  free(acqui->rec_azimuth);
  free(acqui->rec_dip);
}

void nugrid_init(emf_t *emf)
{
  int i1, i2, i3, i, m, loop;
  float *xx, *aa, *ff;
  
  emf->x1n = alloc1float(emf->n1pad);
  emf->x2n = alloc1float(emf->n2pad);
  emf->x3n = alloc1float(emf->n3pad);
  emf->x1s = alloc1float(emf->n1pad);
  emf->x2s = alloc1float(emf->n2pad);
  emf->x3s = alloc1float(emf->n3pad);

  emf->dx1_start  = emf->x1nu[1]    - emf->x1nu[0];
  emf->dx1_end    = emf->x1nu[emf->n1-1] - emf->x1nu[emf->n1-2];
  emf->dx2_start  = emf->x2nu[1]    - emf->x2nu[0];
  emf->dx2_end    = emf->x2nu[emf->n2-1] - emf->x2nu[emf->n2-2];
  emf->dx3_start  = emf->x3nu[1]    - emf->x3nu[0];
  emf->dx3_end    = emf->x3nu[emf->n3-1] - emf->x3nu[emf->n3-2];

  for(i1=0; i1<emf->n1; i1++) emf->x1n[i1+emf->nbe] = emf->x1nu[i1];
  for(i2=0; i2<emf->n2; i2++) emf->x2n[i2+emf->nbe] = emf->x2nu[i2];
  for(i3=0; i3<emf->n3; i3++) emf->x3n[i3+emf->nbe] = emf->x3nu[i3];
  
  for(i1=0; i1<emf->nbe; i1++) {
    emf->x1n[i1]      = emf->x1n[emf->nbe]      - (emf->nbe-i1)*emf->dx1_start;
    emf->x1n[emf->nbe+emf->n1+i1] = emf->x1n[emf->nbe+emf->n1-1] + (i1+1)*emf->dx1_end;
  }
  for(i2=0; i2<emf->nbe; i2++){
    emf->x2n[i2]      = emf->x2n[emf->nbe]      - (emf->nbe-i2)*emf->dx2_start;
    emf->x2n[emf->nbe+emf->n2+i2] = emf->x2n[emf->nbe+emf->n2-1] + (i2+1)*emf->dx2_end;
  }
  for(i3=0; i3<emf->nbe; i3++){
    emf->x3n[i3]      = emf->x3n[emf->nbe]      - (emf->nbe-i3)*emf->dx3_start;
    emf->x3n[emf->nbe+emf->n3+i3] = emf->x3n[emf->nbe+emf->n3-1] + (i3+1)*emf->dx3_end;
  }
  
  for(i1=0; i1<emf->n1pad-1; i1++) emf->x1s[i1] = 0.5*(emf->x1n[i1] + emf->x1n[i1+1]);
  emf->x1s[emf->n1pad-1] = emf->x1s[emf->n1pad-2] + emf->dx1_end;
  for(i2=0; i2<emf->n2pad-1; i2++) emf->x2s[i2] = 0.5*(emf->x2n[i2] + emf->x2n[i2+1]);
  emf->x2s[emf->n2pad-1] = emf->x2s[emf->n2pad-2] + emf->dx2_end;
  for(i3=0; i3<emf->n3pad-1; i3++) emf->x3s[i3] = 0.5*(emf->x3n[i3] + emf->x3n[i3+1]);
  emf->x3s[emf->n3pad-1] = emf->x3s[emf->n3pad-2] + emf->dx3_end;
  
  emf->v1 = alloc2float(2*emf->rd1, emf->n1pad);
  emf->v2 = alloc2float(2*emf->rd2, emf->n2pad);
  emf->v3 = alloc2float(2*emf->rd3, emf->n3pad);
  emf->v1s = alloc2float(2*emf->rd1, emf->n1pad);
  emf->v2s = alloc2float(2*emf->rd2, emf->n2pad);
  emf->v3s = alloc2float(2*emf->rd3, emf->n3pad);
  memset(&emf->v1[0][0], 0, 2*emf->rd1*emf->n1pad*sizeof(float));
  memset(&emf->v2[0][0], 0, 2*emf->rd2*emf->n2pad*sizeof(float));
  memset(&emf->v3[0][0], 0, 2*emf->rd3*emf->n3pad*sizeof(float));
  memset(&emf->v1s[0][0], 0, 2*emf->rd1*emf->n1pad*sizeof(float));
  memset(&emf->v2s[0][0], 0, 2*emf->rd2*emf->n2pad*sizeof(float));
  memset(&emf->v3s[0][0], 0, 2*emf->rd3*emf->n3pad*sizeof(float));

  xx = alloc1float(2*emf->rd);
  aa = alloc1float(2*emf->rd);
  ff = alloc1float(2*emf->rd);

  for(loop=0; loop<2; loop++){
    /*------------------------------------------------------------------*/
    for(i1=emf->rd1; i1<emf->n1pad-emf->rd1; i1++){
      for(i=0; i<2*emf->rd1; i++){/* construct vector x1[], x2[], x3[] */
	m = i-emf->rd1;/* m=offset/shift between [-emf->rd1+1, emf->rd1] */
	if(loop==0) xx[i] = emf->x1s[i1+m]-emf->x1n[i1];//non-shifted point as the center
	else        xx[i] = emf->x1n[i1+1+m]-emf->x1s[i1];//shifted point as the center
      }
      for(i=0; i<2*emf->rd1; i++){
	//prepare a different rhs for Vandemonde matrix
	memset(ff, 0, 2*emf->rd1*sizeof(float));
	ff[i] =  1.;
	vandermonde(2*emf->rd1-1, xx, aa, ff);//invert vandemonde matrix system
	if(loop==0) emf->v1[i1][i] = aa[1]; //take coefficients to extract derivatives f'(x)
	else        emf->v1s[i1][i] = aa[1]; //take coefficients to extract derivatives f'(x)
      }
    }
    /*------------------------------------------------------------------*/
    for(i2=emf->rd2; i2<emf->n2pad-emf->rd2; i2++){
      for(i=0; i<2*emf->rd2; i++){/* construct vector x1[], x2[], x3[] */
	m = i-emf->rd2;/* m=offset/shift between [-emf->rd2+1, emf->rd2] */
	if(loop==0) xx[i] = emf->x2s[i2+m]-emf->x2n[i2];//non-shifted point as the center
	else        xx[i] = emf->x2n[i2+1+m]-emf->x2s[i2];//non-shifted point as the center
      }
      for(i=0; i<2*emf->rd2; i++){
	//prepare a different rhs for Vandemonde matrix
	memset(ff, 0, 2*emf->rd2*sizeof(float));
	ff[i] =  1.;
	vandermonde(2*emf->rd2-1, xx, aa, ff);//invert vandemonde matrix system
	if(loop==0) emf->v2[i2][i] = aa[1]; //take coefficients to extract derivatives f'(x)
	else        emf->v2s[i2][i] = aa[1];
      }
    }
    /*------------------------------------------------------------------*/
    for(i3=emf->rd3; i3<emf->n3pad-emf->rd3; i3++){
      for(i=0; i<2*emf->rd3; i++){/* construct vector x1[], x2[], x3[] */
	m = i-emf->rd3;/* m=offset/shift between [-emf->rd3+1, emf->rd3] */
	if(loop==0) xx[i] = emf->x3s[i3+m]-emf->x3n[i3];//non-shifted point as the center
	else        xx[i] = emf->x3n[i3+1+m]-emf->x3s[i3];//non-shifted point as the center
      }
      for(i=0; i<2*emf->rd3; i++){
	//prepare a different rhs for Vandemonde matrix
	memset(ff, 0, 2*emf->rd3*sizeof(float));
	ff[i] =  1.;
	vandermonde(2*emf->rd3-1, xx, aa, ff);//invert vandemonde matrix system
	if(loop==0) emf->v3[i3][i] = aa[1]; //take coefficients to extract derivatives f'(x)
	else        emf->v3s[i3][i] = aa[1];
      }
    }
  }
  
  free1float(xx);
  free1float(aa);
  free1float(ff);
}

void nugrid_close(emf_t *emf)
{
  free1float(emf->x1n);
  free1float(emf->x2n);
  free1float(emf->x3n);
  free1float(emf->x1s);
  free1float(emf->x2s);
  free1float(emf->x3s);

  free2float(emf->v1);
  free2float(emf->v2);
  free2float(emf->v3);
  free2float(emf->v1s);
  free2float(emf->v2s);
  free2float(emf->v3s);

}


void sanity_check(emf_t *emf)
{
  int i1, i2, i3, i1_, i2_, i3_, i;
  float rho_water, rho_formation, tmp;
  float tmp1, tmp2, Rmax, cfl;
  float s1, s2, s3, t1, t2, t3;
  float D1, D2, D3, kappa;
  
  /*-------------------------------------------------------------------*/
  /* Stage 1: find the water resistivity and top formation resistivity */
  /*-------------------------------------------------------------------*/
  rho_water = emf->rho11[0][0][0];
  for(i=0; i<emf->n1*emf->n2*emf->n3; i++){
    i1 = i%emf->n1;
    i2 = i/emf->n1%emf->n2;
    i3 = i/(emf->n1*emf->n2);
    tmp = emf->rho11[i3][i2][i1];
    if(tmp>rho_water*3) {
      rho_formation = tmp;
      break;
    }
  }
  emf->sigma_water = 1./rho_water;
  emf->sigma_formation = 1./rho_formation;
  if(iproc==0){
    printf("rho_water=%g \n", rho_water);
    printf("rho_formation=%g \n", rho_formation);
  }

  /*----------------------------------------------------------------------------------*/
  /* Stage 2: find minimum and maximum velocity for stability conditon and dispersion */
  /*    emf->vmin: important for minimum number of points per wavelength              */
  /*    emf->vmax: important for CFL condition and fdtd computing box                 */
  /*----------------------------------------------------------------------------------*/
  emf->rhomax = MAX( MAX(emf->rho11[0][0][0], emf->rho22[0][0][0]), emf->rho33[0][0][0]);
  emf->rhomin = MIN( MIN(emf->rho11[0][0][0], emf->rho22[0][0][0]), emf->rho33[0][0][0]);
  /* sigma=2*omega0*eps --> inveps=2*omega0*rho */
  emf->vmin = sqrt(2.*emf->omega0*emf->rhomin*invmu0);
  emf->vmax = sqrt(2.*emf->omega0*emf->rhomax*invmu0);
  kappa = 0;
  for(i3=0; i3<emf->n3; ++i3){
    i3_ = i3+emf->nbe;
    for(i2=0; i2<emf->n2; ++i2){
      i2_ = i2+emf->nbe;
      for(i1=0; i1<emf->n1; ++i1){
	i1_ = i1+emf->nbe;

	tmp1 = MIN( MIN(emf->rho11[i3][i2][i1], emf->rho22[i3][i2][i1]), emf->rho33[i3][i2][i1]);
	tmp2 = MAX( MAX(emf->rho11[i3][i2][i1], emf->rho22[i3][i2][i1]), emf->rho33[i3][i2][i1]);
	if(emf->rhomin>tmp1)  emf->rhomin = tmp1;
	if(emf->rhomax<tmp2)  emf->rhomax = tmp2;
	emf->vmin = sqrt(2.*emf->omega0*emf->rhomin*invmu0);
	emf->vmax = sqrt(2.*emf->omega0*emf->rhomax*invmu0);

	s1 = 0;
	s2 = 0;
	s3 = 0;
	t1 = 0;
	t2 = 0;
	t3 = 0;
	for(i=0; i<2*emf->rd; i++) {
	  if(i<2*emf->rd1){
	    s1 += fabs(emf->v1[i1_][i]);
	    t1 += fabs(emf->v1s[i1_][i]);
	  }
	  if(i<2*emf->rd2){
	    s2 += fabs(emf->v2[i2_][i]);
	    t2 += fabs(emf->v2s[i2_][i]);
	  }
	  if(i<2*emf->rd3){
	    s3 += fabs(emf->v3[i3_][i]);
	    t3 += fabs(emf->v3s[i3_][i]);
	  }
	}
	D1 = MAX(s1, t1);
	D2 = MAX(s2, t2);
	D3 = MAX(s3, t3);
	tmp = 0.5*sqrt(D1*D1 + D2*D2 + D3*D3);
	tmp *= emf->vmax;
	if(tmp>kappa) kappa = tmp;
      }
    }
  }

  /*------------------------------------------------------------------------*/
  /* Stage 3: determine the optimal dt and nt automatically                 */
  /*------------------------------------------------------------------------*/
  if(!getparfloat("dt", &emf->dt)) emf->dt = 0.99/kappa;
  /* temporal sampling, determine dt by stability condition if not provided */
  cfl = emf->dt*kappa;
  if(iproc==0) printf("cfl=%g\n", cfl); 
  if(cfl > 1.0) err("CFL condition not satisfied!");

  if(!getparfloat("freqmax", &emf->freqmax)) {
    emf->freqmax = emf->vmin/(emf->Glim*MAX(MAX(emf->dx1min, emf->dx2min), emf->dx3min));
  }
  
  if(!getparint("nt", &emf->nt)){
    Rmax = MAX(emf->x1nu[emf->n1-1]-emf->x1nu[0], emf->x2nu[emf->n2-1]-emf->x2nu[0]);
    printf("Rmax=%g\n", Rmax);
    emf->nt =1.5*Rmax/(emf->vmin*emf->dt);
  }/* automatically determine nt using maximum offset if not provided */
  if(iproc==0){
    printf("[rhomin, rhomax]=[%g, %g] Ohm-m\n", emf->rhomin, emf->rhomax);
    printf("[vmin, vmax]=[%g, %g] m/s\n", emf->vmin, emf->vmax);
    printf("freqmax=%g Hz\n", emf->freqmax);
    printf("dt=%g s\n",  emf->dt);
    printf("nt=%d\n",  emf->nt);
  }

}

void cpml_init(emf_t *emf)
/*< initialize PML abosorbing coefficients >*/
{
  float x, damp0, damp;// L;
  int i1, i2, i3;

  /* by default, we choose: kappa=1, alpha=PI*emf->f0 for CPML */
  float alpha=PI*emf->f0; /* alpha>0 makes CPML effectively attenuates evanescent waves */
  //const float Rc = 1e-5; /* theoretic reflection coefficient for PML */

  emf->a1 = alloc1float(emf->nb);
  emf->a2 = alloc1float(emf->nb);
  emf->a3 = alloc1float(emf->nb);
  emf->b1 = alloc1float(emf->nb);
  emf->b2 = alloc1float(emf->nb);
  emf->b3 = alloc1float(emf->nb);

  //L=emf->nb*(emf->x1nu[1]- emf->x1nu[0]);
  damp0= 349.1; //-3.*emf->vmax*logf(Rc)/(2.*L);
  for(i1=0; i1<emf->nb; ++i1)    {
    x=(float)(emf->nb-i1)/emf->nb;
    damp = damp0*x*x; /* damping profile in direction 1, sigma/epsilon0 */
    // damp = damp0*(1.0-cos(0.5*PI*x));
    emf->b1[i1] = expf(-(damp+alpha)*emf->dt);
    emf->a1[i1] = damp*(emf->b1[i1]-1.0)/(damp+alpha);
  }

  //L= emf->nb*(emf->x2nu[1]- emf->x2nu[0]);    
  //damp0 =-3.*emf->vmax*logf(Rc)/(2.*L);
  for(i2=0; i2<emf->nb; ++i2)    {
    x=(float)(emf->nb-i2)/emf->nb;
    damp = damp0*x*x;/* damping profile in direction 2, sigma/epsilon0 */
    //damp = damp0*(1.0-cos(0.5*PI*x));
    emf->b2[i2] = expf(-(damp+alpha)*emf->dt);
    emf->a2[i2] = damp*(emf->b2[i2]-1.0)/(damp+alpha);
  }

  //L=emf->nb*(emf->x3nu[1]- emf->x3nu[0]);    
  //damp0=-3.*emf->vmax*logf(Rc)/(2.*L);
  for(i3=0; i3<emf->nb; ++i3)    {
    x=(float)(emf->nb-i3)/emf->nb;
    damp = damp0*x*x;/* damping profile in direction 3, sigma/epsilon0 */
    //damp = damp0*(1.0-cos(0.5*PI*x));
    emf->b3[i3] = expf(-(damp+alpha)*emf->dt);
    emf->a3[i3] = damp*(emf->b3[i3]-1.0)/(damp+alpha);
  }
}

void cpml_close(emf_t *emf)
{
  free(emf->a1);
  free(emf->a2);
  free(emf->a3);
  free(emf->b1);
  free(emf->b2);
  free(emf->b3);
}


void fdtd_init(emf_t *emf)
{
  cpml_init(emf);
  
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
  cpml_close(emf);
  
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


void nufdtd_curlE(emf_t *emf)
{
  int i1min = emf->rd1-1;
  int i2min = emf->rd2-1;
  int i3min = emf->airwave?emf->nbe:emf->rd3-1; 
  int i1max = emf->n1pad-1-emf->rd1;
  int i2max = emf->n2pad-1-emf->rd2;
  int i3max = emf->n3pad-1-emf->rd3;

  int i1, i2, i3, j1, j2, j3, k1, k2, k3;
  float D2E3, D3E2, D3E1, D1E3, D1E2, D2E1;

#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3, D2E3, D3E2, D3E1, D1E3, D1E2, D2E1) \
  shared(i1min, i1max, i2min, i2max, i3min, i3max, emf)
#endif
  for(i3=i3min; i3<=i3max; ++i3){
    for(i2=i2min; i2<=i2max; ++i2){
      for(i1=i1min; i1<=i1max; ++i1){
	//unroll the above for loop
	if(emf->rd1==1){
	  D1E3 = emf->v1[i1][0]*emf->E3[i3][i2][i1]
	    + emf->v1[i1][1]*emf->E3[i3][i2][i1+1];
	  D1E2 = emf->v1[i1][0]*emf->E2[i3][i2][i1]
	    + emf->v1[i1][1]*emf->E2[i3][i2][i1+1];
	}else if(emf->rd1==2){
	  D1E3 = emf->v1[i1][0]*emf->E3[i3][i2][i1-1]
	    + emf->v1[i1][1]*emf->E3[i3][i2][i1]
	    + emf->v1[i1][2]*emf->E3[i3][i2][i1+1]
	    + emf->v1[i1][3]*emf->E3[i3][i2][i1+2];
	  D1E2 = emf->v1[i1][0]*emf->E2[i3][i2][i1-1]
	    + emf->v1[i1][1]*emf->E2[i3][i2][i1]
	    + emf->v1[i1][2]*emf->E2[i3][i2][i1+1]
	    + emf->v1[i1][3]*emf->E2[i3][i2][i1+2];
	}else if(emf->rd1==3){
	  D1E3 = emf->v1[i1][0]*emf->E3[i3][i2][i1-2]
	    + emf->v1[i1][1]*emf->E3[i3][i2][i1-1]
	    + emf->v1[i1][2]*emf->E3[i3][i2][i1]
	    + emf->v1[i1][3]*emf->E3[i3][i2][i1+1]
	    + emf->v1[i1][4]*emf->E3[i3][i2][i1+2]
	    + emf->v1[i1][5]*emf->E3[i3][i2][i1+3];
	  D1E2 = emf->v1[i1][0]*emf->E2[i3][i2][i1-2]
	    + emf->v1[i1][1]*emf->E2[i3][i2][i1-1]
	    + emf->v1[i1][2]*emf->E2[i3][i2][i1]
	    + emf->v1[i1][3]*emf->E2[i3][i2][i1+1]
	    + emf->v1[i1][4]*emf->E2[i3][i2][i1+2]
	    + emf->v1[i1][5]*emf->E2[i3][i2][i1+3];
	}

	if(emf->rd2==1){
	  D2E3 = emf->v2[i2][0]*emf->E3[i3][i2][i1]
	    + emf->v2[i2][1]*emf->E3[i3][i2+1][i1];
	  D2E1 = emf->v2[i2][0]*emf->E1[i3][i2][i1]
	    + emf->v2[i2][1]*emf->E1[i3][i2+1][i1];
	}else if(emf->rd2==2){
	  D2E3 = emf->v2[i2][0]*emf->E3[i3][i2-1][i1]
	    + emf->v2[i2][1]*emf->E3[i3][i2][i1]
	    + emf->v2[i2][2]*emf->E3[i3][i2+1][i1]
	    + emf->v2[i2][3]*emf->E3[i3][i2+2][i1];
	  D2E1 = emf->v2[i2][0]*emf->E1[i3][i2-1][i1]
	    + emf->v2[i2][1]*emf->E1[i3][i2][i1]
	    + emf->v2[i2][2]*emf->E1[i3][i2+1][i1]
	    + emf->v2[i2][3]*emf->E1[i3][i2+2][i1];
	}else if(emf->rd2==3){
	  D2E3 = emf->v2[i2][0]*emf->E3[i3][i2-2][i1]
	    + emf->v2[i2][1]*emf->E3[i3][i2-1][i1]
	    + emf->v2[i2][2]*emf->E3[i3][i2][i1]
	    + emf->v2[i2][3]*emf->E3[i3][i2+1][i1]
	    + emf->v2[i2][4]*emf->E3[i3][i2+2][i1]
	    + emf->v2[i2][5]*emf->E3[i3][i2+3][i1];
	  D2E1 = emf->v2[i2][0]*emf->E1[i3][i2-2][i1]
	    + emf->v2[i2][1]*emf->E1[i3][i2-1][i1]
	    + emf->v2[i2][2]*emf->E1[i3][i2][i1]
	    + emf->v2[i2][3]*emf->E1[i3][i2+1][i1]
	    + emf->v2[i2][4]*emf->E1[i3][i2+2][i1]
	    + emf->v2[i2][5]*emf->E1[i3][i2+3][i1];
	}

	if(emf->rd3==1){
	  D3E2 = emf->v3[i3][0]*emf->E2[i3][i2][i1]
	    + emf->v3[i3][1]*emf->E2[i3+1][i2][i1];
	  D3E1 = emf->v3[i3][0]*emf->E1[i3][i2][i1]
	    + emf->v3[i3][1]*emf->E1[i3+1][i2][i1];
	}else if(emf->rd3==2){
	  D3E2 = emf->v3[i3][0]*emf->E2[i3-1][i2][i1]
	    + emf->v3[i3][1]*emf->E2[i3][i2][i1]
	    + emf->v3[i3][2]*emf->E2[i3+1][i2][i1]
	    + emf->v3[i3][3]*emf->E2[i3+2][i2][i1];
	  D3E1 = emf->v3[i3][0]*emf->E1[i3-1][i2][i1]
	    + emf->v3[i3][1]*emf->E1[i3][i2][i1]
	    + emf->v3[i3][2]*emf->E1[i3+1][i2][i1]
	    + emf->v3[i3][3]*emf->E1[i3+2][i2][i1];
	}else if(emf->rd3==3){
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

	/* CPML: mem=memory variable */
	if(i1<emf->nb){
	  emf->memD1E3[i3][i2][i1] = emf->b1[i1]*emf->memD1E3[i3][i2][i1] + emf->a1[i1]*D1E3;
	  emf->memD1E2[i3][i2][i1] = emf->b1[i1]*emf->memD1E2[i3][i2][i1] + emf->a1[i1]*D1E2;
	  D1E3 += emf->memD1E3[i3][i2][i1];
	  D1E2 += emf->memD1E2[i3][i2][i1];
	}else if(i1>emf->n1pad-1-emf->nb){
	  j1 = emf->n1pad-1-i1;
	  k1 = j1+emf->nb;
	  emf->memD1E3[i3][i2][k1] = emf->b1[j1]*emf->memD1E3[i3][i2][k1] + emf->a1[j1]*D1E3;
	  emf->memD1E2[i3][i2][k1] = emf->b1[j1]*emf->memD1E2[i3][i2][k1] + emf->a1[j1]*D1E2;
	  D1E3 += emf->memD1E3[i3][i2][k1];
	  D1E2 += emf->memD1E2[i3][i2][k1];
	}
	if(i2<emf->nb){
	  emf->memD2E3[i3][i2][i1] = emf->b2[i2]*emf->memD2E3[i3][i2][i1] + emf->a2[i2]*D2E3;
	  emf->memD2E1[i3][i2][i1] = emf->b2[i2]*emf->memD2E1[i3][i2][i1] + emf->a2[i2]*D2E1;
	  D2E3 += emf->memD2E3[i3][i2][i1];
	  D2E1 += emf->memD2E1[i3][i2][i1];
	}else if(i2>emf->n2pad-1-emf->nb){
	  j2 = emf->n2pad-1-i2;
	  k2 = j2+emf->nb;
	  emf->memD2E3[i3][k2][i1] = emf->b2[j2]*emf->memD2E3[i3][k2][i1] + emf->a2[j2]*D2E3;
	  emf->memD2E1[i3][k2][i1] = emf->b2[j2]*emf->memD2E1[i3][k2][i1] + emf->a2[j2]*D2E1;
	  D2E3 += emf->memD2E3[i3][k2][i1];
	  D2E1 += emf->memD2E1[i3][k2][i1];
	}
	if(i3<emf->nb){
	  emf->memD3E2[i3][i2][i1] = emf->b3[i3]*emf->memD3E2[i3][i2][i1] + emf->a3[i3]*D3E2;
	  emf->memD3E1[i3][i2][i1] = emf->b3[i3]*emf->memD3E1[i3][i2][i1] + emf->a3[i3]*D3E1;
	  D3E2 += emf->memD3E2[i3][i2][i1];
	  D3E1 += emf->memD3E1[i3][i2][i1];
	}else if(i3>emf->n3pad-1-emf->nb){
	  j3 = emf->n3pad-1-i3;
	  k3 = j3+emf->nb;
	  emf->memD3E2[k3][i2][i1] = emf->b3[j3]*emf->memD3E2[k3][i2][i1] + emf->a3[j3]*D3E2;
	  emf->memD3E1[k3][i2][i1] = emf->b3[j3]*emf->memD3E1[k3][i2][i1] + emf->a3[j3]*D3E1;
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

void nufdtd_update_H(emf_t *emf)
{
  int i1, i2, i3;

  int i1min = 0;
  int i2min = 0;
  int i3min = emf->airwave?emf->nbe:0;
  int i1max = emf->n1pad-1;
  int i2max = emf->n2pad-1;
  int i3max = emf->n3pad-1;
  float factor = emf->dt*invmu0;

#ifdef _OPENMP
#pragma omp parallel for default(none)				\
  schedule(static)						\
  private(i1, i2, i3)						\
  shared(i1min, i1max, i2min, i2max, i3min, i3max, factor, emf)
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


void nufdtd_curlH(emf_t *emf)
{
  int i1min = emf->rd1;
  int i2min = emf->rd2;
  int i3min = emf->airwave?emf->nbe:emf->rd3;
  int i1max = emf->n1pad-emf->rd1;
  int i2max = emf->n2pad-emf->rd2;
  int i3max = emf->n3pad-emf->rd3;

  int i1, i2, i3, j1, j2, j3, k1, k2, k3;
  float D2H3, D3H2, D3H1, D1H3, D1H2, D2H1;

#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3, D2H3, D3H2, D3H1, D1H3, D1H2, D2H1) \
  shared(i1min, i1max, i2min, i2max, i3min, i3max, emf)
#endif
  for(i3=i3min; i3<=i3max; ++i3){
    for(i2=i2min; i2<=i2max; ++i2){
      for(i1=i1min; i1<=i1max; ++i1){
	//unroll the above loop
	if(emf->rd1==1){
	  D1H3 = emf->v1s[i1][0]*emf->H3[i3][i2][i1-1]
	    + emf->v1s[i1][1]*emf->H3[i3][i2][i1];
	  D1H2 = emf->v1s[i1][0]*emf->H2[i3][i2][i1-1]
	    + emf->v1s[i1][1]*emf->H2[i3][i2][i1];
	}else if(emf->rd1==2){
	  D1H3 = emf->v1s[i1][0]*emf->H3[i3][i2][i1-2]
	    + emf->v1s[i1][1]*emf->H3[i3][i2][i1-1]
	    + emf->v1s[i1][2]*emf->H3[i3][i2][i1]
	    + emf->v1s[i1][3]*emf->H3[i3][i2][i1+1];
	  D1H2 = emf->v1s[i1][0]*emf->H2[i3][i2][i1-2]
	    + emf->v1s[i1][1]*emf->H2[i3][i2][i1-1]
	    + emf->v1s[i1][2]*emf->H2[i3][i2][i1]
	    + emf->v1s[i1][3]*emf->H2[i3][i2][i1+1];
	}else if(emf->rd1==3){
	  D1H3 = emf->v1s[i1][0]*emf->H3[i3][i2][i1-3]
	    + emf->v1s[i1][1]*emf->H3[i3][i2][i1-2]
	    + emf->v1s[i1][2]*emf->H3[i3][i2][i1-1]
	    + emf->v1s[i1][3]*emf->H3[i3][i2][i1]
	    + emf->v1s[i1][4]*emf->H3[i3][i2][i1+1]
	    + emf->v1s[i1][5]*emf->H3[i3][i2][i1+2];
	  D1H2 = emf->v1s[i1][0]*emf->H2[i3][i2][i1-3]
	    + emf->v1s[i1][1]*emf->H2[i3][i2][i1-2]
	    + emf->v1s[i1][2]*emf->H2[i3][i2][i1-1]
	    + emf->v1s[i1][3]*emf->H2[i3][i2][i1]
	    + emf->v1s[i1][4]*emf->H2[i3][i2][i1+1]
	    + emf->v1s[i1][5]*emf->H2[i3][i2][i1+2];
	}

	if(emf->rd2==1){
	  D2H3 = emf->v2s[i2][0]*emf->H3[i3][i2-1][i1]
	    + emf->v2s[i2][1]*emf->H3[i3][i2][i1];
	  D2H1 = emf->v2s[i2][0]*emf->H1[i3][i2-1][i1]
	    + emf->v2s[i2][1]*emf->H1[i3][i2][i1];
	}else if(emf->rd2==2){
	  D2H3 = emf->v2s[i2][0]*emf->H3[i3][i2-2][i1]
	    + emf->v2s[i2][1]*emf->H3[i3][i2-1][i1]
	    + emf->v2s[i2][2]*emf->H3[i3][i2][i1]
	    + emf->v2s[i2][3]*emf->H3[i3][i2+1][i1];
	  D2H1 = emf->v2s[i2][0]*emf->H1[i3][i2-2][i1]
	    + emf->v2s[i2][1]*emf->H1[i3][i2-1][i1]
	    + emf->v2s[i2][2]*emf->H1[i3][i2][i1]
	    + emf->v2s[i2][3]*emf->H1[i3][i2+1][i1];
	}else if(emf->rd2==3){
	  D2H3 = emf->v2s[i2][0]*emf->H3[i3][i2-3][i1]
	    + emf->v2s[i2][1]*emf->H3[i3][i2-2][i1]
	    + emf->v2s[i2][2]*emf->H3[i3][i2-1][i1]
	    + emf->v2s[i2][3]*emf->H3[i3][i2][i1]
	    + emf->v2s[i2][4]*emf->H3[i3][i2+1][i1]
	    + emf->v2s[i2][5]*emf->H3[i3][i2+2][i1];
	  D2H1 = emf->v2s[i2][0]*emf->H1[i3][i2-3][i1]
	    + emf->v2s[i2][1]*emf->H1[i3][i2-2][i1]
	    + emf->v2s[i2][2]*emf->H1[i3][i2-1][i1]
	    + emf->v2s[i2][3]*emf->H1[i3][i2][i1]
	    + emf->v2s[i2][4]*emf->H1[i3][i2+1][i1]
	    + emf->v2s[i2][5]*emf->H1[i3][i2+2][i1];
	}

	if(emf->rd3==1){
	  D3H2 = emf->v3s[i3][0]*emf->H2[i3-1][i2][i1]
	    + emf->v3s[i3][1]*emf->H2[i3][i2][i1];
	  D3H1 = emf->v3s[i3][0]*emf->H1[i3-1][i2][i1]
	    + emf->v3s[i3][1]*emf->H1[i3][i2][i1];
	}else if(emf->rd3==2){
	  D3H2 = emf->v3s[i3][0]*emf->H2[i3-2][i2][i1]
	    + emf->v3s[i3][1]*emf->H2[i3-1][i2][i1]
	    + emf->v3s[i3][2]*emf->H2[i3][i2][i1]
	    + emf->v3s[i3][3]*emf->H2[i3+1][i2][i1];
	  D3H1 = emf->v3s[i3][0]*emf->H1[i3-2][i2][i1]
	    + emf->v3s[i3][1]*emf->H1[i3-1][i2][i1]
	    + emf->v3s[i3][2]*emf->H1[i3][i2][i1]
	    + emf->v3s[i3][3]*emf->H1[i3+1][i2][i1];
	}else if(emf->rd3==3){
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

	/* CPML: mem=memory variable */
	if(i1<emf->nb){
	  emf->memD1H3[i3][i2][i1] = emf->b1[i1]*emf->memD1H3[i3][i2][i1] + emf->a1[i1]*D1H3;
	  emf->memD1H2[i3][i2][i1] = emf->b1[i1]*emf->memD1H2[i3][i2][i1] + emf->a1[i1]*D1H2;
	  D1H3 += emf->memD1H3[i3][i2][i1];
	  D1H2 += emf->memD1H2[i3][i2][i1];
	}else if(i1>emf->n1pad-1-emf->nb){
	  j1 = emf->n1pad-1-i1;
	  k1 = j1+emf->nb;
	  emf->memD1H3[i3][i2][k1] = emf->b1[j1]*emf->memD1H3[i3][i2][k1] + emf->a1[j1]*D1H3;
	  emf->memD1H2[i3][i2][k1] = emf->b1[j1]*emf->memD1H2[i3][i2][k1] + emf->a1[j1]*D1H2;
	  D1H3 += emf->memD1H3[i3][i2][k1];
	  D1H2 += emf->memD1H2[i3][i2][k1];
	}
	if(i2<emf->nb){
	  emf->memD2H3[i3][i2][i1] = emf->b2[i2]*emf->memD2H3[i3][i2][i1] + emf->a2[i2]*D2H3;
	  emf->memD2H1[i3][i2][i1] = emf->b2[i2]*emf->memD2H1[i3][i2][i1] + emf->a2[i2]*D2H1;
	  D2H3 += emf->memD2H3[i3][i2][i1];
	  D2H1 += emf->memD2H1[i3][i2][i1];
	}else if(i2>emf->n2pad-1-emf->nb){
	  j2 = emf->n2pad-1-i2;
	  k2 = j2+emf->nb;
	  emf->memD2H3[i3][k2][i1] = emf->b2[j2]*emf->memD2H3[i3][k2][i1] + emf->a2[j2]*D2H3;
	  emf->memD2H1[i3][k2][i1] = emf->b2[j2]*emf->memD2H1[i3][k2][i1] + emf->a2[j2]*D2H1;
	  D2H3 += emf->memD2H3[i3][k2][i1];
	  D2H1 += emf->memD2H1[i3][k2][i1];
	}
	if(i3<emf->nb){
	  emf->memD3H2[i3][i2][i1] = emf->b3[i3]*emf->memD3H2[i3][i2][i1] + emf->a3[i3]*D3H2;
	  emf->memD3H1[i3][i2][i1] = emf->b3[i3]*emf->memD3H1[i3][i2][i1] + emf->a3[i3]*D3H1;
	  D3H2 += emf->memD3H2[i3][i2][i1];
	  D3H1 += emf->memD3H1[i3][i2][i1];
	}else if(i3>emf->n3pad-1-emf->nb){
	  j3 = emf->n3pad-1-i3;
	  k3 = j3+emf->nb;
	  emf->memD3H2[k3][i2][i1] = emf->b3[j3]*emf->memD3H2[k3][i2][i1] + emf->a3[j3]*D3H2;
	  emf->memD3H1[k3][i2][i1] = emf->b3[j3]*emf->memD3H1[k3][i2][i1] + emf->a3[j3]*D3H1;
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

void nufdtd_update_E(emf_t *emf)
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
	emf->E1[i3][i2][i1] += emf->dt*emf->inveps11[i3][i2][i1]* emf->curlH1[i3][i2][i1];
	emf->E2[i3][i2][i1] += emf->dt*emf->inveps22[i3][i2][i1]* emf->curlH2[i3][i2][i1];
	emf->E3[i3][i2][i1] += emf->dt*emf->inveps33[i3][i2][i1]* emf->curlH3[i3][i2][i1];
      }
    }
  }
}


/* find the index k in x[] such that x[k]<= val <x[k+1] 
 *
 * Copyright (c) 2020 Pengliang Yang. All rights reserved.
 * Email: ypl.2100@gmail.com
 */
int find_index(int n, float *x, float val)
{
  /*assume x[] has been sorted ascendingly */
  if(val<=x[0]) return 0;//warning: should not happen
  if(val>=x[n-1]) return n-1;

  int low = 0;
  int high = n-1;
  int i = (low+high)/2;
  
  while(low<high){
    i=(low+high)/2;
    if(x[i]<=val) low = i;
    if(x[i]>val) high=i;
    if(low==high||low==high-1) break;
  }

  return low;
}


fftw_complex *H1kxkyz0, *H2kxkyz0, *kxky, *kxkyz0;
fftw_plan fft_airwave, ifft_airwave;

int fft_next_fast_size(int n)
{
  int m;
  int p = 2*n;
    
  /* m = 1; */
  /* while(m<p) m *= 2; */
  /* return m; */

  while(1) {
    m=p;
    while ( (m%2) == 0 ) m/=2;
    while ( (m%3) == 0 ) m/=3;
    while ( (m%5) == 0 ) m/=5;
    if (m<=1)
      break; /* n is completely factorable by twos, threes, and fives */
    p++;
  }
  return p;
}


/*--------------------------------------------------------------------------*/
void airwave_bc_init(emf_t *emf)
{
  float dkx, dky, o1, o2;
  int i1, i2, j1, j2;
  float x0;
  const float eps = 1e-15;

  emf->d1uni = emf->dx1min; //0.5*(emf->dx1min+emf->dx1_start);
  emf->d2uni = emf->dx2min; //0.5*(emf->dx2min+emf->dx2_start);
  emf->d3uni = emf->dx3min;
  emf->n1uni = ceilf( (emf->x1n[emf->n1pad-1] - emf->x1n[0])/emf->d1uni + 1);
  emf->n2uni = ceilf( (emf->x2n[emf->n2pad-1] - emf->x2n[0])/emf->d2uni + 1);
  if(iproc==0){
    printf("Airwave uniform grid: d1uni=%g d2uni=%g \n", emf->d1uni, emf->d2uni);
    printf("Uniform grid size: n1uni=%d n2uni=%d\n", emf->n1uni, emf->n2uni);
  }
  
  /*-----------------------------------------------------*/
  /* determine the closest index on nonuniform grid for a point in uniform grid */
  /*-----------------------------------------------------*/
  emf->nu_i1 = alloc1int(emf->n1uni);
  emf->nu_i2 = alloc1int(emf->n2uni);
  emf->nu_i1_s = alloc1int(emf->n1uni);/* shift half cell on uniform grid in x coordinate */
  emf->nu_i2_s = alloc1int(emf->n2uni);/* shift half cell on uniform grid in x coordinate */
  o1 = emf->x1n[0];
  for(i1=0; i1<emf->n1uni; i1++){
    x0 = o1 + i1*emf->d1uni;
    j1 = find_index(emf->n1pad, emf->x1n, x0);
    emf->nu_i1[i1] = j1;//index on NU grid
  }
  o1 = emf->x1n[0] + 0.5*emf->d1uni;
  for(i1=0; i1<emf->n1uni; i1++){
    x0 = o1 + i1*emf->d1uni;
    j1 = find_index(emf->n1pad, emf->x1s, x0);
    emf->nu_i1_s[i1] = j1;
  }
  o2 = emf->x2n[0];
  for(i2=0; i2<emf->n2uni; i2++){
    x0 = o2 + i2*emf->d2uni;
    j2 = find_index(emf->n2pad, emf->x2n, x0);
    emf->nu_i2[i2] = j2;//index on NU grid
  }
  o2 = emf->x2n[0] +0.5*emf->d2uni;
  for(i2=0; i2<emf->n2uni; i2++){
    x0 = o2 + i2*emf->d2uni;
    j2 = find_index(emf->n2pad, emf->x2s, x0);
    emf->nu_i2_s[i2] = j2;
  }
  emf->uni_H1 = alloc3float(emf->n1uni, emf->n2uni, emf->rd3);
  emf->uni_H2 = alloc3float(emf->n1uni, emf->n2uni, emf->rd3);
  emf->uni_E1 = alloc3float(emf->n1uni, emf->n2uni, emf->rd3-1);
  emf->uni_E2 = alloc3float(emf->n1uni, emf->n2uni, emf->rd3-1);
  memset(emf->uni_H1[0][0], 0, emf->n1uni*emf->n2uni*emf->rd3*sizeof(float));
  memset(emf->uni_H2[0][0], 0, emf->n1uni*emf->n2uni*emf->rd3*sizeof(float));
  if(emf->rd3>1){
    memset(emf->uni_E1[0][0], 0, emf->n1uni*emf->n2uni*(emf->rd3-1)*sizeof(float));
    memset(emf->uni_E2[0][0], 0, emf->n1uni*emf->n2uni*(emf->rd3-1)*sizeof(float));
  }

  /*----------------------------------------------------------*/
  emf->n1fft = fft_next_fast_size(emf->n1uni);
  emf->n2fft = fft_next_fast_size(emf->n2uni);
  if(iproc==0) printf("[n1fft, n2fft]=[%d, %d]\n", emf->n1fft, emf->n2fft);
  
  emf->kx = alloc1float(emf->n1fft);
  emf->ky = alloc1float(emf->n2fft);
  emf->hw1 = alloc1float(emf->n1fft);
  emf->hw2 = alloc1float(emf->n2fft);
  emf->sqrtkx2ky2 = alloc2float(emf->n1fft, emf->n2fft);
  
  /* pre-compute the discrete wavenumber - kx */
  dkx=2.0*PI/(emf->d1uni*emf->n1fft);
  emf->kx[0]=0;
  for(i1=1; i1<(emf->n1fft+1)/2; i1++) {
    emf->kx[i1]=i1*dkx;
    emf->kx[emf->n1fft-i1]=-i1*dkx;
  }
  if(emf->n1fft%2==0) emf->kx[emf->n1fft/2] = (emf->n1fft/2)*dkx;/* Nyquist freq*/
  /* pre-compute the discrete wavenumber - ky */
  dky=2.0*PI/(emf->d2uni*emf->n2fft);
  emf->ky[0]=0;
  for(i2=1; i2<(emf->n2fft+1)/2; i2++) {
    emf->ky[i2]=i2*dky;
    emf->ky[emf->n2fft-i2]=-i2*dky;
  }
  if(emf->n2fft%2==0) emf->ky[emf->n2fft/2] = (emf->n2fft/2)*dky;/* Nyquist freq*/

  for(i1=0; i1<emf->n1fft; i1++) emf->hw1[i1] = 0.5*(1.+cos(2.*PI*i1/emf->n1fft));
  for(i2=0; i2<emf->n2fft; i2++) emf->hw2[i2] = 0.5*(1.+cos(2.*PI*i2/emf->n2fft));
  
  for(i2=0; i2<emf->n2fft; i2++){
    for(i1=0; i1<emf->n1fft; i1++){
      emf->sqrtkx2ky2[i2][i1] = sqrt(emf->kx[i1]*emf->kx[i1]+emf->ky[i2]*emf->ky[i2] + eps);
    }
  }

  /* FE3 is not necessary in the air because we do not compute derivates of Hx
   * and Hy in the air: Hx and Hy are derived directly by extrapolation from Hz. */
  kxky=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*emf->n1fft*emf->n2fft);
  kxkyz0=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*emf->n1fft*emf->n2fft);
  H1kxkyz0 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*emf->n1fft*emf->n2fft);
  H2kxkyz0 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*emf->n1fft*emf->n2fft);
  /* comapred with FFTW, we have opposite sign convention for time, same sign convetion for space */
  fft_airwave=fftw_plan_dft_2d(emf->n1fft, emf->n2fft, kxky, kxky, FFTW_FORWARD, FFTW_ESTIMATE);
  ifft_airwave=fftw_plan_dft_2d(emf->n1fft, emf->n2fft, kxky, kxky, FFTW_BACKWARD, FFTW_ESTIMATE);
}

void airwave_bc_close(emf_t *emf)
{
  free3float(emf->uni_H1);
  free3float(emf->uni_H2);
  if(emf->rd3>1){
    free3float(emf->uni_E1);
    free3float(emf->uni_E2);
  }
  free1int(emf->nu_i1);
  free1int(emf->nu_i2);
  free1int(emf->nu_i1_s);
  free1int(emf->nu_i2_s);


  free(emf->kx);
  free(emf->ky);
  free(emf->hw1);
  free(emf->hw2);
  free2float(emf->sqrtkx2ky2);
  fftw_free(kxky);
  fftw_free(kxkyz0);
  fftw_free(H1kxkyz0);
  fftw_free(H2kxkyz0);
  fftw_destroy_plan(fft_airwave);
  fftw_destroy_plan(ifft_airwave);
}


void airwave_bc_update_H(emf_t *emf)
{
  int i1, i2, i3, i1_, i2_;
  float w1, w2, o1, o2, s;
  bool ok1, ok2, ok1p, ok2p;
  
  memset(kxky, 0, emf->n1fft*emf->n2fft*sizeof(fftw_complex));
  i3 = emf->nbe;
  //interpolation from non-uniform grid to uniform grid
  //Hz is staggered in both x and y direction
  o1 = emf->x1n[0] + 0.5*emf->d1uni;
  o2 = emf->x2n[0] + 0.5*emf->d2uni;
  for(i2=0; i2<emf->n2uni; i2++){
    i2_ = emf->nu_i2_s[i2];
    ok2 = (i2_>=0 && i2_+1<emf->n2pad)?true:false;
    w2 = ok2?(o2 + i2*emf->d2uni -emf->x2s[i2_])/(emf->x2s[i2_+1]-emf->x2s[i2_]):0;
    ok2 = (i2_>=0 && i2_<emf->n2pad)?true:false;
    ok2p = (i2_+1>=0 && i2_+1<emf->n2pad)?true:false;
    for(i1=0; i1<emf->n1uni; i1++){
      i1_ = emf->nu_i1_s[i1];
      ok1 = (i1_>=0 && i1_+1<emf->n1pad)?true:false;
      w1 = ok1?(o1 + i1*emf->d1uni -emf->x1s[i1_])/(emf->x1s[i1_+1]-emf->x1s[i1_]):0;
      ok1 = (i1_>=0 && i1_<emf->n1pad)?true:false;
      ok1p = (i1_+1>=0 && i1_+1<emf->n1pad)?true:false;

      s = 0;
      if(ok1 && ok2) s += emf->H3[i3][i2_][i1_]*(1.-w1)*(1.-w2);
      if(ok1p && ok2) s += emf->H3[i3][i2_][i1_+1]*w1*(1.-w2);
      if(ok1 && ok2p) s += emf->H3[i3][i2_+1][i1_]*(1.-w1)*w2;
      if(ok1p && ok2p) s += emf->H3[i3][i2_+1][i1_+1]*w1*w2;
      kxky[i1+emf->n1fft*i2] = s;
    }
  }
  fftw_execute(fft_airwave);/*Hz(x, y, z=0)-->Hz(kx, ky, z=0)*/
  memcpy(kxkyz0, kxky, emf->n1fft*emf->n2fft*sizeof(fftw_complex));
  /* for(i2=0; i2<emf->n2fft; i2++){ */
  /*   for(i1=0; i1<emf->n1fft; i1++){ */
  /*     kxkyz0[i1+emf->n1fft*i2] *= emf->hw1[i1]*emf->hw2[i2]; */
  /*   } */
  /* } */
  
  /*----------------------------------- H1 -------------------------------------*/
  for(i2=0; i2<emf->n2fft; i2++){
    for(i1=0; i1<emf->n1fft; i1++){
      /* at z=0 level */
      H1kxkyz0[i1+emf->n1fft*i2] = kxkyz0[i1+emf->n1fft*i2]*cexp(-I*emf->kx[i1]*0.5*emf->d1uni)*I*emf->kx[i1]/emf->sqrtkx2ky2[i2][i1];
      /* at z=-0.5*emf->d3 level */
      kxky[i1+emf->n1fft*i2] = H1kxkyz0[i1+emf->n1fft*i2]*exp(-emf->sqrtkx2ky2[i2][i1]*0.5*emf->d3uni);
    }
  }
  if(emf->rd3>=1){//order=2
    fftw_execute(ifft_airwave);/*Hz(x, y, z=0)-->Hz(kx, ky, z=0)*/
    for(i2=0; i2<emf->n2uni; i2++){
      for(i1=0; i1<emf->n1uni; i1++){
	emf->uni_H1[0][i2][i1] = creal(kxky[i1+emf->n1fft*i2]/(emf->n1fft*emf->n2fft));
      }
    }
  }
  if(emf->rd3>=2){//order=4
    for(i2=0; i2<emf->n2fft; i2++){
      for(i1=0; i1<emf->n1fft; i1++){
	/* at z=-1.5*emf->d3 level */
	kxky[i1+emf->n1fft*i2] = H1kxkyz0[i1+emf->n1fft*i2]*exp(-emf->sqrtkx2ky2[i2][i1]*1.5*emf->d3uni);
      }
    }
    fftw_execute(ifft_airwave);/*Hz(x, y, z=0)-->Hz(kx, ky, z=0)*/
    for(i2=0; i2<emf->n2uni; i2++){
      for(i1=0; i1<emf->n1uni; i1++){
	emf->uni_H1[1][i2][i1] = creal(kxky[i1+emf->n1fft*i2]/(emf->n1fft*emf->n2fft));
      }
    }
  }
  if(emf->rd3>=3){//order=6
    for(i2=0; i2<emf->n2fft; i2++){
      for(i1=0; i1<emf->n1fft; i1++){
	/* at z=-1.5*emf->d3 level */
	kxky[i1+emf->n1fft*i2] = H1kxkyz0[i1+emf->n1fft*i2]*exp(-emf->sqrtkx2ky2[i2][i1]*2.5*emf->d3uni);
      }
    }
    fftw_execute(ifft_airwave);/*Hz(x, y, z=0)-->Hz(kx, ky, z=0)*/
    for(i2=0; i2<emf->n2uni; i2++){
      for(i1=0; i1<emf->n1uni; i1++){
	emf->uni_H1[2][i2][i1] = creal(kxky[i1+emf->n1fft*i2]/(emf->n1fft*emf->n2fft));
      }
    }
  }

  /*----------------------------------- H2 -------------------------------------*/
  for(i2=0; i2<emf->n2fft; i2++){
    for(i1=0; i1<emf->n1fft; i1++){
      /* at z=0 level */
      H2kxkyz0[i1+emf->n1fft*i2] = kxkyz0[i1+emf->n1fft*i2]*cexp(-I*emf->ky[i2]*0.5*emf->d2uni)*I*emf->ky[i2]/emf->sqrtkx2ky2[i2][i1];
      /* at z=-0.5*emf->d3 level */
      kxky[i1+emf->n1fft*i2] = H2kxkyz0[i1+emf->n1fft*i2]*exp(-emf->sqrtkx2ky2[i2][i1]*0.5*emf->d3uni);
    }
  }
  if(emf->rd3>=1){//order=2
    fftw_execute(ifft_airwave);/*Hz(x, y, z=0)-->Hz(kx, ky, z=0)*/
    for(i2=0; i2<emf->n2uni; i2++){
      for(i1=0; i1<emf->n1uni; i1++){
	emf->uni_H2[0][i2][i1] = creal(kxky[i1+emf->n1fft*i2]/(emf->n1fft*emf->n2fft));
      }
    }
  }
  if(emf->rd3>=2){//order=4
    for(i2=0; i2<emf->n2fft; i2++){
      for(i1=0; i1<emf->n1fft; i1++){
	/* at z=-1.5*emf->d3 level */
	kxky[i1+emf->n1fft*i2] = H2kxkyz0[i1+emf->n1fft*i2]*exp(-emf->sqrtkx2ky2[i2][i1]*1.5*emf->d3uni);
      }
    }
    fftw_execute(ifft_airwave);/*Hz(x, y, z=0)-->Hz(kx, ky, z=0)*/
    for(i2=0; i2<emf->n2uni; i2++){
      for(i1=0; i1<emf->n1uni; i1++){
	emf->uni_H2[1][i2][i1] = creal(kxky[i1+emf->n1fft*i2]/(emf->n1fft*emf->n2fft));
      }
    }
  }
  if(emf->rd3>=3){//order=6
    for(i2=0; i2<emf->n2fft; i2++){
      for(i1=0; i1<emf->n1fft; i1++){
	/* at z=-1.5*emf->d3 level */
	kxky[i1+emf->n1fft*i2] = H2kxkyz0[i1+emf->n1fft*i2]*exp(-emf->sqrtkx2ky2[i2][i1]*2.5*emf->d3uni);
      }
    }
    fftw_execute(ifft_airwave);/*Hz(x, y, z=0)-->Hz(kx, ky, z=0)*/
    for(i2=0; i2<emf->n2uni; i2++){
      for(i1=0; i1<emf->n1uni; i1++){
	emf->uni_H2[2][i2][i1] = creal(kxky[i1+emf->n1fft*i2]/(emf->n1fft*emf->n2fft));
      }
    }
  }

  /*---------------------------------------------------------------------
   * interpolation from dense (uniform) grid to coarse (nonuniform) grid
   * use bilinear interpolation here, high order interpolation is instable
   * when interpolating from dense grid to coarse grid
   * in air-water boundary
   *---------------------------------------------------------------------*/
  for(i3=0; i3<emf->rd3; i3++){
    //----------------------------H1--------------------------------
    o1 = emf->x1n[0];//origin of the coordinate
    o2 = emf->x2n[0] + 0.5*emf->d2uni;//origin of the coordinate
    for(i2=0; i2<emf->n2pad; i2++){
      i2_ = (emf->x2s[i2]-o2)/emf->d2uni;//index on uniform grid
      ok2 = (i2_>=0 && i2_<emf->n2uni)?true:false;
      w2 = ok2?(emf->x2s[i2] - i2_*emf->d2uni-o2)/emf->d2uni:0;
      ok2p = (i2_+1>=0 && i2_+1<emf->n2uni)?true:false;
      for(i1=0; i1<emf->n1pad; i1++){
	i1_ = (emf->x1n[i1]-o1)/emf->d1uni;//index on uniform grid
	ok1 = (i1_>=0 && i1_<emf->n1uni)?true:false;
	w1 = ok1?(emf->x1n[i1] - i1_*emf->d1uni-o1)/emf->d1uni:0;
	ok1p = (i1_+1>=0 && i1_+1<emf->n1uni)?true:false;

	s = 0;
	if(ok1 && ok2) s += (1.-w1)*(1.-w2)*emf->uni_H1[i3][i2_][i1_];
	if(ok1p && ok2)	s += w1*(1.-w2)*emf->uni_H1[i3][i2_][i1_+1];
	if(ok1 && ok2p) s += (1.-w1)*w2*emf->uni_H1[i3][i2_+1][i1_];
	if(ok1p && ok2p)  s += w1*w2*emf->uni_H1[i3][i2_+1][i1_+1];
	emf->H1[emf->nbe-1-i3][i2][i1] = s;
      }
    }
    //----------------------------H2----------------------------------
    o1 = emf->x1n[0] + 0.5*emf->d1uni;//origin of the coordinate
    o2 = emf->x2n[0];//origin of the coordinate
    for(i2=0; i2<emf->n2pad; i2++){
      i2_ = (emf->x2n[i2]-o2)/emf->d2uni;//index on uniform grid
      ok2 = (i2_>=0 && i2_<emf->n2uni)?true:false;
      w2 = ok2?(emf->x2n[i2]-i2_*emf->d2uni-o2)/emf->d2uni:0;
      ok2p = (i2_+1>=0 && i2_+1<emf->n2uni)?true:false;
      for(i1=0; i1<emf->n1pad; i1++){
	i1_ = (emf->x1s[i1]-o1)/emf->d1uni;//index on uniform grid
	ok1 = (i1_>=0 && i1_<emf->n1uni)?true:false;
	w1 = ok1?(emf->x1s[i1]-i1_*emf->d1uni-o1)/emf->d1uni:0;
	ok1p = (i1_+1>=0 && i1_+1<emf->n1uni)?true:false;

	s = 0;
	if(ok1 && ok2) s += (1.-w1)*(1.-w2)*emf->uni_H2[i3][i2_][i1_];
	if(ok1p && ok2) s += w1*(1.-w2)*emf->uni_H2[i3][i2_][i1_+1];
	if(ok1 && ok2p) s += (1.-w1)*w2*emf->uni_H2[i3][i2_+1][i1_];
	if(ok1p && ok2p) s += w1*w2*emf->uni_H2[i3][i2_+1][i1_+1];
	emf->H2[emf->nbe-1-i3][i2][i1] = s;
      }
    }
  }
  
}


void airwave_bc_update_E(emf_t *emf)
{
  int i1, i2, i3, i1_, i2_;
  float w1, w2, o1, o2, s;
  bool ok1, ok2, ok1p, ok2p;
  
  /*----------------------------------E1------------------------------------*/
  if(emf->rd3>=2){//order=4
    memset(kxky, 0, emf->n1fft*emf->n2fft*sizeof(fftw_complex));
    i3 = emf->nbe;
    o1 = emf->x1n[0] + 0.5*emf->d1uni;
    o2 = emf->x2n[0];
    for(i2=0; i2<emf->n2uni; i2++){
      i2_ = emf->nu_i2[i2];
      ok2 = (i2_>=0 && i2_+1<emf->n2pad)?true:false;
      w2 = ok2?(o2 + i2*emf->d2uni -emf->x2n[i2_])/(emf->x2n[i2_+1]-emf->x2n[i2_]):0;
      ok2 = (i2_>=0 && i2_<emf->n2pad)?true:false;
      ok2p = (i2_+1>=0 && i2_+1<emf->n2pad)?true:false;
      for(i1=0; i1<emf->n1uni; i1++){
	i1_ = emf->nu_i1_s[i1];
	ok1 = (i1_>=0 && i1_+1<emf->n1pad)?true:false;
	w1 = ok1?(o1 + i1*emf->d1uni -emf->x1s[i1_])/(emf->x1s[i1_+1]-emf->x1s[i1_]):0;
	ok1 = (i1_>=0 && i1_<emf->n1pad)?true:false;
	ok1p = (i1_+1>=0 && i1_+1<emf->n1pad)?true:false;
	
	s = 0;
	if(ok1 && ok2) s += emf->E1[i3][i2_][i1_]*(1.-w1)*(1.-w2);
	if(ok1p && ok2) s += emf->E1[i3][i2_][i1_+1]*w1*(1.-w2);
	if(ok1 && ok2p) s += emf->E1[i3][i2_+1][i1_]*(1.-w1)*w2;
	if(ok1p && ok2p) s += emf->E1[i3][i2_+1][i1_+1]*w1*w2;
	kxky[i1+emf->n1fft*i2] = s;
      }
    }
    fftw_execute(fft_airwave);/* Ex(x, y, z=0)-->Hx(kx, ky, z=0) */
    memcpy(kxkyz0, kxky, emf->n1fft*emf->n2fft*sizeof(fftw_complex));
    /* for(i2=0; i2<emf->n2fft; i2++){ */
    /*   for(i1=0; i1<emf->n1fft; i1++){ */
    /* 	kxkyz0[i1+emf->n1fft*i2] *= emf->hw1[i1]*emf->hw2[i2]; */
    /*   } */
    /* } */
    if(emf->rd3>=2){
      for(i2=0; i2<emf->n2fft; i2++){
	for(i1=0; i1<emf->n1fft; i1++){
	  kxky[i1+emf->n1fft*i2] = kxkyz0[i1+emf->n1fft*i2]* exp(-emf->sqrtkx2ky2[i2][i1]*emf->d3uni);
	}
      }
      fftw_execute(ifft_airwave);
      for(i2=0; i2<emf->n2uni; i2++){
	for(i1=0; i1<emf->n1uni; i1++){
	  emf->uni_E1[0][i2][i1] = creal(kxky[i1+emf->n1fft*i2]/(emf->n1fft*emf->n2fft));
	}
      }
    }
  }
  if(emf->rd3>=3){// order=6
    for(i2=0; i2<emf->n2fft; i2++){
      for(i1=0; i1<emf->n1fft; i1++){
	kxky[i1+emf->n1fft*i2] = kxkyz0[i1+emf->n1fft*i2]* exp(-emf->sqrtkx2ky2[i2][i1]*2*emf->d3uni);
      }
    }
    fftw_execute(ifft_airwave);
    for(i2=0; i2<emf->n2uni; i2++){
      for(i1=0; i1<emf->n1uni; i1++){
	emf->uni_E1[1][i2][i1] = creal(kxky[i1+emf->n1fft*i2]/(emf->n1fft*emf->n2fft));
      }
    }
  }
  
  /*-----------------------------E2---------------------------------------*/
  if(emf->rd3>=2){//order 4
    memset(kxky, 0, emf->n1fft*emf->n2fft*sizeof(fftw_complex));
    i3 = emf->nbe;
    o1 = emf->x1n[0];
    o2 = emf->x2n[0] + 0.5*emf->d2uni;
    for(i2=0; i2<emf->n2uni; i2++){
      i2_ = emf->nu_i2_s[i2];
      ok2 = (i2_>=0 && i2_+1<emf->n2pad)?true:false;
      w2 = ok2?(o2 + i2*emf->d2uni -emf->x2s[i2_])/(emf->x2s[i2_+1]-emf->x2s[i2_]):0;
      ok2 = (i2_>=0 && i2_<emf->n2pad)?true:false;
      ok2p = (i2_+1>=0 && i2_+1<emf->n2pad)?true:false;
      for(i1=0; i1<emf->n1uni; i1++){
	i1_ = emf->nu_i1[i1];
	ok1 = (i1_>=0 && i1_+1<emf->n1pad)?true:false;
	w1 = ok1?(o1 + i1*emf->d1uni -emf->x1n[i1_])/(emf->x1n[i1_+1]-emf->x1n[i1_]):0;
	ok1 = (i1_>=0 && i1_<emf->n1pad)?true:false;
	ok1p = (i1_+1>=0 && i1_+1<emf->n1pad)?true:false;

	s = 0;
	if(ok1 && ok2) s += emf->E2[i3][i2_][i1_]*(1.-w1)*(1.-w2);
	if(ok1p && ok2) s += emf->E2[i3][i2_][i1_+1]*w1*(1.-w2);
	if(ok1 && ok2p) s += emf->E2[i3][i2_+1][i1_]*(1.-w1)*w2;
	if(ok1p && ok2p) s += emf->E2[i3][i2_+1][i1_+1]*w1*w2;
	kxky[i1+emf->n1fft*i2] = s;
      }
    }
    fftw_execute(fft_airwave);/* Ex(x, y, z=0)-->Hx(kx, ky, z=0) */
    memcpy(kxkyz0, kxky, emf->n1fft*emf->n2fft*sizeof(fftw_complex));
    /* for(i2=0; i2<emf->n2fft; i2++){ */
    /*   for(i1=0; i1<emf->n1fft; i1++){ */
    /* 	kxkyz0[i1+emf->n1fft*i2] *= emf->hw1[i1]*emf->hw2[i2]; */
    /*   } */
    /* } */
    for(i2=0; i2<emf->n2fft; i2++){
      for(i1=0; i1<emf->n1fft; i1++){
	kxky[i1+emf->n1fft*i2] = kxkyz0[i1+emf->n1fft*i2]* exp(-emf->sqrtkx2ky2[i2][i1]*emf->d3uni);
      }
    }
    fftw_execute(ifft_airwave);
    for(i2=0; i2<emf->n2uni; i2++){
      for(i1=0; i1<emf->n1uni; i1++){
	emf->uni_E2[0][i2][i1] = creal(kxky[i1+emf->n1fft*i2]/(emf->n1fft*emf->n2fft));
      }
    }
  }
  if(emf->rd3>=3){//order=6
    for(i2=0; i2<emf->n2fft; i2++){
      for(i1=0; i1<emf->n1fft; i1++){
	kxky[i1+emf->n1fft*i2] = kxkyz0[i1+emf->n1fft*i2]* exp(-emf->sqrtkx2ky2[i2][i1]*2*emf->d3uni);
      }
    }
    fftw_execute(ifft_airwave);
    for(i2=0; i2<emf->n2uni; i2++){
      for(i1=0; i1<emf->n1uni; i1++){
	emf->uni_E2[1][i2][i1] = creal(kxky[i1+emf->n1fft*i2]/(emf->n1fft*emf->n2fft));
      }
    }
  }

  /*-----------------------------------------------------------------------*/
  /* now we interpolate from uniform grid to nonuniform grid */
  /*-----------------------------------------------------------------------*/
  for(i3=0; i3<emf->rd3-1; i3++){
    //-----------------------E1-------------------
    o2 = emf->x2n[0];//origin of the coordinate
    o1 = emf->x1n[0] + 0.5*emf->d1uni;//origin of the coordinate
    for(i2=0; i2<emf->n2pad; i2++){
      i2_ = (emf->x2n[i2]-o2)/emf->d2uni;//index on uniform grid
      ok2 = (i2_>=0 && i2_<emf->n2uni)?true:false;
      w2 = ok2?(emf->x2n[i2]-o2-i2_*emf->d2uni)/emf->d2uni:0;
      ok2p = (i2_+1>=0 && i2_+1<emf->n2uni)?true:false;
      for(i1=0; i1<emf->n1pad; i1++){
	i1_ = (emf->x1s[i1]-o1)/emf->d1uni;//index on uniform grid
	ok1 = (i1_>=0 && i1_<emf->n1uni)?true:false;
	w1 = ok1?(emf->x1s[i1]-o1-i1_*emf->d1uni)/emf->d1uni:0;
	ok1p = (i1_+1>=0 && i1_+1<emf->n1uni)?true:false;

	s = 0;
	if(ok1 && ok2) s += (1.-w1)*(1-w2)*emf->uni_E1[i3][i2_][i1_];
	if(ok1p && ok2) s += w1*(1-w2)*emf->uni_E1[i3][i2_][i1_+1];
	if(ok1 && ok2p)	s += (1.-w1)*w2*emf->uni_E1[i3][i2_+1][i1_];
	if(ok1p && ok2p) s += w1*w2*emf->uni_E1[i3][i2_+1][i1_+1];
	emf->E1[emf->nbe-1-i3][i2][i1] = s;
      }
    }
    //--------------------E2----------------------------
    o2 = emf->x2n[0] + 0.5*emf->d2uni;//origin of the coordinate
    o1 = emf->x1n[0];//origin of the coordinate
    for(i2=0; i2<emf->n2pad; i2++){
      i2_ = (emf->x2s[i2]-o2)/emf->d2uni;//index on uniform grid
      ok2 = (i2_>=0 && i2_<emf->n2uni)?true:false;
      w2 = ok2?(emf->x2s[i2]-o2-i2_*emf->d2uni)/emf->d2uni:0;
      ok2p = (i2_+1>=0 && i2_+1<emf->n2uni)?true:false;
      for(i1=0; i1<emf->n1pad; i1++){
	i1_ = (emf->x1n[i1]-o1)/emf->d1uni;//index on uniform grid
	ok1 = (i1_>=0 && i1_<emf->n1uni)?true:false;
	w1 = ok1?(emf->x1n[i1]-o1-i1_*emf->d1uni)/emf->d1uni:0;
	ok1 = (i1_+1>=0 && i1_+1<emf->n1uni)?true:false;

	s = 0;
	if(ok1 && ok2) s += (1.-w1)*(1-w2)*emf->uni_E2[i3][i2_][i1_];
	if(ok1p && ok2) s += w1*(1-w2)*emf->uni_E2[i3][i2_][i1_+1];
	if(ok1 && ok2p)	s += (1.-w1)*w2*emf->uni_E2[i3][i2_+1][i1_];
	if(ok1p && ok2p) s += w1*w2*emf->uni_E2[i3][i2_+1][i1_+1];
	emf->E2[emf->nbe-1-i3][i2][i1] = s;
      }
    }

  }

}


/*----------------------------------------------------------------------------*/
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

  interp_sg->rec_i1 = alloc2int(acqui->nsubrec, acqui->nrec);
  interp_sg->rec_i2 = alloc2int(acqui->nsubrec, acqui->nrec);
  interp_sg->rec_i3 = alloc2int(acqui->nsubrec, acqui->nrec);
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
 * Given the vector f=(f(x0), f(x1), ..., f(xn))^T and Vandermonde matrix 
 * V(x0, x1, ..., xn), the solution of Vandermonde matrix inversion V^T a =f
 * gives:
 *  (a0)  (f (x)    )  (w0 w1 ... wn) ( f(x0) )
 *  (a1)= (f'(x)    ) =(            ) ( f(x1) )
 *  (.)     ...        (            ) ( ...   )
 *  (an)  (f^n(x)/n!)  (            ) ( f(xn) )
 *              --------------
 *                 V^{-1}
 * we use the first row of inverse Vandermonde matrix as the interpolation weights:
 *  a0 = f(x) = \sum_{i=0}^{i=n} wi * f(xi)
 * From the above expression, we know all weights wi (i=0, ..., n) can be obtained by
 * setting f(xi)=1 and f(xj)=0 (for all j\neq i).
 *-------------------------------------------------------------------------------*/
void interpolation_weights(acqui_t *acqui, emf_t *emf, interp_t *interp_rg, interp_t *interp_sg)
{
  const float eps=1e-3*MIN(emf->dx1min, emf->dx2min);
  
  float x1j, x2j, x3j; //, o1, o2, o3;
  int j1, j2, j3;
  int i, j, isub, loop, m, isrc, irec;
  bool skip1, skip2, skip3;
  float *x1, *x2, *x3, *aa, *ff;
  float *x1m, *x2m, *x3m;
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
	  x1m = emf->x1n;
	  x2m = emf->x2n;
	  x3m = emf->x3n;
	}else{/* staggered grid */
	  interp = interp_sg;
	  x1m = emf->x1s;
	  x2m = emf->x2s;
	  x3m = emf->x3s;
	}
	j1 = find_index(emf->n1pad, x1m, x1j);
	j2 = find_index(emf->n2pad, x2m, x2j);
	j3 = find_index(emf->n3pad, x3m, x3j);
	interp->rec_i1[irec][isub] = j1;
	interp->rec_i2[irec][isub] = j2;
	interp->rec_i3[irec][isub] = j3;	

	skip1 = false;
	skip2 = false;
	skip3 = false;
	if(fabs(x1m[j1]-x1j)<eps){
	  memset(interp->rec_w1[irec][isub], 0, 2*emf->rd*sizeof(float));
	  memset(interp->rec_v1[irec][isub], 0, 2*emf->rd*sizeof(float));
	  interp->rec_w1[irec][isub][emf->rd1-1] = 1.;
	  tmp = 1./(x1m[j1+1]-x1m[j1]);
	  interp->rec_v1[irec][isub][emf->rd1-1] = -tmp;
	  interp->rec_v1[irec][isub][emf->rd1] = tmp;
	  skip1 = true;
	}
	if(fabs(x2m[j2]-x2j)<eps){
	  memset(interp->rec_w2[irec][isub], 0, 2*emf->rd*sizeof(float));
	  memset(interp->rec_v2[irec][isub], 0, 2*emf->rd*sizeof(float));
	  interp->rec_w2[irec][isub][emf->rd2-1] = 1.;
	  tmp = 1./(x2m[j2+1]-x2m[j2]);
	  interp->rec_v2[irec][isub][emf->rd2-1] = -tmp;
	  interp->rec_v2[irec][isub][emf->rd2] = tmp;
	  skip2 = true;
	}
	if(fabs(x3m[j3]-x3j)<eps){
	  memset(interp->rec_w3[irec][isub], 0, 2*emf->rd*sizeof(float));
	  memset(interp->rec_v3[irec][isub], 0, 2*emf->rd*sizeof(float));
	  interp->rec_w3[irec][isub][emf->rd3-1] = 1.;
	  tmp = 1./(x3m[j3+1]-x3m[j3]);
	  interp->rec_v3[irec][isub][emf->rd3-1] = -tmp;
	  interp->rec_v3[irec][isub][emf->rd3] = tmp;
	  skip3 = true;
	}
	
	for(i=0; i<2*emf->rd; i++){/* construct vector x1[], x2[], x3[] */
	  m = i-emf->rd1+1;/* m=offset/shift between [-emf->rd+1, emf->rd] */
	  if(!skip1) x1[i] = x1m[j1+m]-x1j;
	  m = i-emf->rd2+1;/* m=offset/shift between [-emf->rd+1, emf->rd] */
	  if(!skip2) x2[i] = x2m[j2+m]-x2j;
	  m = i-emf->rd3+1;/* m=offset/shift between [-emf->rd+1, emf->rd] */
	  if(!skip3) x3[i] = x3m[j3+m]-x3j;
	}	
	for(i=0; i<2*emf->rd; i++){
	  memset(ff, 0, 2*emf->rd*sizeof(float));
	  ff[i] =  1.;
	  if(!skip1){
	    vandermonde(2*emf->rd1-1, x1, aa, ff);
	    interp->rec_w1[irec][isub][i] = aa[0];
	    interp->rec_v1[irec][isub][i] = aa[1];
	  }
	  if(!skip2){
	    vandermonde(2*emf->rd2-1, x2, aa, ff);
	    interp->rec_w2[irec][isub][i] = aa[0];
	    interp->rec_v2[irec][isub][i] = aa[1];
	  }
	  if(!skip3){
	    vandermonde(2*emf->rd3-1, x3, aa, ff);
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
	  x1m = emf->x1n;
	  x2m = emf->x2n;
	  x3m = emf->x3n;
	}else{/* staggered grid */
	  interp = interp_sg;
	  x1m = emf->x1s;
	  x2m = emf->x2s;
	  x3m = emf->x3s;
	}
	j1 = find_index(emf->n1pad, x1m, x1j);
	j2 = find_index(emf->n2pad, x2m, x2j);
	j3 = find_index(emf->n3pad, x3m, x3j);
	interp->src_i1[isrc][isub] = j1;
	interp->src_i2[isrc][isub] = j2;
	interp->src_i3[isrc][isub] = j3;

	skip1 = false;
	skip2 = false;
	skip3 = false;
	if(fabs(x1m[j1]-x1j)<eps){
	  memset(interp->src_w1[isrc][isub], 0, 2*emf->rd*sizeof(float));
	  interp->src_w1[isrc][isub][emf->rd1-1] = 1.;
	  skip1 = true;
	}
	if(fabs(x2m[j2]-x2j)<eps){
	  memset(interp->src_w2[isrc][isub], 0, 2*emf->rd*sizeof(float));
	  interp->src_w2[isrc][isub][emf->rd2-1] = 1.;
	  skip2 = true;
	}
	if(fabs(x3m[j3]-x3j)<eps){
	  memset(interp->src_w3[isrc][isub], 0, 2*emf->rd*sizeof(float));
	  interp->src_w3[isrc][isub][emf->rd3-1] = 1.;
	  skip3 = true;
	}

	for(i=0; i<2*emf->rd; i++){/* construct vector x1[], x2[], x3[] */
	  m = i-emf->rd1+1;/* m=offset/shift between [-emf->rd+1, emf->rd] */
	  if(!skip1) x1[i] = x1m[j1+m]-x1j;
	  m = i-emf->rd2+1;/* m=offset/shift between [-emf->rd+1, emf->rd] */
	  if(!skip2) x2[i] = x2m[j2+m]-x2j;
	  m = i-emf->rd3+1;/* m=offset/shift between [-emf->rd+1, emf->rd] */
	  if(!skip3) x3[i] = x3m[j3+m]-x3j;
	}
	for(i=0; i<2*emf->rd; i++){
	  memset(ff, 0, 2*emf->rd*sizeof(float));
	  ff[i] =  1.;
	  if(!skip1){
	    vandermonde(2*emf->rd1-1, x1, aa, ff);
	    interp->src_w1[isrc][isub][i] = aa[0];
	  }
	  if(!skip2){
	    vandermonde(2*emf->rd2-1, x2, aa, ff);
	    interp->src_w2[isrc][isub][i] = aa[0];
	  }
	  if(!skip3){
	    vandermonde(2*emf->rd3-1, x3, aa, ff);
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

void inject_electric_src_fwd(acqui_t *acqui, emf_t *emf, interp_t *interp_rg, interp_t *interp_sg, int it)
/*< inject a source time function into EM field >*/
{
  int ic, isrc, isub, i1, i2, i3, ix1, ix2, ix3, i1_, i2_, i3_;
  float w1, w2, w3, ss, s;
  float d1, d2, d3;
  
  ss = emf->stf[it];
  ss /= (float)acqui->nsubsrc; /*since one source is distributed over many points */
  for(isrc=0; isrc<acqui->nsrc; isrc++){
    for(isub=0; isub<acqui->nsubsrc; isub++){

      for(ic=0; ic<emf->nch_src; ++ic){
	if(strcmp(emf->ch_src[ic], "Ex") == 0){
	  /* staggered grid: E1[i1, i2, i3] = Ex[i1+0.5, i2, i3] */
	  ix1 = interp_sg->src_i1[isrc][isub];
	  ix2 = interp_rg->src_i2[isrc][isub];
	  ix3 = interp_rg->src_i3[isrc][isub];
	  d1 = emf->x1s[ix1+1]-emf->x1s[ix1];
	  d2 = emf->x2n[ix2+1]-emf->x2n[ix2];
	  d3 = emf->x3n[ix3+1]-emf->x3n[ix3];
	  s = ss/(d1*d2*d3);/* source normalized by volume */
	  for(i3=-emf->rd3+1; i3<=emf->rd3; i3++){
	    w3 = interp_rg->src_w3[isrc][isub][i3+emf->rd3-1];
	    i3_ = ix3+i3;
	    for(i2=-emf->rd2+1; i2<=emf->rd2; i2++){
	      w2 = interp_rg->src_w2[isrc][isub][i2+emf->rd2-1];
	      i2_ = ix2+i2;
	      for(i1=-emf->rd1+1; i1<=emf->rd1; i1++){
		w1 = interp_sg->src_w1[isrc][isub][i1+emf->rd1-1];
		i1_ = ix1+i1;
	  
		emf->curlH1[i3_][i2_][i1_] -= s*w1*w2*w3;
	      }/* end for i1 */
	    }/* end for i2 */
	  }/* end for i3 */
	}else if(strcmp(emf->ch_src[ic], "Ey") == 0){
	  /* staggered grid: E2[i1, i2, i3] = Ey[i1, i2+0.5, i3] */
	  ix1 = interp_rg->src_i1[isrc][isub];
	  ix2 = interp_sg->src_i2[isrc][isub];
	  ix3 = interp_rg->src_i3[isrc][isub];
	  d1 = emf->x1n[ix1+1]-emf->x1n[ix1];
	  d2 = emf->x2s[ix2+1]-emf->x2s[ix2];
	  d3 = emf->x3n[ix3+1]-emf->x3n[ix3];
	  s = ss/(d1*d2*d3);/* source normalized by volume */
	  for(i3=-emf->rd3+1; i3<=emf->rd3; i3++){
	    w3 = interp_rg->src_w3[isrc][isub][i3+emf->rd3-1];
	    i3_ = ix3+i3;
	    for(i2=-emf->rd2+1; i2<=emf->rd2; i2++){
	      w2 = interp_sg->src_w2[isrc][isub][i2+emf->rd2-1];
	      i2_ = ix2+i2;
	      for(i1=-emf->rd1+1; i1<=emf->rd1; i1++){
		w1 = interp_rg->src_w1[isrc][isub][i1+emf->rd1-1];
		i1_ = ix1+i1;

		emf->curlH2[i3_][i2_][i1_] -= s*w1*w2*w3;
	      }
	    }
	  }
	}else if(strcmp(emf->ch_src[ic], "Ez") == 0){
	  /* staggered grid: E3[i1, i2, i3] = Ez[i1, i2, i3+0.5] */
	  ix1 = interp_rg->src_i1[isrc][isub];
	  ix2 = interp_rg->src_i2[isrc][isub];
	  ix3 = interp_sg->src_i3[isrc][isub];
	  d1 = emf->x1n[ix1+1]-emf->x1n[ix1];
	  d2 = emf->x2n[ix2+1]-emf->x2n[ix2];
	  d3 = emf->x3s[ix3+1]-emf->x3s[ix3];
	  s = ss/(d1*d2*d3);/* source normalized by volume */
	  for(i3=-emf->rd3+1; i3<=emf->rd3; i3++){
	    w3 = interp_sg->src_w3[isrc][isub][i3+emf->rd3-1];
	    i3_ = ix3+i3;
	    for(i2=-emf->rd2+1; i2<=emf->rd2; i2++){
	      w2 = interp_rg->src_w2[isrc][isub][i2+emf->rd2-1];
	      i2_ = ix2+i2;
	      for(i1=-emf->rd1+1; i1<=emf->rd1; i1++){
		w1 = interp_rg->src_w1[isrc][isub][i1+emf->rd1-1];
		i1_ = ix1+i1;

		emf->curlH3[i3_][i2_][i1_] -= s*w1*w2*w3;
	      }
	    }
	  }
	}
      }
    }/* end for isub */
  }/* end for isrc */
    
}

void compute_green_function(emf_t *emf)
{
  int ifreq, it, i1, i2, i3;
  float _Complex s, t, omegap, src_fd;

  int i1min=emf->nb;
  int i2min=emf->nb;
  int i3min=emf->nb;
  int i1max=emf->n1pad-1-emf->nb;
  int i2max=emf->n2pad-1-emf->nb;
  int i3max=emf->n3pad-1-emf->nb;

  for(ifreq=0; ifreq < emf->nfreq; ++ifreq){
    /*------------------- DFT over omega'-------------------------*/
    omegap = (1.0+I)*sqrtf(emf->omega0*emf->omegas[ifreq]);/* omega' in fictitous wave domain */
    src_fd=0.;
    for(it=0; it<emf->nt; ++it)  src_fd += emf->stf[it]*cexp(I*omegap*it*emf->dt);
    s = csqrt(-I*0.5*emf->omegas[ifreq]/emf->omega0);

    for(i3=i3min; i3<=i3max; ++i3){
      for(i2=i2min; i2<=i2max; ++i2){
	for(i1=i1min; i1<=i1max; ++i1){
	  emf->fwd_E1[ifreq][i3][i2][i1] /= src_fd;
	  emf->fwd_E2[ifreq][i3][i2][i1] /= src_fd;
	  emf->fwd_E3[ifreq][i3][i2][i1] /= src_fd;
	  emf->fwd_E1[ifreq][i3][i2][i1] *= s;
	  emf->fwd_E2[ifreq][i3][i2][i1] *= s;
	  emf->fwd_E3[ifreq][i3][i2][i1] *= s;

	  t = 2.*emf->omega0/emf->inveps33[i3][i2][i1];//sigma=2*omega0*epsil
	  emf->fwd_Jz[ifreq][i3][i2][i1] = t*emf->fwd_E3[ifreq][i3][i2][i1];
	}
      }
    }

  }
}

void extract_emf(acqui_t *acqui, emf_t *emf, interp_t *interp_rg, interp_t *interp_sg)
/*< extract data from EM field by interpolation >*/
{   
  int ic, irec, isub, ifreq, i1, i2, i3, ix1, ix2, ix3, i1_, i2_, i3_;
  float w1, w2, w3;
  float _Complex s;// t;

  for(ifreq=0; ifreq < emf->nfreq; ++ifreq){

    for(irec = 0; irec<acqui->nrec; irec++) {
      for(isub=0; isub<acqui->nsubrec; isub++){
	
	/* We use eqn 11 in Mittet (2010) */
	for(ic=0; ic<emf->nchrec; ++ic){
	  if(strcmp(emf->ch_rec[ic], "Ex") == 0){
	    /* staggered grid: E1[i1, i2, i3] = Ex[i1+0.5, i2, i3] */
	    s = 0.;
	    ix1 = interp_sg->rec_i1[irec][isub];
	    ix2 = interp_rg->rec_i2[irec][isub];
	    ix3 = interp_rg->rec_i3[irec][isub];
	    for(i3=-emf->rd3+1; i3<=emf->rd3; i3++){
	      w3 = interp_rg->rec_w3[irec][isub][i3+emf->rd3-1];
	      i3_ = ix3+i3;
	      for(i2=-emf->rd2+1; i2<=emf->rd2; i2++){
		w2 = interp_rg->rec_w2[irec][isub][i2+emf->rd2-1];
		i2_ = ix2+i2;
		for(i1=-emf->rd1+1; i1<=emf->rd1; i1++){
		  w1 = interp_sg->rec_w1[irec][isub][i1+emf->rd1-1];
		  i1_ = ix1+i1;

		  s += emf->fwd_E1[ifreq][i3_][i2_][i1_]*w1*w2*w3;
		}
	      }
	    }

	    /* /\* derivative jump correction *\/ */
	    /* /\* staggered grid: E3[i1, i2, i3] = Ez[i1, i2, i3+0.5] *\/ */
	    /* ix1 = interp_rg->rec_i1[irec][isub]; */
	    /* ix2 = interp_rg->rec_i2[irec][isub]; */
	    /* ix3 = interp_sg->rec_i3[irec][isub]; */
	    /* for(i3=-emf->rd3+1; i3<=emf->rd3; i3++){ */
	    /*   w3 = interp_rg->rec_w3[irec][isub][i3+emf->rd3-1]; */
	    /*   i3_ = ix3+i3; */
	    /*   distance = emf->x3n[i3_] -emf->x3n[emf->nbe]- emf->waterdepth; */
	    /*   if(distance>0) { */
	    /* 	for(i2=-emf->rd2+1; i2<=emf->rd2; i2++){ */
	    /* 	  w2 = interp_rg->rec_w2[irec][isub][i2+emf->rd2-1]; */
	    /* 	  i2_ = ix2+i2; */
	    /* 	  for(i1=-emf->rd1+1; i1<=emf->rd1; i1++){ */
	    /* 	    w1 = interp_rg->rec_v1[irec][isub][i1+emf->rd1-1]; */
	    /* 	    i1_ = ix1+i1; */

	    /* 	    t = emf->fwd_Jz[ifreq][i3_][i2_][i1_]*w1*w2*w3; */
	    /* 	    t *= (1./emf->sigma_formation-1./emf->sigma_water); */
	    /* 	    s -= distance*t; */
	    /* 	  } */
	    /* 	} */
	    /*   } */
	    /* } */

	    emf->dcal_fd[ic][ifreq][irec] = s;
	  }/* end if */

	}/* end for ic */

      }/* end for isub */
    }/* end for irec */

  }/* end for ifreq */

}

void dtft_emf_init(emf_t *emf, int adj)
{
  if(adj){
    emf->adj_E1 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
    emf->adj_E2 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
    emf->adj_E3 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
    emf->adj_H1 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
    emf->adj_H2 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
    emf->adj_H3 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
    emf->adj_Jz = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);

    memset(&emf->adj_E1[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
    memset(&emf->adj_E2[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
    memset(&emf->adj_E3[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
    memset(&emf->adj_H1[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
    memset(&emf->adj_H2[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
    memset(&emf->adj_H3[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
    memset(&emf->adj_Jz[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));

  }else{
    emf->fwd_E1 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
    emf->fwd_E2 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
    emf->fwd_E3 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
    emf->fwd_H1 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
    emf->fwd_H2 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
    emf->fwd_H3 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
    emf->fwd_Jz = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);

    memset(&emf->fwd_E1[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
    memset(&emf->fwd_E2[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
    memset(&emf->fwd_E3[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
    memset(&emf->fwd_H1[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
    memset(&emf->fwd_H2[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
    memset(&emf->fwd_H3[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
    memset(&emf->fwd_Jz[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
  }

}

void dtft_emf_close(emf_t *emf, int adj)
{
  if(adj){
    free4complexf(emf->adj_E1);
    free4complexf(emf->adj_E2);
    free4complexf(emf->adj_E3);
    free4complexf(emf->adj_H1);
    free4complexf(emf->adj_H2);
    free4complexf(emf->adj_H3);
    free4complexf(emf->adj_Jz);

  }else{
    free4complexf(emf->fwd_E1);
    free4complexf(emf->fwd_E2);
    free4complexf(emf->fwd_E3);
    free4complexf(emf->fwd_H1);
    free4complexf(emf->fwd_H2);
    free4complexf(emf->fwd_H3);
    free4complexf(emf->fwd_Jz);
  }

}


void dtft_emf(emf_t *emf, int it, int adj)
{
  int i1, i2, i3, ifreq;
  float _Complex omegap, factor;
  float ***emf1, ***emf2, ***emf3;
  float _Complex ***ptr1, ***ptr2, ***ptr3;
  
  int i1min=emf->nb;
  int i2min=emf->nb;
  int i3min=emf->nb;
  int i1max=emf->n1pad-1-emf->nb;
  int i2max=emf->n2pad-1-emf->nb;
  int i3max=emf->n3pad-1-emf->nb;

  for(ifreq=0; ifreq<emf->nfreq; ++ifreq){
    omegap = (1.0+I)*sqrt(emf->omega0*emf->omegas[ifreq]);/* omega' in fictitous wave domain */
    
    factor = cexp(I*omegap*(it+0.5)*emf->dt);
    ptr1 = adj?emf->adj_E1[ifreq]:emf->fwd_E1[ifreq];
    ptr2 = adj?emf->adj_E2[ifreq]:emf->fwd_E2[ifreq];
    ptr3 = adj?emf->adj_E3[ifreq]:emf->fwd_E3[ifreq];
    emf1 = emf->E1;
    emf2 = emf->E2;
    emf3 = emf->E3;
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3)							\
  shared(i1min, i1max, i2min, i2max, i3min, i3max, factor, emf1, emf2, emf3, ptr1, ptr2, ptr3)
#endif
    for(i3=i3min; i3<=i3max; ++i3){
      for(i2=i2min; i2<=i2max; ++i2){
	for(i1=i1min; i1<=i1max; ++i1){
	  ptr1[i3][i2][i1] += emf1[i3][i2][i1]*factor;
	  ptr2[i3][i2][i1] += emf2[i3][i2][i1]*factor;
	  ptr3[i3][i2][i1] += emf3[i3][i2][i1]*factor;
	}
      }
    }

  }/* end for ifreq */
}

int check_convergence(emf_t *emf)
{
  static float _Complex old[8], new[8];
  static int first = 1;
  int j, k;
  
  k = 0;
  if(first){
    old[0] = emf->fwd_E1[0][emf->nbe][emf->nbe][emf->nbe];
    old[1] = emf->fwd_E1[0][emf->nbe][emf->nbe][emf->nbe+emf->n1-1];
    old[2] = emf->fwd_E1[0][emf->nbe][emf->nbe+emf->n2-1][emf->nbe];
    old[3] = emf->fwd_E1[0][emf->nbe][emf->nbe+emf->n2-1][emf->nbe+emf->n1-1];
    old[4] = emf->fwd_E1[0][emf->nbe+emf->n3-1][emf->nbe][emf->nbe];
    old[5] = emf->fwd_E1[0][emf->nbe+emf->n3-1][emf->nbe][emf->nbe+emf->n1-1];
    old[6] = emf->fwd_E1[0][emf->nbe+emf->n3-1][emf->nbe+emf->n2-1][emf->nbe];
    old[7] = emf->fwd_E1[0][emf->nbe+emf->n3-1][emf->nbe+emf->n2-1][emf->nbe+emf->n1-1];
    first = 0;
  }else{
    new[0] = emf->fwd_E1[0][emf->nbe][emf->nbe][emf->nbe];
    new[1] = emf->fwd_E1[0][emf->nbe][emf->nbe][emf->nbe+emf->n1-1];
    new[2] = emf->fwd_E1[0][emf->nbe][emf->nbe+emf->n2-1][emf->nbe];
    new[3] = emf->fwd_E1[0][emf->nbe][emf->nbe+emf->n2-1][emf->nbe+emf->n1-1];
    new[4] = emf->fwd_E1[0][emf->nbe+emf->n3-1][emf->nbe][emf->nbe];
    new[5] = emf->fwd_E1[0][emf->nbe+emf->n3-1][emf->nbe][emf->nbe+emf->n1-1];
    new[6] = emf->fwd_E1[0][emf->nbe+emf->n3-1][emf->nbe+emf->n2-1][emf->nbe];
    new[7] = emf->fwd_E1[0][emf->nbe+emf->n3-1][emf->nbe+emf->n2-1][emf->nbe+emf->n1-1];

    for(j=0; j<8; j++){
      if(cabs(new[j])>0 && cabs(new[j]-old[j])<1e-4*cabs(new[j])) k++;
      old[j] = new[j];
    }
  }
  return k;
}


void write_data(acqui_t *acqui, emf_t *emf, char *fname)
/*< write synthetic data according to shot/process index >*/
{
  FILE *fp;
  int isrc, irec, ichsrc, ichrec, ifreq;
  float dp_re, dp_im;

  ichsrc = 0;
  fp=fopen(fname,"w");
  if(fp==NULL) err("error opening file for writing");
  fprintf(fp, "iTx 	 iRx     ichsrc  ichrec  ifreq 	 emf_real 	 emf_imag\n");
  isrc = acqui->shot_idx[iproc];//index starts from 1
  for(irec=0; irec<acqui->nrec; irec++){
    for(ichrec=0; ichrec<emf->nchrec; ichrec++){
      for(ifreq=0; ifreq<emf->nfreq; ifreq++){
	dp_re = creal(emf->dcal_fd[ichrec][ifreq][irec]);
	dp_im = cimag(emf->dcal_fd[ichrec][ifreq][irec]);
	fprintf(fp, "%d \t %d \t %d \t %d \t %d \t %e \t %e\n",
		isrc, irec+1, ichsrc+1, ichrec+1, ifreq+1, dp_re, dp_im);
      }
    }
  }
  fclose(fp);
}


int main(int argc, char* argv[])
{
  int it, adj, k;
  float beta, td, tmp;
  acqui_t *acqui;
  emf_t *emf;
  interp_t *interp_rg, *interp_sg;
  char fname[sizeof("emf_0000.txt")];
  FILE *fp;
  int i1, i2, i3, i1_, i2_, i3_;
  
  MPI_Init(&argc, &argv);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  initargs(argc, argv);

  acqui = (acqui_t *)malloc(sizeof(acqui_t));
  emf = (emf_t *)malloc(sizeof(emf_t));
  interp_rg = (interp_t *)malloc(sizeof(interp_t));
  interp_sg = (interp_t *)malloc(sizeof(interp_t));

  emf_init(emf);
  acqui_init(acqui, emf);

  adj = 0;
  extend_model_init(emf);
  nugrid_init(emf);
  fdtd_init(emf);
  dtft_emf_init(emf, adj);
  airwave_bc_init(emf);

  interpolation_init(acqui, emf, interp_rg, interp_sg);
  interpolation_weights(acqui, emf, interp_rg, interp_sg);
  sanity_check(emf);//determine dt and nt

  emf->stf = alloc1float(emf->nt);
  beta = PI*emf->freqmax*emf->freqmax;
  td = PI/emf->freqmax;
  for(it=0; it<emf->nt; ++it){
    tmp = it*emf->dt - td;
    emf->stf[it] = -2.0*beta*tmp*emf->freqmax*exp(-beta*tmp*tmp);
  }
  emf->dcal_fd = alloc3complexf(acqui->nrec, emf->nfreq, emf->nchrec);
  memset(&emf->dcal_fd[0][0][0], 0, acqui->nrec*emf->nchrec*emf->nfreq*sizeof(float _Complex));

  for(it=0; it<emf->nt; it++){
    if(it%50==0) printf("it-----%d\n", it);

    nufdtd_curlH(emf);
    inject_electric_src_fwd(acqui, emf, interp_rg, interp_sg, it);
    nufdtd_update_E(emf); 
    if(emf->airwave) airwave_bc_update_E(emf);      
    
    nufdtd_curlE(emf); 
    nufdtd_update_H(emf); 
    if(emf->airwave) airwave_bc_update_H(emf);    
    
    dtft_emf(emf, it, adj);
    
    if(it%100==0){/* convergence check */
      k = check_convergence(emf);
      if(iproc==0) printf("%d corners of the cube converged!\n", k);
      if(k==8) break;/* all 8 corners converged, exit now */
    }
  }
  compute_green_function(emf);
  extract_emf(acqui, emf, interp_rg, interp_sg);
  sprintf(fname, "emf_%04d.txt", acqui->shot_idx[iproc]);
  write_data(acqui, emf, fname);

  fp = fopen("Gf_Ex", "wb");
  i2 = emf->n2/2 + emf->nbe;
  i2_ = i2+ emf->nbe;
  for(i1=0; i1<emf->n1; ++i1){
    i1_ = i1+ emf->nbe;
    for(i3=0; i3<emf->n3; ++i3){
      i3_ = i3+ emf->nbe;

      tmp = cabs(emf->fwd_E1[0][i3_][i2_][i1_]);
      tmp = log(tmp)/log(10.);
      fwrite(&tmp, 1, sizeof(float), fp);
    }
  }
  fclose(fp);

  fp = fopen("Gf_Ez", "wb");
  i2 = emf->n2/2 + emf->nbe;
  i2_ = i2+ emf->nbe;
  for(i1=0; i1<emf->n1; ++i1){
    i1_ = i1+ emf->nbe;
    for(i3=0; i3<emf->n3; ++i3){
      i3_ = i3+ emf->nbe;

      tmp = cabs(emf->fwd_E3[0][i3_][i2_][i1_]);
      tmp = log(tmp)/log(10.);
      fwrite(&tmp, 1, sizeof(float), fp);
    }
  }
  fclose(fp);


  extend_model_close(emf);
  nugrid_close(emf);
  fdtd_close(emf);
  dtft_emf_close(emf, adj);
  airwave_bc_close(emf);


  emf_close(emf);
  acqui_close(acqui);
  interpolation_close(emf, interp_rg, interp_sg);

  free(emf->stf);
  free3complexf(emf->dcal_fd);
  free(acqui);
  free(emf);
  free(interp_rg);
  free(interp_sg);

  
  MPI_Finalize();
  return 0;
}
