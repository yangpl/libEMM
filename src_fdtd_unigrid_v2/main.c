/* EM modeling using FDTD method
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
#define mu0 (4.*PI*1e-7)


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
} acqui_t;/* type of acquisition geometry */


typedef struct {
  int mode; //mode=0, modeling; mode=1, FWI; mode=2, FWI gradient only
  int reciprocity;
  
  float omega0;
  int n1, n2, n3, nb, n123;
  int n1pad, n2pad, n3pad, n123pad;
  float d1, d2, d3, dt;
  float o1, o2, o3;//origin of the model
  
  int nchsrc, nchrec;//number of active src/rec channels for E and H
  char **chsrc, **chrec;

  int nfreq;//number of frequencies
  float *freqs, *omegas;//a list of frequencies
  float f0;//dominant frequency for the wavelet
  float *stf;//source time function

  float ***rho11, ***rho22, ***rho33;//normal and transverse resistivities
  float rhomin, rhomax; //mimum and maximum resistivity
  float vmin, vmax;//minimum and maximum velocities of the EM wave

  int rd1, rd2, rd3, rd;/* half length/radius of the interpolation operator */
  int nt; //number of time steps in total

  int n1fft, n2fft;
  float ***sE12kxky;
  float _Complex ***sH1kxky, ***sH2kxky;

  float ***inveps11, ***inveps22, ***inveps33;//fictitous domain dielectric permittivity
  float *apml, *bpml;
  float ***E1, ***E2, ***E3, ***H1, ***H2, ***H3;
  float ***curlE1, ***curlE2, ***curlE3, ***curlH1, ***curlH2, ***curlH3;
  float ***memD2H3, ***memD3H2, ***memD3H1, ***memD1H3, ***memD1H2, ***memD2H1;
  float ***memD2E3, ***memD3E2, ***memD3E1, ***memD1E3, ***memD1E2, ***memD2E1;

  float _Complex ***dcal_fd, ***dobs_fd, ***dres_fd;
  float ***dres_td;

  float _Complex ****fwd_E1, ****fwd_E2, ****fwd_E3;
  float _Complex ****fwd_H1, ****fwd_H2, ****fwd_H3;
  int ncorner;
} emf_t;


typedef struct {
  /*---------------------sources----------------------------*/
  int **src_i1, **src_i2, **src_i3;/* index on FD grid */
  float ***src_w1, ***src_w2, ***src_w3; /* weights on FD grid for f(x) */
  /*---------------------receivers--------------------------*/
  int **rec_i1, **rec_i2, **rec_i3;/* index on FD grid */
  float ***rec_w1, ***rec_w2, ***rec_w3; /* weights on FD grid for f(x) */
} interp_t;/* type of interpolation on regular and staggerred grid */

int cmpfunc(const void *a, const void *b) { return ( *(int*)a - *(int*)b ); }

void emf_init(emf_t *emf)
{
  char *frho11, *frho22, *frho33;
  FILE *fp=NULL;
  int ifreq, ic, istat;

  if(!getparint("reciprocity", &emf->reciprocity)) emf->reciprocity=0; 
  if(!getparint("n1", &emf->n1)) emf->n1=101; 
  /* number of cells in axis-1, nx */
  if(!getparint("n2", &emf->n2)) emf->n2=101; 
  /* number of cells in axis-2, ny */
  if(!getparint("n3", &emf->n3)) emf->n3=51; 
  /* number of cells in axis-3, nz */
  if(!getparint("nb", &emf->nb)) emf->nb=12; 
  /* number of PML layers on each side */
  if(!getparfloat("d1", &emf->d1)) emf->d1=100.;
  /* grid spacing in 1st dimension, dx */
  if(!getparfloat("d2", &emf->d2)) emf->d2=100.;
  /* grid spacing in 2nd dimension, dy */
  if(!getparfloat("d3", &emf->d3)) emf->d3=100.;
  /* grid spacing in 3rd dimension, dz */
  if(!getparfloat("o1", &emf->o1)) emf->o1=0.;
  if(!getparfloat("o2", &emf->o2)) emf->o2=0.;
  if(!getparfloat("o3", &emf->o3)) emf->o3=0.;  
  if(!getparint("rd1", &emf->rd1)) emf->rd1=2; 
  /* half length of FD stencil */
  if(!getparint("rd2", &emf->rd2)) emf->rd2=2; 
  /* half length of FD stencil */
  if(!getparint("rd3", &emf->rd3)) emf->rd3=2; 
  /* half length of FD stencil */
  emf->rd = MAX(MAX(emf->rd1, emf->rd2), emf->rd3);
  emf->n123 = emf->n1*emf->n2*emf->n3;
  emf->n1pad = emf->n1+2*emf->nb;/* total number of grid points after padding PML */
  emf->n2pad = emf->n2+2*emf->nb;/* total number of grid points after padding PML */
  emf->n3pad = emf->n3+  emf->nb;/* total number of grid points after padding PML */
  emf->n123pad = emf->n1pad*emf->n2pad*emf->n3pad;
  if(iproc==0){
    printf("[o1, o2, o3]=[%g, %g, %g]\n", emf->o1, emf->o2, emf->o3);    
    printf("[d1, d2, d3]=[%g, %g, %g]\n", emf->d1, emf->d2, emf->d3);
    printf("[n1, n2, n3, nb]=[%d, %d, %d, %d]\n", emf->n1, emf->n2, emf->n3, emf->nb);
    printf("[n1pad, n2pad, n3pad]=[%d, %d, %d]\n", emf->n1pad, emf->n2pad, emf->n3pad);
    printf("[rd1, rd2, rd3]=[%d, %d, %d]\n", emf->rd1, emf->rd2, emf->rd3);
  }

  if(!getparfloat("f0", &emf->f0)) emf->f0 = 1;
  emf->omega0 = 2.*PI*emf->f0;  /* reference frequency */
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
    if(iproc==0) printf("freq[%d]=%g\n", ifreq+1, emf->freqs[ifreq]);
  }
  
  /* read active source channels */
  if((emf->nchsrc=countparval("chsrc"))!=0) {
    emf->chsrc=(char**)alloc1(emf->nchsrc, sizeof(void*));
    getparstringarray("chsrc", emf->chsrc);
    /* active source channels: Ex, Ey, Ez, Hx, Hy, Hz or their combinations */
  }else{
    emf->nchsrc=1;
    emf->chsrc=(char**)alloc1(emf->nchsrc, sizeof(void*));
    emf->chsrc[0]="Ex";
  }
  /* read active receiver channels */
  if((emf->nchrec=countparval("chrec"))!=0) {
    emf->chrec=(char**)alloc1(emf->nchrec, sizeof(void*));
    getparstringarray("chrec", emf->chrec);
    /* active receiver channels: Ex, Ey, Ez, Hx, Hy, Hz or their combinations */
  }else{
    emf->nchrec=4;
    emf->chrec=(char**)alloc1(emf->nchrec, sizeof(void*));
    emf->chrec[0] = "Ex";
    emf->chrec[1] = "Ey";
    emf->chrec[2] = "Hx";
    emf->chrec[3] = "Hy";
  }
  if(iproc==0){
    printf("Active source channels:");
    for(ic=0; ic<emf->nchsrc; ++ic) printf(" %s", emf->chsrc[ic]);
    printf("\n");
    printf("Active recever channels:");
    for(ic=0; ic<emf->nchrec; ++ic) printf(" %s", emf->chrec[ic]);
    printf("\n");
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
  free3float(emf->rho11); 
  free3float(emf->rho22);
  free3float(emf->rho33);
}

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
  	emf->inveps11[i3][i2+emf->nb][i1+emf->nb] = t*emf->rho11[i3][i2][i1];
  	emf->inveps22[i3][i2+emf->nb][i1+emf->nb] = t*emf->rho22[i3][i2][i1];
  	emf->inveps33[i3][i2+emf->nb][i1+emf->nb] = t*emf->rho33[i3][i2][i1];
      }
    }
  }
  /* pad the left and the right face */
  for(i3=0; i3<emf->n3pad; i3++) {
    for(i2=0; i2<emf->n2pad; i2++) {
      for(i1=0; i1<emf->nb; i1++) {
  	j1 = emf->n1pad-1-i1;
  	emf->inveps11[i3][i2][i1] = emf->inveps11[i3][i2][emf->nb        ];
  	emf->inveps22[i3][i2][i1] = emf->inveps22[i3][i2][emf->nb        ];
  	emf->inveps33[i3][i2][i1] = emf->inveps33[i3][i2][emf->nb        ];
  	emf->inveps11[i3][i2][j1] = emf->inveps11[i3][i2][emf->n1pad-emf->nb-1];
  	emf->inveps22[i3][i2][j1] = emf->inveps22[i3][i2][emf->n1pad-emf->nb-1];
  	emf->inveps33[i3][i2][j1] = emf->inveps33[i3][i2][emf->n1pad-emf->nb-1];
      }
    }
  }
  /* pad the front and the rear face */
  for(i3=0; i3<emf->n3pad; i3++) {
    for(i2=0; i2<emf->nb; i2++) {
      j2 = emf->n2pad-i2-1;
      for(i1=0; i1<emf->n1pad; i1++) {
  	emf->inveps11[i3][i2][i1] = emf->inveps11[i3][emf->nb        ][i1];
  	emf->inveps22[i3][i2][i1] = emf->inveps22[i3][emf->nb        ][i1];
  	emf->inveps33[i3][i2][i1] = emf->inveps33[i3][emf->nb        ][i1];
  	emf->inveps11[i3][j2][i1] = emf->inveps11[i3][emf->n2pad-emf->nb-1][i1];
  	emf->inveps22[i3][j2][i1] = emf->inveps22[i3][emf->n2pad-emf->nb-1][i1];
  	emf->inveps33[i3][j2][i1] = emf->inveps33[i3][emf->n2pad-emf->nb-1][i1];
      }
    }
  }
  /* pad the top and the bottom face */
  for(i3=0; i3<emf->nb; i3++) {
    j3 = emf->n3pad-i3-1;
    for(i2=0; i2<emf->n2pad; i2++) {
      for(i1=0; i1<emf->n1pad; i1++) {
  	emf->inveps11[j3][i2][i1] = emf->inveps11[emf->n3pad-emf->nb-1][i2][i1];
  	emf->inveps22[j3][i2][i1] = emf->inveps22[emf->n3pad-emf->nb-1][i2][i1];
  	emf->inveps33[j3][i2][i1] = emf->inveps33[emf->n3pad-emf->nb-1][i2][i1];
      }
    }
  }

  /* air water interface: sigma = 0.5*(sigma_air+sigma_water)=0.5*sigma_water */
  //sigma = 2*w*epsil; epsil=sigma/(2w);
  i3=0;
  for(i2=0; i2<emf->n2pad; i2++){
    for(i1=0; i1<emf->n1pad; i1++){
      emf->inveps11[i3][i2][i1] *= 2;
      emf->inveps22[i3][i2][i1] *= 2;
    }//end for i1
  }//end for i2
}

void extend_model_close(emf_t *emf)
{
  free3float(emf->inveps11);
  free3float(emf->inveps22);
  free3float(emf->inveps33);
}


void sanity_check(emf_t *emf)
{
  int i1, i2, i3;
  float tmp1,tmp2,kappa,Glim, freqmax,cfl;

  /*----------------------------------------------------------------------------------*/
  /* Stage 2: find minimum and maximum velocity for stability conditon and dispersion */
  /*    emf->vmin: important for minimum number of points per wavelength              */
  /*    emf->vmax: important for CFL condition and fdtd computing box                 */
  /*----------------------------------------------------------------------------------*/
  emf->rhomax = MAX( MAX(emf->rho11[0][0][0], emf->rho22[0][0][0]), emf->rho33[0][0][0]);
  emf->rhomin = MIN( MIN(emf->rho11[0][0][0], emf->rho22[0][0][0]), emf->rho33[0][0][0]);
  for(i3=0; i3<emf->n3; ++i3){
    for(i2=0; i2<emf->n2; ++i2){
      for(i1=0; i1<emf->n1; ++i1){
	tmp1 = MIN( MIN(emf->rho11[i3][i2][i1], emf->rho22[i3][i2][i1]), emf->rho33[i3][i2][i1]);
	tmp2 = MAX( MAX(emf->rho11[i3][i2][i1], emf->rho22[i3][i2][i1]), emf->rho33[i3][i2][i1]);
	if(i3==0){//average conductivity over air-water interface, sigma_interface=0.5*sigma_water
	  //rho_interface=2*rho_water
	  tmp1 *= 2;
	  tmp2 *= 2;
	}
	if(emf->rhomin>tmp1)  emf->rhomin = tmp1;
	if(emf->rhomax<tmp2)  emf->rhomax = tmp2;
      }
    }
  }
  /* sigma=2*omega0*eps --> inveps=2*omega0*rho */
  emf->vmin = sqrt(2.*emf->omega0*emf->rhomin/mu0);
  emf->vmax = sqrt(2.*emf->omega0*emf->rhomax/mu0);

  /*------------------------------------------------------------------------*/
  /* Stage 3: determine the optimal dt and nt automatically                 */
  /*------------------------------------------------------------------------*/
  kappa = sqrt(1./(emf->d1*emf->d1)+1./(emf->d2*emf->d2)+1./(emf->d3*emf->d3));
  if(emf->rd==1){
    kappa *= 1;
    Glim = 10.;
  }else if(emf->rd==2){
    kappa *= (fabs(1.125) + fabs(-0.04167));
    Glim = 5.;
  }else if(emf->rd==3){
    kappa *= (fabs(1.171875) + fabs(-0.065104166666667) + fabs(0.0046875));
    Glim = 4.;
  }
  freqmax = emf->vmin/(Glim*MIN(MIN(emf->d1,emf->d2),emf->d3));
  
  if(!getparfloat("dt", &emf->dt)) emf->dt = 0.99/(kappa*emf->vmax);
  /* temporal sampling, determine dt by stability condition if not provided */
  cfl = emf->dt*emf->vmax*kappa;
  if(iproc==0) printf("cfl=%g\n", cfl); 
  if(cfl > 1.0) err("CFL condition not satisfied!");

  if(!getparint("nt", &emf->nt)){
    emf->nt = 2.*MAX((emf->n1-1)*emf->d1, (emf->n2-1)*emf->d2)/(emf->vmin*emf->dt);
  }/* automatically determine nt using maximum offset if not provided */
  if(iproc==0){
    printf("[rhomin, rhomax]=[%g, %g] Ohm-m\n", emf->rhomin, emf->rhomax);
    printf("wavespeed [vmin, vmax]=[%g, %g] m/s\n", emf->vmin, emf->vmax);
    printf("FD order=%d, Glim=%g ppw\n", 2*emf->rd, Glim);
    printf("freq<=%g Hz can be simulated\n", freqmax);
    printf("dt=%g s\n",  emf->dt);
    printf("nt=%d (maximum number of time steps)\n",  emf->nt);
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

  if(!(getparstring("fsrc", &fsrc))) err("Need fsrc= ");
  /* file to specify all possible source locations */
  if(!(getparstring("frec", &frec))) err("Need frec= ");
  /* file to specify all possible receiver locations */
  if(!(getparstring("fsrcrec", &fsrcrec))) err("Need fsrcrec= ");
  /* file to specify how source and receiver are combined */

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

      isrc++;
    }
  }
  acqui->nsrc_total = isrc;
  fclose(fp);

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
      
      irec++;
    }
  }
  acqui->nrec_total = irec;
  fclose(fp);

  //-----------------------------------------------------------
  acqui->nsrc = 1; /* assume 1 source per process by default */
  acqui->src_x1 = alloc1float(acqui->nsrc);
  acqui->src_x2 = alloc1float(acqui->nsrc);
  acqui->src_x3 = alloc1float(acqui->nsrc);
  acqui->src_azimuth = alloc1float(acqui->nsrc);
  acqui->src_dip = alloc1float(acqui->nsrc);
  x1min = src_x1[0];
  x1max = src_x1[0];
  x2min = src_x2[0];
  x2max = src_x2[0];
  x3min = src_x3[0];
  x3max = src_x3[0];
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
    x1min = MIN(x1min, acqui->src_x1[isrc]);
    x1max = MAX(x1max, acqui->src_x1[isrc]);
    x2min = MIN(x2min, acqui->src_x2[isrc]);
    x2max = MAX(x2max, acqui->src_x2[isrc]);
    x3min = MIN(x3min, acqui->src_x3[isrc]);
    x3max = MAX(x3max, acqui->src_x3[isrc]);
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
    x1min = MIN(x1min, acqui->rec_x1[irec]);
    x1max = MAX(x1max, acqui->rec_x1[irec]);
    x2min = MIN(x2min, acqui->rec_x2[irec]);
    x2max = MAX(x2max, acqui->rec_x2[irec]);
    x3min = MIN(x3min, acqui->rec_x3[irec]);
    x3max = MAX(x3max, acqui->rec_x3[irec]);    
  }/* end for irec */
  if(iproc==0){
    printf("src-rec [x1min, x1max]=[%g, %g]\n", x1min, x1max);
    printf("src-rec [x2min, x2max]=[%g, %g]\n", x2min, x2max);
    printf("src-rec [x3min, x3max]=[%g, %g]\n", x3min, x3max);
    printf("nsrc_total=%d\n", acqui->nsrc_total);
    printf("nrec_total=%d\n", acqui->nrec_total);
  }
  printf("isrc=%d, nrec=%d (x,y,z)=(%.2f, %.2f, %.2f)\n",
	 acqui->shot_idx[iproc], acqui->nrec, acqui->src_x1[0], acqui->src_x2[0], acqui->src_x3[0]);
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

void cpml_init(emf_t *emf)
/*< initialize PML abosorbing coefficients >*/
{
  float x, damp0, damp;
  int i1;

  /* by default, we choose: kappa=1, alpha=PI*f0 for CPML */
  float alpha=PI*emf->f0; /* alpha>0 makes CPML effectively attenuates evanescent waves */
  
  emf->apml = alloc1float(emf->nb);
  emf->bpml = alloc1float(emf->nb);
  //damp0=-3.*vmax*logf(Rc)/(2.*L);
  damp0 = 341.9;
  for(i1=0; i1<emf->nb; ++i1)    {
    x=(float)(emf->nb-i1)/(float)(emf->nb);
    damp = damp0*x*x; /* damping profile in direction 1, sigma/epsilon0 */
    emf->bpml[i1] = expf(-(damp+alpha)*emf->dt);
    emf->apml[i1] = damp*(emf->bpml[i1]-1.0)/(damp+alpha);
  }
}


void cpml_close(emf_t *emf)
{
  free(emf->apml);
  free(emf->bpml);
}


/*----------------------------------------------------------------*/
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
  
}

/* nullify EM fields which have beeen allocated */
void fdtd_null(emf_t *emf)
{  
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

void fdtd_curlH(emf_t *emf, int it)
{
  int i1, i2, i3, j1, j2, j3, k1, k2, k3;
  float D2H3, D3H2, D3H1, D1H3, D1H2, D2H1;
  float c11, c21, c31, c12, c22, c32, c13, c23, c33;

  if(emf->rd1==1){
    c11 = 1./emf->d1;
  }else if(emf->rd1==2){
    c11 = 1.125/emf->d1;
    c21 = -0.041666666666666664/emf->d1;
  }else if(emf->rd1==3){
    c11 = 1.171875/emf->d1;
    c21 = -0.065104166666667/emf->d1;
    c31 = 0.0046875/emf->d1;
  }

  if(emf->rd2==1){
    c12 = 1./emf->d2;
  }else if(emf->rd2==2){
    c12 = 1.125/emf->d2;
    c22 = -0.041666666666666664/emf->d2;
  }else if(emf->rd2==3){
    c12 = 1.171875/emf->d2;
    c22 = -0.065104166666667/emf->d2;
    c32 = 0.0046875/emf->d2;
  }

  if(emf->rd3==1){
    c13 = 1./emf->d3;
  }else if(emf->rd3==2){
    c13 = 1.125/emf->d3;
    c23 = -0.041666666666666664/emf->d3;
  }else if(emf->rd3==3){
    c13 = 1.171875/emf->d3;
    c23 = -0.065104166666667/emf->d3;
    c33 = 0.0046875/emf->d3;
  }

#ifdef _OPENMP
#pragma omp parallel for default(none)				\
  schedule(static)						\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3,			\
	  D2H3, D3H2, D3H1, D1H3, D1H2, D2H1)			\
  shared(c11, c21, c31, c12, c22, c32, c13, c23, c33, emf)
#endif
  for(i3=0; i3<=emf->n3pad-emf->rd3; ++i3){
    for(i2=emf->rd2; i2<=emf->n2pad-emf->rd2; ++i2){
      for(i1=emf->rd1; i1<=emf->n1pad-emf->rd1; ++i1){
	if(emf->rd1==1){
	  D1H3 = c11*(emf->H3[i3][i2][i1]-emf->H3[i3][i2][i1-1]);
	  D1H2 = c11*(emf->H2[i3][i2][i1]-emf->H2[i3][i2][i1-1]);
	}else if(emf->rd1==2){
	  D1H3 = c11*(emf->H3[i3][i2][i1]-emf->H3[i3][i2][i1-1])
	    + c21*(emf->H3[i3][i2][i1+1]-emf->H3[i3][i2][i1-2]);
	  D1H2 = c11*(emf->H2[i3][i2][i1]-emf->H2[i3][i2][i1-1])
	    + c21*(emf->H2[i3][i2][i1+1]-emf->H2[i3][i2][i1-2]);
	}else if(emf->rd1==3){
	  D1H3 = c11*(emf->H3[i3][i2][i1]-emf->H3[i3][i2][i1-1])
	    + c21*(emf->H3[i3][i2][i1+1]-emf->H3[i3][i2][i1-2])
	    + c31*(emf->H3[i3][i2][i1+2]-emf->H3[i3][i2][i1-3]);
	  D1H2 = c11*(emf->H2[i3][i2][i1]-emf->H2[i3][i2][i1-1])
	    + c21*(emf->H2[i3][i2][i1+1]-emf->H2[i3][i2][i1-2])
	    + c31*(emf->H2[i3][i2][i1+2]-emf->H2[i3][i2][i1-3]);
	}	

	if(emf->rd2==1){
	  D2H3 = c12*(emf->H3[i3][i2][i1]-emf->H3[i3][i2-1][i1]);
	  D2H1 = c12*(emf->H1[i3][i2][i1]-emf->H1[i3][i2-1][i1]);
	}else if(emf->rd2==2){
	  D2H3 = c12*(emf->H3[i3][i2][i1]-emf->H3[i3][i2-1][i1])
	    + c22*(emf->H3[i3][i2+1][i1]-emf->H3[i3][i2-2][i1]);
	  D2H1 = c12*(emf->H1[i3][i2][i1]-emf->H1[i3][i2-1][i1])
	    + c22*(emf->H1[i3][i2+1][i1]-emf->H1[i3][i2-2][i1]);
	}else if(emf->rd2==3){
	  D2H3 = c12*(emf->H3[i3][i2][i1]-emf->H3[i3][i2-1][i1])
	    + c22*(emf->H3[i3][i2+1][i1]-emf->H3[i3][i2-2][i1])
	    + c32*(emf->H3[i3][i2+2][i1]-emf->H3[i3][i2-3][i1]);
	  D2H1 = c12*(emf->H1[i3][i2][i1]-emf->H1[i3][i2-1][i1])
	    + c22*(emf->H1[i3][i2+1][i1]-emf->H1[i3][i2-2][i1])
	    + c32*(emf->H1[i3][i2+2][i1]-emf->H1[i3][i2-3][i1]);
	}
	//note the extrapolated H1,H2 are in memory variables memD3H1 and memD3H2
	if(emf->rd3==1){
	  if(i3>=1){
	    D3H2 = c13*(emf->H2[i3][i2][i1]-emf->H2[i3-1][i2][i1]);
	    D3H1 = c13*(emf->H1[i3][i2][i1]-emf->H1[i3-1][i2][i1]);
	  }else{
	    D3H2 = c13*(emf->H2[i3][i2][i1]-emf->memD3H2[0][i2][i1]);
	    D3H1 = c13*(emf->H1[i3][i2][i1]-emf->memD3H1[0][i2][i1]);
	  }
	}else if(emf->rd3==2){
	  if(i3>=2){
	    D3H2 = c13*(emf->H2[i3][i2][i1]-emf->H2[i3-1][i2][i1])
	      + c23*(emf->H2[i3+1][i2][i1]-emf->H2[i3-2][i2][i1]);
	    D3H1 = c13*(emf->H1[i3][i2][i1]-emf->H1[i3-1][i2][i1])
	      + c23*(emf->H1[i3+1][i2][i1]-emf->H1[i3-2][i2][i1]);
	  }else if(i3==1){
	    D3H2 = c13*(emf->H2[i3][i2][i1]-emf->H2[i3-1][i2][i1])
	      + c23*(emf->H2[i3+1][i2][i1]-emf->memD3H2[0][i2][i1]);
	    D3H1 = c13*(emf->H1[i3][i2][i1]-emf->H1[i3-1][i2][i1])
	      + c23*(emf->H1[i3+1][i2][i1]-emf->memD3H1[0][i2][i1]);
	  }else{//i3==0
	    D3H2 = c13*(emf->H2[i3][i2][i1]-emf->memD3H2[0][i2][i1])
	      + c23*(emf->H2[i3+1][i2][i1]-emf->memD3H2[1][i2][i1]);
	    D3H1 = c13*(emf->H1[i3][i2][i1]-emf->memD3H1[0][i2][i1])
	      + c23*(emf->H1[i3+1][i2][i1]-emf->memD3H1[0][i2][i1]);
	  }
	}else if(emf->rd3==3){
	  if(i3>=3){
	    D3H2 = c13*(emf->H2[i3][i2][i1]-emf->H2[i3-1][i2][i1])
	      + c23*(emf->H2[i3+1][i2][i1]-emf->H2[i3-2][i2][i1])
	      + c33*(emf->H2[i3+2][i2][i1]-emf->H2[i3-3][i2][i1]);
	    D3H1 = c13*(emf->H1[i3][i2][i1]-emf->H1[i3-1][i2][i1])
	      + c23*(emf->H1[i3+1][i2][i1]-emf->H1[i3-2][i2][i1])
	      + c33*(emf->H1[i3+2][i2][i1]-emf->H1[i3-3][i2][i1]);
	  }else if(i3==2){
	    D3H2 = c13*(emf->H2[i3][i2][i1]-emf->H2[i3-1][i2][i1])
	      + c23*(emf->H2[i3+1][i2][i1]-emf->H2[i3-2][i2][i1])
	      + c33*(emf->H2[i3+2][i2][i1]-emf->memD3H2[0][i2][i1]);
	    D3H1 = c13*(emf->H1[i3][i2][i1]-emf->H1[i3-1][i2][i1])
	      + c23*(emf->H1[i3+1][i2][i1]-emf->H1[i3-2][i2][i1])
	      + c33*(emf->H1[i3+2][i2][i1]-emf->memD3H1[0][i2][i1]);
	  }else if(i3==1){
	    D3H2 = c13*(emf->H2[i3][i2][i1]-emf->H2[i3-1][i2][i1])
	      + c23*(emf->H2[i3+1][i2][i1]-emf->memD3H2[0][i2][i1])
	      + c33*(emf->H2[i3+2][i2][i1]-emf->memD3H2[1][i2][i1]);
	    D3H1 = c13*(emf->H1[i3][i2][i1]-emf->H1[i3-1][i2][i1])
	      + c23*(emf->H1[i3+1][i2][i1]-emf->memD3H1[0][i2][i1])
	      + c33*(emf->H1[i3+2][i2][i1]-emf->memD3H1[1][i2][i1]);
	  }else if(i3==0){
	    D3H2 = c13*(emf->H2[i3][i2][i1]-emf->memD3H2[0][i2][i1])
	      + c23*(emf->H2[i3+1][i2][i1]-emf->memD3H2[1][i2][i1])
	      + c33*(emf->H2[i3+2][i2][i1]-emf->memD3H2[2][i2][i1]);
	    D3H1 = c13*(emf->H1[i3][i2][i1]-emf->memD3H1[0][i2][i1])
	      + c23*(emf->H1[i3+1][i2][i1]-emf->memD3H1[1][i2][i1])
	      + c33*(emf->H1[i3+2][i2][i1]-emf->memD3H1[2][i2][i1]);
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
	if(i3>emf->n3pad-1-emf->nb){
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


void fdtd_update_E(emf_t *emf, int it)
{
  int i1, i2, i3;

#ifdef _OPENMP
#pragma omp parallel for default(none)		\
  schedule(static)				\
  private(i1, i2, i3)				\
  shared(emf)
#endif
  for(i3=0; i3<emf->n3pad; ++i3){
    for(i2=0; i2<emf->n2pad; ++i2){
      for(i1=0; i1<emf->n1pad; ++i1){
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

void fdtd_curlE(emf_t *emf, int it)
{
  int i1, i2, i3, j1, j2, j3, k1, k2, k3;
  float D2E3, D3E2, D3E1, D1E3, D1E2, D2E1;
  float c11, c21, c31, c12, c22, c32, c13, c23, c33;

  if(emf->rd1==1){
    c11 = 1./emf->d1;
  }else if(emf->rd1==2){
    c11 = 1.125/emf->d1;
    c21 = -0.041666666666666664/emf->d1;
  }else if(emf->rd1==3){
    c11 = 1.171875/emf->d1;
    c21 = -0.065104166666667/emf->d1;
    c31 = 0.0046875/emf->d1;
  }

  if(emf->rd2==1){
    c12 = 1./emf->d2;
  }else if(emf->rd2==2){
    c12 = 1.125/emf->d2;
    c22 = -0.041666666666666664/emf->d2;
  }else if(emf->rd2==3){
    c12 = 1.171875/emf->d2;
    c22 = -0.065104166666667/emf->d2;
    c32 = 0.0046875/emf->d2;
  }

  if(emf->rd3==1){
    c13 = 1./emf->d3;
  }else if(emf->rd3==2){
    c13 = 1.125/emf->d3;
    c23 = -0.041666666666666664/emf->d3;
  }else if(emf->rd3==3){
    c13 = 1.171875/emf->d3;
    c23 = -0.065104166666667/emf->d3;
    c33 = 0.0046875/emf->d3;
  }

#ifdef _OPENMP
#pragma omp parallel for default(none)				\
  schedule(static)						\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3,			\
	  D2E3, D3E2, D3E1, D1E3, D1E2, D2E1)			\
  shared(c11, c21, c31, c12, c22, c32, c13, c23, c33, emf)
#endif
  for(i3=0; i3<emf->n3pad-emf->rd3; ++i3){
    for(i2=emf->rd2; i2<emf->n2pad-emf->rd2; ++i2){
      for(i1=emf->rd1-1; i1<emf->n1pad-emf->rd1; ++i1){
	if(emf->rd1==1){
	  D1E3 = c11*(emf->E3[i3][i2][i1+1]-emf->E3[i3][i2][i1]);
	  D1E2 = c11*(emf->E2[i3][i2][i1+1]-emf->E2[i3][i2][i1]);
	}else if(emf->rd1==2){
	  D1E3 = c11*(emf->E3[i3][i2][i1+1]-emf->E3[i3][i2][i1])
	    + c21*(emf->E3[i3][i2][i1+2]-emf->E3[i3][i2][i1-1]);
	  D1E2 = c11*(emf->E2[i3][i2][i1+1]-emf->E2[i3][i2][i1])
	    + c21*(emf->E2[i3][i2][i1+2]-emf->E2[i3][i2][i1-1]);
	}else if(emf->rd1==3){
	  D1E3 = c11*(emf->E3[i3][i2][i1+1]-emf->E3[i3][i2][i1])
	    + c21*(emf->E3[i3][i2][i1+2]-emf->E3[i3][i2][i1-1])
	    + c31*(emf->E3[i3][i2][i1+3]-emf->E3[i3][i2][i1-2]);
	  D1E2 = c11*(emf->E2[i3][i2][i1+1]-emf->E2[i3][i2][i1])
	    + c21*(emf->E2[i3][i2][i1+2]-emf->E2[i3][i2][i1-1])
	    + c31*(emf->E2[i3][i2][i1+3]-emf->E2[i3][i2][i1-2]);
	}

	if(emf->rd2==1){
	  D2E3 = c12*(emf->E3[i3][i2+1][i1]-emf->E3[i3][i2][i1]);
	  D2E1 = c12*(emf->E1[i3][i2+1][i1]-emf->E1[i3][i2][i1]);
	}else if(emf->rd2==2){
	  D2E3 = c12*(emf->E3[i3][i2+1][i1]-emf->E3[i3][i2][i1])
	    + c22*(emf->E3[i3][i2+2][i1]-emf->E3[i3][i2-1][i1]);
	  D2E1 = c12*(emf->E1[i3][i2+1][i1]-emf->E1[i3][i2][i1])
	    + c22*(emf->E1[i3][i2+2][i1]-emf->E1[i3][i2-1][i1]);
	}else if(emf->rd2==3){
	  D2E3 = c12*(emf->E3[i3][i2+1][i1]-emf->E3[i3][i2][i1])
	    + c22*(emf->E3[i3][i2+2][i1]-emf->E3[i3][i2-1][i1])
	    + c32*(emf->E3[i3][i2+3][i1]-emf->E3[i3][i2-2][i1]);
	  D2E1 = c12*(emf->E1[i3][i2+1][i1]-emf->E1[i3][i2][i1])
	    + c22*(emf->E1[i3][i2+2][i1]-emf->E1[i3][i2-1][i1])
	    + c32*(emf->E1[i3][i2+3][i1]-emf->E1[i3][i2-2][i1]);
	}

	//note the extrapolated E1,E2 are in memory variables memD3E1 and memD3E2
	if(emf->rd3==1){
	  D3E2 = c13*(emf->E2[i3+1][i2][i1]-emf->E2[i3][i2][i1]);
	  D3E1 = c13*(emf->E1[i3+1][i2][i1]-emf->E1[i3][i2][i1]);
	}else if(emf->rd3==2){
	  if(i3>=1){
	    D3E2 = c13*(emf->E2[i3+1][i2][i1]-emf->E2[i3][i2][i1])
	      + c23*(emf->E2[i3+2][i2][i1]-emf->E2[i3-1][i2][i1]);
	    D3E1 = c13*(emf->E1[i3+1][i2][i1]-emf->E1[i3][i2][i1])
	      + c23*(emf->E1[i3+2][i2][i1]-emf->E1[i3-1][i2][i1]);
	  }else if(i3==0){
	    D3E2 = c13*(emf->E2[i3+1][i2][i1]-emf->E2[i3][i2][i1])
	      + c23*(emf->E2[i3+2][i2][i1]-emf->memD3E2[0][i2][i1]);
	    D3E1 = c13*(emf->E1[i3+1][i2][i1]-emf->E1[i3][i2][i1])
	      + c23*(emf->E1[i3+2][i2][i1]-emf->memD3E1[0][i2][i1]);
	  }
	}else if(emf->rd3==3){
	  if(i3>=2){
	    D3E2 = c13*(emf->E2[i3+1][i2][i1]-emf->E2[i3][i2][i1])
	      + c23*(emf->E2[i3+2][i2][i1]-emf->E2[i3-1][i2][i1])
	      + c33*(emf->E2[i3+3][i2][i1]-emf->E2[i3-2][i2][i1]);
	    D3E1 = c13*(emf->E1[i3+1][i2][i1]-emf->E1[i3][i2][i1])
	      + c23*(emf->E1[i3+2][i2][i1]-emf->E1[i3-1][i2][i1])
	      + c33*(emf->E1[i3+3][i2][i1]-emf->E1[i3-2][i2][i1]);
	  }else if(i3==1){
	    D3E2 = c13*(emf->E2[i3+1][i2][i1]-emf->E2[i3][i2][i1])
	      + c23*(emf->E2[i3+2][i2][i1]-emf->E2[i3-1][i2][i1])
	      + c33*(emf->E2[i3+3][i2][i1]-emf->memD3E2[0][i2][i1]);
	    D3E1 = c13*(emf->E1[i3+1][i2][i1]-emf->E1[i3][i2][i1])
	      + c23*(emf->E1[i3+2][i2][i1]-emf->E1[i3-1][i2][i1])
	      + c33*(emf->E1[i3+3][i2][i1]-emf->memD3E1[0][i2][i1]);
	  }else if(i3==2){
	    D3E2 = c13*(emf->E2[i3+1][i2][i1]-emf->E2[i3][i2][i1])
	      + c23*(emf->E2[i3+2][i2][i1]-emf->memD3E2[0][i2][i1])
	      + c33*(emf->E2[i3+3][i2][i1]-emf->memD3E2[1][i2][i1]);
	    D3E1 = c13*(emf->E1[i3+1][i2][i1]-emf->E1[i3][i2][i1])
	      + c23*(emf->E1[i3+2][i2][i1]-emf->memD3E1[0][i2][i1])
	      + c33*(emf->E1[i3+3][i2][i1]-emf->memD3E1[1][i2][i1]);
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
	if(i3>emf->n3pad-1-emf->nb){
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

void fdtd_update_H(emf_t *emf, int it)
{
  int i1, i2, i3;
  
  float factor = emf->dt/mu0;
#ifdef _OPENMP
#pragma omp parallel for default(none)		\
  schedule(static)				\
  private(i1, i2, i3)				\
  shared(emf, factor)
#endif
  for(i3=0; i3<emf->n3pad; ++i3){
    for(i2=0; i2<emf->n2pad; ++i2){
      for(i1=0; i1<emf->n1pad; ++i1){
	emf->H1[i3][i2][i1] -= factor* emf->curlE1[i3][i2][i1];
	emf->H2[i3][i2][i1] -= factor* emf->curlE2[i3][i2][i1];
	emf->H3[i3][i2][i1] -= factor* emf->curlE3[i3][i2][i1];
      }
    }
  }
}

void vandermonde(int n, float *x, float *a, float *f)
{
  int i, k;

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

/*----------------------------------------------------------------*/
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
  const float eps=1e-5;

  float o1, o2, o3;
  float x1j, x2j, x3j;
  int j1, j2, j3;
  int i, j, isub, loop, m, isrc, irec;
  bool skip1, skip2, skip3;
  float *x1, *x2, *x3, *aa, *ff;
  float tmp, dlen1, dlen2, dlen3;
  interp_t *interp;
  
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
	  o1 = emf->o1;
	  o2 = emf->o2;
	  o3 = emf->o3;
	}else{/* staggered grid */
	  interp = interp_sg;
	  o1 = emf->o1+0.5*emf->d1;
	  o2 = emf->o2+0.5*emf->d2;
	  o3 = emf->o3+0.5*emf->d3;
	}
	j1 = (int)((x1j-o1)/emf->d1);/* integer part */
	j2 = (int)((x2j-o2)/emf->d2);/* integer part */
	j3 = (int)((x3j-o3)/emf->d3);/* integer part */

	interp->rec_i1[irec][isub] = j1 + emf->nb;
	interp->rec_i2[irec][isub] = j2 + emf->nb;
	interp->rec_i3[irec][isub] = j3;//no points above water

	skip1 = false;
	skip2 = false;
	skip3 = false;
	if(fabs(x1j-o1-j1*emf->d1)< eps){
	  memset(interp->rec_w1[irec][isub], 0, 2*emf->rd*sizeof(float));
	  interp->rec_w1[irec][isub][emf->rd1-1] = 1.;
	  skip1 = true;
	}
	if(fabs(x2j-o2-j2*emf->d2)< eps) {
	  memset(interp->rec_w2[irec][isub], 0, 2*emf->rd*sizeof(float));
	  interp->rec_w2[irec][isub][emf->rd2-1] = 1.;
	  skip2 = true;
	}
	if(fabs(x3j-o2-j3*emf->d3)< eps) {
	  memset(interp->rec_w3[irec][isub], 0, 2*emf->rd*sizeof(float));
	  interp->rec_w3[irec][isub][emf->rd3-1] = 1.;
	  skip3 = true;
	}

	for(i=0; i<2*emf->rd; i++){/* construct vector x1[], x2[], x3[] */
	  m = i-emf->rd1+1;/* m=offset/shift between [-emf->rd+1, emf->rd] */
	  if(!skip1) x1[i] = o1+(j1+m)*emf->d1-x1j;
	  m = i-emf->rd2+1;/* m=offset/shift between [-emf->rd+1, emf->rd] */
	  if(!skip2) x2[i] = o2+(j2+m)*emf->d2-x2j;
	  m = i-emf->rd3+1;/* m=offset/shift between [-emf->rd+1, emf->rd] */
	  if(!skip3) x3[i] = o3+(j3+m)*emf->d3-x3j;
	}/* end for i */
	
	for(i=0; i<2*emf->rd; i++){
	  memset(ff, 0, 2*emf->rd*sizeof(float));
	  ff[i] =  1.;
	  if(!skip1){
	    vandermonde(2*emf->rd1-1, x1, aa, ff);
	    interp->rec_w1[irec][isub][i] = aa[0];
	  }
	  if(!skip2){
	    vandermonde(2*emf->rd2-1, x2, aa, ff);
	    interp->rec_w2[irec][isub][i] = aa[0];
	  }
	  if(!skip3){
	    vandermonde(2*emf->rd3-1, x3, aa, ff);
	    interp->rec_w3[irec][isub][i] = aa[0];
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
	  o1 = emf->o1;
	  o2 = emf->o2;
	  o3 = emf->o3;
	}else{/* staggered grid */
	  interp = interp_sg;
	  o1 = emf->o1+0.5*emf->d1;
	  o2 = emf->o2+0.5*emf->d2;
	  o3 = emf->o3+0.5*emf->d3;
	}
	j1 = (int)((x1j-o1)/emf->d1);/* integer part */
	j2 = (int)((x2j-o2)/emf->d2);/* integer part */
	j3 = (int)((x3j-o3)/emf->d3);/* integer part */

	interp->src_i1[isrc][isub] = j1 + emf->nb;
	interp->src_i2[isrc][isub] = j2 + emf->nb;
	interp->src_i3[isrc][isub] = j3;//no points above water

	skip1 = false;
	skip2 = false;
	skip3 = false;
	if(fabs(x1j-o1-j1*emf->d1)< eps){
	  memset(interp->src_w1[isrc][isub], 0, 2*emf->rd*sizeof(float));
	  interp->src_w1[isrc][isub][emf->rd1-1] = 1.;
	  skip1 = true;
	}
	if(fabs(x2j-o2-j2*emf->d2)< eps) {
	  memset(interp->src_w2[isrc][isub], 0, 2*emf->rd*sizeof(float));
	  interp->src_w2[isrc][isub][emf->rd2-1] = 1.;
	  skip2 = true;
	}
	if(fabs(x3j-o3-j3*emf->d3)< eps) {
	  memset(interp->src_w3[isrc][isub], 0, 2*emf->rd*sizeof(float));
	  interp->src_w3[isrc][isub][emf->rd3-1] = 1.;
	  skip3 = true;
	}
	for(i=0; i<2*emf->rd; i++){/* construct vector x1[], x2[], x3[] */
	  m = i-emf->rd1+1;/* m=offset/shift between [-emf->rd+1, emf->rd] */
	  if(!skip1) x1[i] = o1+(j1+m)*emf->d1-x1j;
	  m = i-emf->rd2+1;/* m=offset/shift between [-emf->rd+1, emf->rd] */
	  if(!skip2) x2[i] = o2+(j2+m)*emf->d2-x2j;
	  m = i-emf->rd3+1;/* m=offset/shift between [-emf->rd+1, emf->rd] */
	  if(!skip3) x3[i] = o3+(j3+m)*emf->d3-x3j;
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
  float w1, w2, w3, s;

  s = emf->stf[it]/(emf->d1*emf->d2*emf->d3);/* source normalized by volume */
  s /= (float)acqui->nsubsrc; /*since one source is distributed over many points */
  for(isrc=0; isrc<acqui->nsrc; isrc++){
    for(isub=0; isub<acqui->nsubsrc; isub++){

      for(ic=0; ic<emf->nchsrc; ++ic){
	if(strcmp(emf->chsrc[ic], "Ex") == 0){
	  /* staggered grid: E1[i1, i2, i3] = Ex[i1+0.5, i2, i3] */
	  ix1 = interp_sg->src_i1[isrc][isub];
	  ix2 = interp_rg->src_i2[isrc][isub];
	  ix3 = interp_rg->src_i3[isrc][isub];
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
	}else if(strcmp(emf->chsrc[ic], "Ey") == 0){
	  /* staggered grid: E2[i1, i2, i3] = Ey[i1, i2+0.5, i3] */
	  ix1 = interp_rg->src_i1[isrc][isub];
	  ix2 = interp_sg->src_i2[isrc][isub];
	  ix3 = interp_rg->src_i3[isrc][isub];
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
	}else if(strcmp(emf->chsrc[ic], "Ez") == 0){
	  /* staggered grid: E3[i1, i2, i3] = Ez[i1, i2, i3+0.5] */
	  ix1 = interp_rg->src_i1[isrc][isub];
	  ix2 = interp_rg->src_i2[isrc][isub];
	  ix3 = interp_sg->src_i3[isrc][isub];
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


fftw_complex *emf_kxky, *emf_kxkyz0;
fftw_plan fft_airwave, ifft_airwave;

int fft_next_fast_size(int n)
{
  int m,p;
  /* m = 1; */
  /* while(m<=n) m *= 2; */
  /* return 2*m; */

  p = 4*n;
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
  float dkx, dky, kz;
  int i1, i2, i3;
  float *kx, *ky;
  
  emf->n1fft = fft_next_fast_size(emf->n1pad);
  emf->n2fft = fft_next_fast_size(emf->n2pad);

  /* complex scaling factor for H1, H2, E1, E2 */
  emf->sH1kxky = alloc3complexf(emf->n1fft, emf->n2fft, emf->rd3);
  emf->sH2kxky = alloc3complexf(emf->n1fft, emf->n2fft, emf->rd3);
  if(emf->rd3>1) emf->sE12kxky = alloc3float(emf->n1fft, emf->n2fft, emf->rd3-1);
  
  /* FE3 is not necessary in the air because we do not compute derivates of Hx
   * and Hy in the air: Hx and Hy are derived directly by extrapolation from Hz. */
  emf_kxky = fftw_malloc(sizeof(fftw_complex)*emf->n1fft*emf->n2fft);
  emf_kxkyz0 = fftw_malloc(sizeof(fftw_complex)*emf->n1fft*emf->n2fft);
  /* comapred with FFTW, we have opposite sign convention for time, same sign convetion for space */
  fft_airwave = fftw_plan_dft_2d(emf->n1fft, emf->n2fft, emf_kxky, emf_kxky, FFTW_FORWARD, FFTW_ESTIMATE);
  ifft_airwave = fftw_plan_dft_2d(emf->n1fft, emf->n2fft, emf_kxky, emf_kxky, FFTW_BACKWARD, FFTW_ESTIMATE);  
  
  kx = alloc1float(emf->n1fft);
  ky = alloc1float(emf->n2fft);
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
  
  for(i3=0; i3<emf->rd3; i3++){
    for(i2=0; i2<emf->n2fft; i2++){
      for(i1=0; i1<emf->n1fft; i1++){
	kz = sqrt(kx[i1]*kx[i1]+ky[i2]*ky[i2]);
	emf->sH1kxky[i3][i2][i1] = exp(-kz*(i3+0.5)*emf->d3)*cexp(-I*kx[i1]*0.5*emf->d1)*I*kx[i1]/(kz+1.e-15);
	emf->sH2kxky[i3][i2][i1] = exp(-kz*(i3+0.5)*emf->d3)*cexp(-I*ky[i2]*0.5*emf->d2)*I*ky[i2]/(kz+1.e-15);
      }
    }
  }
  for(i3=0; i3<emf->rd3-1; i3++){
    for(i2=0; i2<emf->n2fft; i2++){
      for(i1=0; i1<emf->n1fft; i1++){
	kz = sqrt(kx[i1]*kx[i1]+ky[i2]*ky[i2]);
	emf->sE12kxky[i3][i2][i1] = exp(-kz*(i3+1)*emf->d3);
      }
    }
  }
 
  free(kx);
  free(ky);
}

void airwave_bc_close(emf_t *emf)
{
  free3complexf(emf->sH1kxky);
  free3complexf(emf->sH2kxky);
  if(emf->rd3>1) free3float(emf->sE12kxky);

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
	emf_kxky[i1+emf->n1fft*i2] = emf->H3[0][i2][i1];
      else
	emf_kxky[i1+emf->n1fft*i2] = 0.;
    }
  }
  fftw_execute(fft_airwave);/*Hz(x,y,z=0)-->Hz(kx,ky,z=0)*/
  memcpy(emf_kxkyz0, emf_kxky, emf->n1fft*emf->n2fft*sizeof(fftw_complex));

  /*----------------------------------- H1 -------------------------------------*/
  for(i3=0; i3<emf->rd3; i3++){
    for(i2=0; i2<emf->n2fft; i2++){
      for(i1=0; i1<emf->n1fft; i1++){
	emf_kxky[i1+emf->n1fft*i2] = emf_kxkyz0[i1+emf->n1fft*i2]*emf->sH1kxky[i3][i2][i1];
      }
    }
    fftw_execute(ifft_airwave);
    for(i2=0; i2<emf->n2pad; i2++){
      for(i1=0; i1<emf->n1pad; i1++){
	//we store extrapolated values in the unused part of memory variables
	emf->memD3H1[i3][i2][i1] = creal(emf_kxky[i1+emf->n1fft*i2]/(emf->n1fft*emf->n2fft));
      }
    }
    
  }

  /*----------------------------------- H2 -------------------------------------*/
  for(i3=0; i3<emf->rd3; i3++){
    for(i2=0; i2<emf->n2fft; i2++){
      for(i1=0; i1<emf->n1fft; i1++){
	emf_kxky[i1+emf->n1fft*i2] = emf_kxkyz0[i1+emf->n1fft*i2]*emf->sH2kxky[i3][i2][i1];
      }
    }
    fftw_execute(ifft_airwave);
    for(i2=0; i2<emf->n2pad; i2++){
      for(i1=0; i1<emf->n1pad; i1++){
	//we store extrapolated values in the unused part of memory variables
	emf->memD3H2[i3][i2][i1] = creal(emf_kxky[i1+emf->n1fft*i2]/(emf->n1fft*emf->n2fft));
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
	emf_kxky[i1+emf->n1fft*i2] = emf->E1[0][i2][i1];
      else
	emf_kxky[i1+emf->n1fft*i2] = 0.;
    }
  }
  fftw_execute(fft_airwave);/* Ex(x,y,z=0)-->Hx(kx,ky,z=0) */
  memcpy(emf_kxkyz0, emf_kxky, emf->n1fft*emf->n2fft*sizeof(fftw_complex));
  
  for(i3=0; i3<emf->rd3-1; i3++){
    for(i2=0; i2<emf->n2fft; i2++){
      for(i1=0; i1<emf->n1fft; i1++){
	emf_kxky[i1+emf->n1fft*i2] = emf_kxkyz0[i1+emf->n1fft*i2]*emf->sE12kxky[i3][i2][i1];
      }
    }
    fftw_execute(ifft_airwave);
    for(i2=0; i2<emf->n2pad; i2++){
      for(i1=0; i1<emf->n1pad; i1++){
	//we store extrapolated values in the unused part of memory variables
	emf->memD3E1[i3][i2][i1] = creal(emf_kxky[i1+emf->n1fft*i2]/(emf->n1fft*emf->n2fft));
      }
    }
  }

  /*-----------------------------E2---------------------------------------*/
  for(i2=0; i2<emf->n2fft; i2++){
    for(i1=0; i1<emf->n1fft; i1++){
      if(i1<emf->n1pad && i2<emf->n2pad)
	emf_kxky[i1+emf->n1fft*i2] = emf->E2[0][i2][i1];
      else
	emf_kxky[i1+emf->n1fft*i2] = 0.;
    }
  }
  fftw_execute(fft_airwave);/* Ex(x,y,z=0)-->Hx(kx,ky,z=0) */
  memcpy(emf_kxkyz0, emf_kxky, emf->n1fft*emf->n2fft*sizeof(fftw_complex));
  
  for(i3=0; i3<emf->rd3-1; i3++){
    for(i2=0; i2<emf->n2fft; i2++){
      for(i1=0; i1<emf->n1fft; i1++){
	emf_kxky[i1+emf->n1fft*i2] = emf_kxkyz0[i1+emf->n1fft*i2]*emf->sE12kxky[i3][i2][i1];
      }
    }
    fftw_execute(ifft_airwave);
    for(i2=0; i2<emf->n2pad; i2++){
      for(i1=0; i1<emf->n1pad; i1++){
	//we store extrapolated values in the unused part of memory variables
	emf->memD3E2[i3][i2][i1] = creal(emf_kxky[i1+emf->n1fft*i2]/(emf->n1fft*emf->n2fft));
      }
    }
  }
}


/*-------------------------------------------------------------*/
void dtft_emf_init(emf_t *emf)
{
  emf->fwd_E1 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
  emf->fwd_E2 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
  emf->fwd_E3 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
  emf->fwd_H1 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
  emf->fwd_H2 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
  emf->fwd_H3 = alloc4complexf(emf->n1pad, emf->n2pad, emf->n3pad, emf->nfreq);
  
  memset(&emf->fwd_E1[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
  memset(&emf->fwd_E2[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
  memset(&emf->fwd_E3[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
  memset(&emf->fwd_H1[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
  memset(&emf->fwd_H2[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
  memset(&emf->fwd_H3[0][0][0][0], 0, emf->nfreq*emf->n123pad*sizeof(float _Complex));
}

void dtft_emf_close(emf_t *emf)
{
  free4complexf(emf->fwd_E1);
  free4complexf(emf->fwd_E2);
  free4complexf(emf->fwd_E3);
  free4complexf(emf->fwd_H1);
  free4complexf(emf->fwd_H2);
  free4complexf(emf->fwd_H3);
}

/*-----------------------------------------------------------*/
void dtft_emf(emf_t *emf, int it)
{
  int i1, i2, i3, ifreq;
  float _Complex omegap, factor;
  
  for(ifreq=0; ifreq<emf->nfreq; ++ifreq){
    omegap = (1.0+I)*sqrtf(emf->omega0*emf->omegas[ifreq]);
    factor = cexp(I*omegap*(it+0.5)*emf->dt);

#ifdef _OPENMP
#pragma omp parallel for default(none)		\
  schedule(static)				\
  private(i1, i2, i3)				\
  shared(ifreq, factor, emf)
#endif
    for(i3=0; i3<emf->n3pad-emf->nb+emf->rd3; ++i3){
      for(i2=emf->nb-emf->rd2; i2<emf->n2pad-emf->nb+emf->rd2; ++i2){
	for(i1=emf->nb-emf->rd1; i1<emf->n1pad-emf->nb+emf->rd1; ++i1){
	  emf->fwd_E1[ifreq][i3][i2][i1] += emf->E1[i3][i2][i1]*factor;
	  emf->fwd_E2[ifreq][i3][i2][i1] += emf->E2[i3][i2][i1]*factor;
	  emf->fwd_E3[ifreq][i3][i2][i1] += emf->E3[i3][i2][i1]*factor;
	}
      }
    }

  }/* end for ifreq */
}

void compute_green_function(emf_t *emf)
{
  int ifreq,it,i1,i2,i3;
  float _Complex omegap, src_fd;

  for(ifreq=0; ifreq<emf->nfreq; ++ifreq){
    /*------------------- DTFT over omega'-------------------------*/
    omegap = (1.0+I)*sqrtf(emf->omega0*emf->omegas[ifreq]);/* omega' in fictitous wave domain */
    src_fd=0.;
    for(it=0; it<emf->nt; ++it)  src_fd += emf->stf[it]*cexp(I*omegap*it*emf->dt);//J'
    src_fd /= csqrt(-I*0.5*emf->omegas[ifreq]/emf->omega0);//J<--J'

    for(i3=0; i3<emf->n3pad-emf->nb; ++i3){
      for(i2=emf->nb; i2<emf->n2pad-emf->nb; ++i2){
	for(i1=emf->nb; i1<emf->n1pad-emf->nb; ++i1){
	  emf->fwd_E1[ifreq][i3][i2][i1] /= src_fd;
	  emf->fwd_E2[ifreq][i3][i2][i1] /= src_fd;
	  emf->fwd_E3[ifreq][i3][i2][i1] /= src_fd;
	}
      }
    }
  }
}


void extract_emf(acqui_t *acqui, emf_t *emf, interp_t *interp_rg, interp_t *interp_sg)
/*< extract data from EM field by interpolation >*/
{   
  int ic,irec,isub,ifreq,i1,i2,i3,ix1,ix2,ix3,i1_,i2_,i3_;
  float w1,w2,w3;
  float _Complex s;

  for(ifreq=0; ifreq < emf->nfreq; ++ifreq){

    for(ic=0; ic<emf->nchrec; ++ic){
      for(irec = 0; irec<acqui->nrec; irec++) {
	for(isub=0; isub<acqui->nsubrec; isub++){
	
	  if(strcmp(emf->chrec[ic],"Ex") == 0){	    
	    /* staggered grid: E1[i1,i2,i3] = Ex[i1+0.5,i2,i3] */
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
	    emf->dcal_fd[ic][ifreq][irec] = s;
	  }
	}/* end for isub */
      }/* end for irec */
    }/* end for ic */

  }/* end for ifreq */

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



int check_convergence(emf_t *emf)
{
  static float _Complex old[8], new[8];
  static int first = 1;
  int j, k;
  
  k = 0;
  if(first){
    old[0] = emf->fwd_E1[0][0][emf->nb][emf->nb];
    old[1] = emf->fwd_E1[0][0][emf->nb][emf->nb+emf->n1-1];
    old[2] = emf->fwd_E1[0][0][emf->nb+emf->n2-1][emf->nb];
    old[3] = emf->fwd_E1[0][0][emf->nb+emf->n2-1][emf->nb+emf->n1-1];
    old[4] = emf->fwd_E1[0][emf->n3-1][emf->nb][emf->nb];
    old[5] = emf->fwd_E1[0][emf->n3-1][emf->nb][emf->nb+emf->n1-1];
    old[6] = emf->fwd_E1[0][emf->n3-1][emf->nb+emf->n2-1][emf->nb];
    old[7] = emf->fwd_E1[0][emf->n3-1][emf->nb+emf->n2-1][emf->nb+emf->n1-1];
    first = 0;
  }else{
    new[0] = emf->fwd_E1[0][0][emf->nb][emf->nb];
    new[1] = emf->fwd_E1[0][0][emf->nb][emf->nb+emf->n1-1];
    new[2] = emf->fwd_E1[0][0][emf->nb+emf->n2-1][emf->nb];
    new[3] = emf->fwd_E1[0][0][emf->nb+emf->n2-1][emf->nb+emf->n1-1];
    new[4] = emf->fwd_E1[0][emf->n3-1][emf->nb][emf->nb];
    new[5] = emf->fwd_E1[0][emf->n3-1][emf->nb][emf->nb+emf->n1-1];
    new[6] = emf->fwd_E1[0][emf->n3-1][emf->nb+emf->n2-1][emf->nb];
    new[7] = emf->fwd_E1[0][emf->n3-1][emf->nb+emf->n2-1][emf->nb+emf->n1-1];

    for(j=0; j<8; j++){
      if(cabs(new[j])>0 && cabs(new[j]-old[j])<1e-4*cabs(new[j])) k++;
      old[j] = new[j];
    }
  }
  return k;
}


int main(int argc, char* argv[])
{
  int it;
  acqui_t *acqui;
  emf_t *emf;
  interp_t *interp_rg, *interp_sg;
  char fname[sizeof("emf_0000.txt")];

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
  interpolation_init(acqui, emf, interp_rg, interp_sg);
  interpolation_weights(acqui, emf, interp_rg, interp_sg);

  sanity_check(emf);

  emf->stf = alloc1float(emf->nt);
  memset(emf->stf, 0, emf->nt*sizeof(float));
  emf->stf[0] = 1.;
  
  emf->dcal_fd = alloc3complexf(acqui->nrec, emf->nfreq, emf->nchrec);
  memset(&emf->dcal_fd[0][0][0], 0, acqui->nrec*emf->nchrec*emf->nfreq*sizeof(float _Complex));

  extend_model_init(emf);
  fdtd_init(emf);
  fdtd_null(emf);
  airwave_bc_init(emf);
  dtft_emf_init(emf);
  
  for(it=0; it<emf->nt; it++){
    if(it%50==0) printf("it-----%d\n", it);

    fdtd_curlH(emf, it);
    inject_electric_src_fwd(acqui, emf, interp_rg, interp_sg, it);
    fdtd_update_E(emf, it);
    airwave_bc_update_E(emf);
    
    dtft_emf(emf, it);
    fdtd_curlE(emf, it); 
    fdtd_update_H(emf, it); 
    airwave_bc_update_H(emf);
    
    if(it%100==0){/* convergence check */
      emf->ncorner = check_convergence(emf);
      if(iproc==0) printf("%d corners of the cube converged!\n", emf->ncorner);
      if(emf->ncorner==8) break;/* all 8 corners converged, exit now */
    }    
  }
  compute_green_function(emf);
  extract_emf(acqui, emf, interp_rg, interp_sg);

  sprintf(fname, "emf_%04d.txt", acqui->shot_idx[iproc]);
  write_data(acqui, emf, fname);

  extend_model_close(emf);
  fdtd_close(emf);
  airwave_bc_close(emf);
  dtft_emf_close(emf);

  free1float(emf->stf);
  free3complexf(emf->dcal_fd);

  acqui_close(acqui);
  emf_close(emf);
  interpolation_close(emf, interp_rg, interp_sg);
  
  free(acqui);
  free(emf);
  free(interp_rg);
  free(interp_sg);
  
  MPI_Finalize();
  return 0;

}
