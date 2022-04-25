/* initialize the parameters for EM fields (emf) to do CSEM modeling
 *--------------------------------------------------------------------
 *
 *   Copyright (c) 2020-2022, Harbin Institute of Technology, China
 *   Author: Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *--------------------------------------------------------------------*/
#include <assert.h>
#include "cstd.h"
#include "emf.h"
#include "acqui.h"
#include "constants.h"

int cmpfunc(const void *a, const void *b) { return ( *(int*)a - *(int*)b ); }

void nugrid_init(emf_t *emf);
void nugrid_close(emf_t *emf);

void emf_init(emf_t *emf)
{
  char *frho11, *frho22, *frho33; /* resistivity files */
  FILE *fp;
  int ifreq, ic, istat;

  if(!getparint("verb", &emf->verb)) emf->verb=1; 
  /* emf->verbose=1 only on master process */
  if(!getparint("n1", &emf->n1)) emf->n1=100; 
  /* number of cells in axis-1, nx */
  if(!getparint("n2", &emf->n2)) emf->n2=100; 
  /* number of cells in axis-2, ny */
  if(!getparint("n3", &emf->n3)) emf->n3=50; 
  /* number of cells in axis-3, nz */
  if(!getparfloat("d1", &emf->d1)) emf->d1=100.;
  /* grid spacing in 1st dimension, dx */
  if(!getparfloat("d2", &emf->d2)) emf->d2=100.;
  /* grid spacing in 2nd dimension, dy */
  if(!getparfloat("d3", &emf->d3)) emf->d3=100.;
  /* grid spacing in 3rd dimension, dz */
  if(!getparint("airwave", &emf->airwave)) emf->airwave=1; 
  /* 1=emf->modeling with air-water interface */
  if(!getparint("nb", &emf->nb)) emf->nb=10; 
  /* number of PML layers on each side */
  if(!getparint("rd", &emf->rd)) emf->rd=2;
  /* emf->rd of Bessel I0 function for sinc */
  if(!getparint("ne", &emf->ne)) emf->ne = 6;
  /* number of extra layers between inner emf->model and PML */
  assert(emf->n1>0);
  assert(emf->n2>0);
  assert(emf->n3>0);
  assert(emf->nb>0);
  assert(emf->d1>0);
  assert(emf->d2>0);
  assert(emf->d3>0);
  assert(emf->ne>=emf->rd);//make sure ne>=rd
  
  emf->n123 = emf->n1*emf->n2*emf->n3;
  emf->nbe = emf->nb+emf->ne;/* number of PML layers + extra 2 points due to 4-th order FD */
  emf->n1pad = emf->n1+2*emf->nbe;/* total number of grid points after padding PML+extra */
  emf->n2pad = emf->n2+2*emf->nbe;/* total number of grid points after padding PML+extra */
  emf->n3pad = emf->n3+2*emf->nbe;/* total number of grid points after padding PML+extra */
  emf->n123pad = emf->n1pad*emf->n2pad*emf->n3pad;
  if(emf->verb){
    printf("Half operator length: rd=%d\n", emf->rd);
    printf("PML layers: nb=%d\n", emf->nb);
    printf("Layers outside model: nbe=%d\n", emf->nbe);
    printf("Model size:   [n1, n2, n3]=[%d, %d, %d]\n", emf->n1, emf->n2, emf->n3);
    printf("Grid spacing: [d1, d2, d3]=[%g, %g, %g]\n", emf->d1, emf->d2, emf->d3);
    printf("[n1pad, n2pad, n3pad]=[%d, %d, %d]\n", emf->n1pad, emf->n2pad, emf->n3pad);
  }
  
  if(!(emf->nfreq=countparval("freqs"))) err("Need freqs= vector");
  /* number of frequencies for electromagnetic emf->modeling */
  emf->freqs=alloc1float(emf->nfreq);
  emf->omegas=alloc1float(emf->nfreq);
  getparfloat("freqs", emf->freqs);/* a list of frequencies separated by comma */
  qsort(emf->freqs, emf->nfreq, sizeof(float), cmpfunc);/*sort frequencies in ascending order*/
  for(ifreq=0; ifreq<emf->nfreq; ++ifreq) {
    assert(emf->freqs[ifreq]>0);//all frequencies must be positive values
    emf->omegas[ifreq]=2.*PI*emf->freqs[ifreq];
    if(emf->verb) printf("freq[%d]=%g Hz\n", ifreq+1, emf->freqs[ifreq]);
  }

  if(!getparfloat("f0", &emf->f0)) emf->f0=0.5; /* frequency to determine emf->omega0 */
  emf->omega0 = 2.0*PI*emf->f0;/*wavespeed=1/sqrt(mu*epsilon)*/
  if(!getparfloat("Glim", &emf->Glim)) emf->Glim=10; /* frequency to determine emf->omega0 */

  
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
    emf->nchrec=1;
    emf->chrec=(char**)alloc1(emf->nchrec, sizeof(void*));
    emf->chrec[0]="Ex";
  }
  if(emf->verb){
    printf("Active source channels:");
    for(ic=0; ic<emf->nchsrc; ++ic) printf(" %s", emf->chsrc[ic]);
    printf("\n");
    printf("Active recever channels:");
    for(ic=0; ic<emf->nchrec; ++ic) printf(" %s", emf->chrec[ic]);
    printf("\n");
  }
  

  /*-------------------------------------------------------*/
  if(!(getparstring("frho11", &frho11))) err("Need frho11= ");
  if(!(getparstring("frho22", &frho22))) err("Need frho22= ");
  if(!(getparstring("frho33", &frho33))) err("Need frho33= ");
  emf->rho11 = alloc3float(emf->n1, emf->n2, emf->n3);
  emf->rho22 = alloc3float(emf->n1, emf->n2, emf->n3);
  emf->rho33 = alloc3float(emf->n1, emf->n2, emf->n3);
  
  /* read resistivity, assuming the homogenization has been done within the inputs */
  fp = fopen(frho11, "rb");
  if(fp==NULL) err("cannot open file");
  istat = fread(emf->rho11[0][0], sizeof(float), emf->n123, fp);
  if(istat != emf->n123) err("size parameter does not match the file-rho11!");
  fclose(fp);

  fp = fopen(frho22, "rb");
  if(fp==NULL) err("cannot open file");
  istat = fread(emf->rho22[0][0], sizeof(float), emf->n123, fp);
  if(istat != emf->n123) err("size parameter does not match the file-rho22!");
  fclose(fp);

  fp = fopen(frho33, "rb");
  if(fp==NULL) err("cannot open file");
  istat = fread(emf->rho33[0][0], sizeof(float), emf->n123, fp);
  if(istat != emf->n123) err("size parameter does not match the file-rho33!");
  fclose(fp);

  if(!getparint("nugrid", &emf->nugrid)) emf->nugrid=0;/* 1=nonuniform grid; 0=uniform grid */
  if(emf->nugrid) {
    printf("nugrid=%d\n", emf->nugrid);
    nugrid_init(emf);
  }

}



void emf_close(emf_t *emf)
{
  free1float(emf->freqs);
  free1float(emf->omegas);
  free3float(emf->rho11); 
  free3float(emf->rho22);
  free3float(emf->rho33);

  if(emf->nugrid) nugrid_close(emf);
}


