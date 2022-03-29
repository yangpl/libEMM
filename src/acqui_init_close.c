/* read acquisition file for source and receiver geometry
 *--------------------------------------------------------------------
 *
 *   Copyright (c) 2020, Harbin Institute of Technology, China
 *   Author: Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *--------------------------------------------------------------------*/
#include <mpi.h>
#include "cstd.h"
#include "acqui.h"
#include "emf.h"
#include "mpi_info.h"

void acqui_init(acqui_t *acqui, emf_t * emf)
/*< read acquisition file to initialize survey geometry >*/
{
  static int nd = 5000;//maximum dimensions for the number of source and receiver
  float src_x1[nd], src_x2[nd], src_x3[nd], src_hd[nd], src_pit[nd];/* source receiver coordinates */
  float rec_x1[nd], rec_x2[nd], rec_x3[nd], rec_hd[nd], rec_pit[nd];/* source receiver coordinates */
  int rec_idx[nd];/* reciver index associated with current processor */
  float x, y, z, hd, pit;
  float x1min, x1max, x2min, x2max, x3min, x3max;
  int isrc, irec, iseof, idx, i, k, nsrc;
  char *fsrc, *frec, *fsrcrec;
  FILE *fp=NULL;

  if(!(getparstring("fsrc", &fsrc))) err("Need fsrc= ");
  /* file to specify all possible source locations */
  if(!(getparstring("frec", &frec))) err("Need frec= ");
  /* file to specify all possible receiver locations */
  if(!(getparstring("fsrcrec", &fsrcrec))) err("Need fsrcrec= ");
  /* file to specify how source and receiver are combined */

  if(!getparfloat("x1min", &acqui->x1min)) acqui->x1min=0;
  /* minimum limit of the survey in x direction */
  if(!getparfloat("x1max", &acqui->x1max)) acqui->x1max=acqui->x1min+(emf->n1-1)*emf->d1;
  /* maximum limit of the survey in x direction */
  if(!getparfloat("x2min", &acqui->x2min)) acqui->x2min=0;
  /* minimum limit of the survey in y direction */
  if(!getparfloat("x2max", &acqui->x2max)) acqui->x2max=acqui->x2min+(emf->n2-1)*emf->d2;
  /* maximum limit of the survey in y direction */
  if(!getparfloat("x3min", &acqui->x3min)) acqui->x3min=0;
  /* minimum limit of the survey in z direction */
  if(!getparfloat("x3max", &acqui->x3max)) acqui->x3max=acqui->x3min+(emf->n3-1)*emf->d3;
  /* maximum limit of the survey in z direction */
  if(!getparint("nsubsrc", &acqui->nsubsrc)) acqui->nsubsrc=1;
  /* number of subpoints to represent one source location */
  if(!getparint("nsubrec", &acqui->nsubrec)) acqui->nsubrec=1;
  /* number of subpoints to represent one receiver location */
  if(!getparfloat("lensrc", &acqui->lensrc)) acqui->lensrc=300.;
  /* length of the source antenna, default=1 m */
  if(!getparfloat("lenrec", &acqui->lenrec)) acqui->lenrec=8.;
  /* length of the receiver antenna, default=8 m */
  if(fabs(acqui->x1max-acqui->x1min-emf->d1*(emf->n1-1))>1e-15)
    err("inconsistent input: x1max-x1min!=d1*(n1-1)");
  if(fabs(acqui->x2max-acqui->x2min-emf->d2*(emf->n2-1))>1e-15)
    err("inconsistent input: x2max-x2min!=d2*(n2-1)");

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
  //find the receiver indices associated with source-idx
  fp = fopen(fsrcrec,"r");
  if(fp==NULL) err("file does not exist!");
  iseof = fscanf(fp, "%*[^\n]\n");//skip a line at the beginning of the file
  i = 0;
  while(1){
    /* (northing,easting,depth)=(y,x,z);   azimuth = heading;  dip=pitch */
    iseof = fscanf(fp,"%d %d", &isrc, &irec);
    if(iseof==EOF)
      break;
    else{
      if(isrc==idx) {//isrc-th source gather
	rec_idx[i] = irec;//the global receiver index associated with current source
	i++;
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
  iseof = fscanf(fp, "%*[^\n]\n");//skip a line at the beginning of the file
  isrc = 0;
  while(1){
    /* (northing,easting,depth)=(y,x,z);   azimuth = heading;  dip=pitch */
    iseof = fscanf(fp,"%f %f %f %f %f %d", &x, &y, &z, &hd, &pit, &idx);
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
  iseof = fscanf(fp, "%*[^\n]\n");//skip a line at the beginning of the file
  irec = 0;
  while(1){
    /* (northing,easting,depth)=(y,x,z);   azimuth = heading;  dip=pitch */
    iseof = fscanf(fp,"%f %f %f %f %f %d", &x, &y, &z, &hd, &pit, &idx);
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
    acqui->src_x1[isrc] = src_x1[i];
    acqui->src_x2[isrc] = src_x2[i];
    acqui->src_x3[isrc] = src_x3[i];
    acqui->src_azimuth[isrc] = src_hd[i];
    acqui->src_dip[isrc] = src_pit[i];
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
    acqui->rec_x1[irec] = rec_x1[i];
    acqui->rec_x2[irec] = rec_x2[i];
    acqui->rec_x3[irec] = rec_x3[i];
    acqui->rec_azimuth[irec] = rec_hd[i];
    acqui->rec_dip[irec] = rec_pit[i];
  }/* end for irec */
  ierr = MPI_Gather(acqui->src_x1, 1, MPI_FLOAT, src_x1, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  ierr = MPI_Gather(acqui->src_x2, 1, MPI_FLOAT, src_x2, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  ierr = MPI_Gather(acqui->src_x3, 1, MPI_FLOAT, src_x3, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  if(emf->verb){
    printf("[x1min,x1max]=[%g, %g]\n", acqui->x1min, acqui->x1max);
    printf("[x2min,x2max]=[%g, %g]\n", acqui->x2min, acqui->x2max);
    printf("[x3min,x3max]=[%g, %g]\n", acqui->x3min, acqui->x3max);
    printf("nsrc_total=%d\n", acqui->nsrc_total);
    printf("nrec_total=%d\n", acqui->nrec_total);
    for(k=0; k<nproc; k++)
      printf("isrc=%d (x,y,z)=(%.2f, %.2f, %.2f)\n", acqui->shot_idx[k], src_x1[k], src_x2[k], src_x3[k]);
  }
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

  free1int(acqui->shot_idx);
}

