/* read and write CSEM data in binary
 *--------------------------------------------------------------------
 *
 *   Copyright (c) 2020, Harbin Institute of Technology, China
 *   Author: Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *--------------------------------------------------------------------*/
#include "cstd.h"

#include "acqui.h"
#include "emf.h"
#include "mpi_info.h"

void write_data(acqui_t *acqui, emf_t *emf)
/*< write synthetic data according to shot/process index >*/
{
  FILE *fp;
  int isrc, irec, ichrec, ifreq;
  float dp_re, dp_im;

  fp=fopen("emf_0001.txt","w");
  if(fp==NULL) err("error opening file for writing");
  fprintf(fp, "iTx 	 iRx    chrec  ifreq 	 emf_real 	 emf_imag\n");
  isrc = acqui->shot_idx[iproc];
  for(ichrec=0; ichrec<emf->nchrec; ichrec++){
    for(ifreq=0; ifreq<emf->nfreq; ifreq++){
      for(irec=0; irec<acqui->nrec; irec++){
	dp_re = creal(emf->dcal_fd[ichrec][ifreq][irec]);
	dp_im = cimag(emf->dcal_fd[ichrec][ifreq][irec]);
	fprintf(fp, "%d \t %d \t %s \t %d \t %e \t %e\n",
		isrc, irec+1, emf->chrec[ichrec], ifreq+1, dp_re, dp_im);
      }
    }
  }
  fclose(fp);

}


