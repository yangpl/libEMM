/* extract EM fields using interpolation
 *--------------------------------------------------------------------
 *
 *   Copyright (c) 2020-2022, Harbin Institute of Technology, China
 *   Author: Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *--------------------------------------------------------------------*/
#include "cstd.h"
#include "acqui.h"
#include "emf.h"
#include "interp.h"

void extract_emf(acqui_t *acqui, emf_t *emf, interp_t *interp_rg, interp_t *interp_sg)
/*< extract data from EM field by interpolation >*/
{   
  int ic,irec,isub,ifreq,i1,i2,i3,ix1,ix2,ix3,i1_,i2_,i3_;
  float w1,w2,w3,t;
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
	    for(i3=-emf->rd+1; i3<=emf->rd; i3++){
	      w3 = interp_rg->rec_w3[irec][isub][i3+emf->rd-1];
	      i3_ = ix3+i3;
	      for(i2=-emf->rd+1; i2<=emf->rd; i2++){
		w2 = interp_rg->rec_w2[irec][isub][i2+emf->rd-1];
		i2_ = ix2+i2;
		for(i1=-emf->rd+1; i1<=emf->rd; i1++){
		  w1 = interp_sg->rec_w1[irec][isub][i1+emf->rd-1];
		  i1_ = ix1+i1;
		  /*interpolate over continuous field Ex */
		  s += emf->fwd_E1[ifreq][i3_][i2_][i1_]*w1*w2*w3;

		}
	      }
	    }
	    emf->dcal_fd[ic][ifreq][irec] = s;

	  }else if(strcmp(emf->chrec[ic],"Ey") == 0){
	    /* staggered grid: E2[i1,i2,i3] = Ey[i1,i2+0.5,i3] */
	    s = 0.;
	    ix1 = interp_rg->rec_i1[irec][isub];
	    ix2 = interp_sg->rec_i2[irec][isub];
	    ix3 = interp_rg->rec_i3[irec][isub];
	    for(i3=-emf->rd+1; i3<=emf->rd; i3++){
	      w3 = interp_rg->rec_w3[irec][isub][i3+emf->rd-1];
	      i3_ = ix3+i3;
	      for(i2=-emf->rd+1; i2<=emf->rd; i2++){
		w2 = interp_sg->rec_w2[irec][isub][i2+emf->rd-1];
		i2_ = ix2+i2;
		for(i1=-emf->rd+1; i1<=emf->rd; i1++){
		  w1 = interp_rg->rec_w1[irec][isub][i1+emf->rd-1];
		  i1_ = ix1+i1;
		  /*interpolate over continuous field Ey */
		  s += emf->fwd_E2[ifreq][i3_][i2_][i1_]*w1*w2*w3;
		}
	      }
	    }
	    emf->dcal_fd[ic][ifreq][irec] = s;
	  }else if(strcmp(emf->chrec[ic],"Ez") == 0){
	    /* staggered grid: E3[i1,i2,i3] = Ez[i1,i2,i3+0.5] */
	    s = 0.;
	    t = 0.;
	    ix1 = interp_rg->rec_i1[irec][isub];
	    ix2 = interp_rg->rec_i2[irec][isub];
	    ix3 = interp_sg->rec_i3[irec][isub];
	    for(i3=-emf->rd+1; i3<=emf->rd; i3++){
	      w3 = interp_sg->rec_w3[irec][isub][i3+emf->rd-1];
	      i3_ = ix3+i3;
	      for(i2=-emf->rd+1; i2<=emf->rd; i2++){
		w2 = interp_rg->rec_w2[irec][isub][i2+emf->rd-1];
		i2_ = ix2+i2;
		for(i1=-emf->rd+1; i1<=emf->rd; i1++){
		  w1 = interp_rg->rec_w1[irec][isub][i1+emf->rd-1];
		  i1_ = ix1+i1;
		  /*interpolate over continuous field Jz=sigma_zz*Ez */
		  s += emf->fwd_E3[ifreq][i3_][i2_][i1_]/emf->rho33[i3_][i2_][i1_]*w1*w2*w3;
		  t += 1./emf->rho33[i3_][i2_][i1_]*w1*w2*w3;
		}
	      }
	    }
	    emf->dcal_fd[ic][ifreq][irec] = s/t;
	  }else if(strcmp(emf->chrec[ic],"Hx") == 0){
	    /* staggered grid: H1[i1,i2,i3] = Hx[i1,i2+0.5,i3+0.5] */
	    s = 0.;
	    ix1 = interp_rg->rec_i1[irec][isub];
	    ix2 = interp_sg->rec_i2[irec][isub];
	    ix3 = interp_sg->rec_i3[irec][isub];
	    for(i3=-emf->rd+1; i3<=emf->rd; i3++){
	      w3 = interp_sg->rec_w3[irec][isub][i3+emf->rd-1];
	      i3_ = ix3+i3;
	      for(i2=-emf->rd+1; i2<=emf->rd; i2++){
		w2 = interp_sg->rec_w2[irec][isub][i2+emf->rd-1];
		i2_ = ix2+i2;
		for(i1=-emf->rd+1; i1<=emf->rd; i1++){
		  w1 = interp_rg->rec_w1[irec][isub][i1+emf->rd-1];
		  i1_ = ix1+i1;

		  s += emf->fwd_H1[ifreq][i3_][i2_][i1_]*w1*w2*w3;
		}
	      }
	    }
	    s /= csqrt(-I*0.5*emf->omegas[ifreq]/emf->omega0);  /* then s=H(omega)<-H(omega') */
	    emf->dcal_fd[ic][ifreq][irec] = s;
	  }else if(strcmp(emf->chrec[ic],"Hy") == 0){
	    /* staggered grid: H2[i1,i2,i3] = Hy[i1+0.5,i2,i3+0.5] */
	    s = 0.;
	    ix1 = interp_sg->rec_i1[irec][isub];
	    ix2 = interp_rg->rec_i2[irec][isub];
	    ix3 = interp_sg->rec_i3[irec][isub];
	    for(i3=-emf->rd+1; i3<=emf->rd; i3++){
	      w3 = interp_sg->rec_w3[irec][isub][i3+emf->rd-1];
	      i3_ = ix3+i3;
	      for(i2=-emf->rd+1; i2<=emf->rd; i2++){
		w2 = interp_rg->rec_w2[irec][isub][i2+emf->rd-1];
		i2_ = ix2+i2;
		for(i1=-emf->rd+1; i1<=emf->rd; i1++){
		  w1 = interp_sg->rec_w1[irec][isub][i1+emf->rd-1];
		  i1_ = ix1+i1;

		  s += emf->fwd_H2[ifreq][i3_][i2_][i1_]*w1*w2*w3;
		}
	      }
	    }
	    s /= csqrt(-I*0.5*emf->omegas[ifreq]/emf->omega0);  /* then s=H(omega)<-H(omega') */
	    emf->dcal_fd[ic][ifreq][irec] = s;	  
	  }else if(strcmp(emf->chrec[ic],"Hz") == 0){
	    /* staggered grid: H3[i1,i2,i3] = Hz[i1+0.5,i2+0.5,i3] */
	    s = 0.;
	    ix1 = interp_sg->rec_i1[irec][isub];
	    ix2 = interp_sg->rec_i2[irec][isub];
	    ix3 = interp_rg->rec_i3[irec][isub];
	    for(i3=-emf->rd+1; i3<=emf->rd; i3++){
	      w3 = interp_rg->rec_w3[irec][isub][i3+emf->rd-1];
	      i3_ = ix3+i3;	      
	      for(i2=-emf->rd+1; i2<=emf->rd; i2++){
		w2 = interp_sg->rec_w2[irec][isub][i2+emf->rd-1];
		i2_ = ix2+i2;
		for(i1=-emf->rd+1; i1<=emf->rd; i1++){
		  w1 = interp_sg->rec_w1[irec][isub][i1+emf->rd-1];
		  i1_ = ix1+i1;

		  s += emf->fwd_H3[ifreq][i3_][i2_][i1_]*w1*w2*w3;
		}
	      }
	    }
	    s /= csqrt(-I*0.5*emf->omegas[ifreq]/emf->omega0);  /* then s=H(omega)<-H(omega') */
	    emf->dcal_fd[ic][ifreq][irec] = s;
	  }/* end if */

	}/* end for isub */
      }/* end for irec */
    }/* end for ic */

  }/* end for ifreq */

}

