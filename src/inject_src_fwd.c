/* inject electric and magnetic sources in forward modeling
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
	  for(i3=-emf->rd+1; i3<=emf->rd; i3++){
	    w3 = interp_rg->src_w3[isrc][isub][i3+emf->rd-1];
	    i3_ = ix3+i3;		
	    for(i2=-emf->rd+1; i2<=emf->rd; i2++){
	      w2 = interp_rg->src_w2[isrc][isub][i2+emf->rd-1];
	      i2_ = ix2+i2;
	      for(i1=-emf->rd+1; i1<=emf->rd; i1++){
		w1 = interp_sg->src_w1[isrc][isub][i1+emf->rd-1];
		i1_ = ix1+i1;
		//printf("ix1=%d w1=%g i1_=%d\n", ix1-emf->nbe, w1, i1);
	  
		emf->curlH1[i3_][i2_][i1_] -= s*w1*w2*w3;
	      }/* end for i1 */
	    }/* end for i2 */
	  }/* end for i3 */
	}else if(strcmp(emf->chsrc[ic], "Ey") == 0){
	  /* staggered grid: E2[i1, i2, i3] = Ey[i1, i2+0.5, i3] */
	  ix1 = interp_rg->src_i1[isrc][isub];
	  ix2 = interp_sg->src_i2[isrc][isub];
	  ix3 = interp_rg->src_i3[isrc][isub];
	  for(i3=-emf->rd+1; i3<=emf->rd; i3++){
	    w3 = interp_rg->src_w3[isrc][isub][i3+emf->rd-1];
	    i3_ = ix3+i3;
	    for(i2=-emf->rd+1; i2<=emf->rd; i2++){
	      w2 = interp_sg->src_w2[isrc][isub][i2+emf->rd-1];
	      i2_ = ix2+i2;
	      for(i1=-emf->rd+1; i1<=emf->rd; i1++){
		w1 = interp_rg->src_w1[isrc][isub][i1+emf->rd-1];
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
	  for(i3=-emf->rd+1; i3<=emf->rd; i3++){
	    w3 = interp_sg->src_w3[isrc][isub][i3+emf->rd-1];
	    i3_ = ix3+i3;
	    for(i2=-emf->rd+1; i2<=emf->rd; i2++){
	      w2 = interp_rg->src_w2[isrc][isub][i2+emf->rd-1];
	      i2_ = ix2+i2;
	      for(i1=-emf->rd+1; i1<=emf->rd; i1++){
		w1 = interp_rg->src_w1[isrc][isub][i1+emf->rd-1];
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


void inject_magnetic_src_fwd(acqui_t *acqui, emf_t *emf, interp_t *interp_rg, interp_t *interp_sg, int it)
/*< inject a source time function into EM field >*/
{
  int ic,isrc,isub,i1,i2,i3,ix1,ix2,ix3,i1_,i2_,i3_;
  float w1,w2,w3,s,t,d3;

  s = emf->stf[it];/* source normalized by volume */
  s /= (float)acqui->nsubsrc; /*since one source is distributed to many points */
  for(isrc=0; isrc<acqui->nsrc; isrc++){
    for(isub=0; isub<acqui->nsubsrc; isub++){

      for(ic=0; ic<emf->nchsrc; ++ic){

	if(strcmp(emf->chsrc[ic],"Hx") == 0){
	  /* staggered grid: H1[i1,i2,i3] = Hx[i1,i2+0.5,i3+0.5] */
	  ix1 = interp_rg->src_i1[isrc][isub];
	  ix2 = interp_sg->src_i2[isrc][isub];
	  ix3 = interp_sg->src_i3[isrc][isub];
	  d3 = (emf->nugrid)?(emf->x3n[ix3+1]-emf->x3n[ix3]):emf->d3;
	  t = s/(emf->d1*emf->d2*d3);/* normalized by volume */
	  for(i3=-emf->rd+1; i3<=emf->rd; i3++){
	    w3 = interp_sg->src_w3[isrc][isub][i3+emf->rd-1];
	    i3_ = ix3+i3;
	    for(i2=-emf->rd+1; i2<=emf->rd; i2++){
	      w2 = interp_sg->src_w2[isrc][isub][i2+emf->rd-1];
	      i2_ = ix2+i2;
	      for(i1=-emf->rd+1; i1<=emf->rd; i1++){
		w1 = interp_rg->src_w1[isrc][isub][i1+emf->rd-1];
		i1_ = ix1+i1;
		emf->curlE1[i3_][i2_][i1_] += t*w1*w2*w3;
	      }
	    }
	  }
	}else if(strcmp(emf->chsrc[ic],"Hy") == 0){
	  /* staggered grid: H2[i1,i2,i3] = Hy[i1+0.5,i2,i3+0.5] */
	  ix1 = interp_sg->src_i1[isrc][isub];
	  ix2 = interp_rg->src_i2[isrc][isub];
	  ix3 = interp_sg->src_i3[isrc][isub];
	  d3 = (emf->nugrid)?(emf->x3n[ix3+1]-emf->x3n[ix3]):emf->d3;
	  t = s/(emf->d1*emf->d2*d3);/* normalized by volume */
	  for(i3=-emf->rd+1; i3<=emf->rd; i3++){
	    w3 = interp_sg->src_w3[isrc][isub][i3+emf->rd-1];
	    i3_ = ix3+i3;
	    for(i2=-emf->rd+1; i2<=emf->rd; i2++){
	      w2 = interp_rg->src_w2[isrc][isub][i2+emf->rd-1];
	      i2_ = ix2+i2;
	      for(i1=-emf->rd+1; i1<=emf->rd; i1++){
		w1 = interp_sg->src_w1[isrc][isub][i1+emf->rd-1];
		i1_ = ix1+i1;

		emf->curlE2[i3_][i2_][i1_] += t*w1*w2*w3;
	      }
	    }
	  }
	}else if(strcmp(emf->chsrc[ic],"Hz") == 0){
	  /* staggered grid: H3[i1,i2,i3] = Hz[i1+0.5,i2+0.5,i3] */
	  ix1 = interp_sg->src_i1[isrc][isub];
	  ix2 = interp_sg->src_i2[isrc][isub];
	  ix3 = interp_rg->src_i3[isrc][isub];
	  d3 = (emf->nugrid)?(emf->x3s[ix3]-emf->x3s[ix3-1]):emf->d3;
	  t = s/(emf->d1*emf->d2*d3);/* normalized by volume */
	  for(i3=-emf->rd+1; i3<=emf->rd; i3++){
	    w3 = interp_rg->src_w3[isrc][isub][i3+emf->rd-1];
	    i3_ = ix3+i3;
	    for(i2=-emf->rd+1; i2<=emf->rd; i2++){
	      w2 = interp_sg->src_w2[isrc][isub][i2+emf->rd-1];
	      i2_ = ix2+i2;
	      for(i1=-emf->rd+1; i1<=emf->rd; i1++){
		w1 = interp_sg->src_w1[isrc][isub][i1+emf->rd-1];
		i1_ = ix1+i1;

		emf->curlE3[i3_][i2_][i1_] += t*w1*w2*w3;
	      }
	    }
	  }
	}
      } /* end for ic */
    }/* end for isub */
  }/* end for isrc */

}

