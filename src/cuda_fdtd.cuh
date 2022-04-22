/* CUDA kernel for 3D electromagnetic modeling using 4th order FDTD 
 *------------------------------------------------------------------------
 *
 * Copyright (c) 2021 Harbin Institute of Technology. All rights reserved.
 * Anothr: Pengliang Yang 
 * Email: ypl.2100@gmail.com
 * Homepage: https://yangpl.wordpress.com
 *-----------------------------------------------------------------------*/
#include <cuComplex.h>

#include "constants.h"

//<<<1,8>>>
__global__ void cuda_check_convergence(int *d_corner_id, cuFloatComplex *d_fwd_E1, cuFloatComplex *d_backup, int *h_ncorner)
{
  const int tid = threadIdx.x;
  *h_ncorner = 0;
  float v1x, v1y, v0x, v0y, tmp0, tmp1;
  
  if (tid < 8){
    v1x = d_fwd_E1[d_corner_id[tid]].x;
    v1y = d_fwd_E1[d_corner_id[tid]].y;
    v0x = v1x - d_backup[tid].x;
    v0y = v1y - d_backup[tid].y;
    d_backup[tid].x = v1x;
    d_backup[tid].y = v1y;

    tmp0 = v0x*v0x+v0y*v0y;
    tmp1 = v1x*v1x+v1y*v1y;
    if(tmp1>0 && tmp0<1e-6*tmp1) atomicAdd(h_ncorner, 1);//add 1 if converge
  }

}


//cuda_fdtd_update_E<<<dimg,dimb>>>()
//dimb.x = BlockSize1, dimb.y = BlockSize2
//grid.x = (n1pad+dimb.x-1)/dimb.x, grid.y = (n2pad+dimb.y-1)/dimb.y
__global__ void cuda_fdtd_curlH(float *H1, float *H2, float *H3,
				float *curlH1, float *curlH2, float *curlH3,
				float *a1, float *b1, float *a2, float *b2, float *a3, float *b3,
				float *memD1H2, float *memD1H3, float *memD2H1,
				float *memD2H3, float *memD3H1, float *memD3H2, 
				float *inveps11, float *inveps22, float *inveps33,
				float c11, float c21, float c12, float c22, float c13, float c23,
				int n1pad, int n2pad, int n3pad, int nb, int nbe, int airwave)
		
{
  const int i1 = threadIdx.x + blockDim.x*blockIdx.x;
  const int i2 = threadIdx.y + blockDim.y*blockIdx.y;
  const int t1 = threadIdx.x + 2;//t1>=2 && t1<BlockSize1+2
  const int t2 = threadIdx.y + 2;//t2>=2 && t2<BlockSize2+2

  __shared__ float sH1[BlockSize2+3][BlockSize1+3];
  __shared__ float sH2[BlockSize2+3][BlockSize1+3];
  __shared__ float sH3[BlockSize2+3][BlockSize1+3];

  int in_idx = i1+n1pad*i2;
  int out_idx = 0;
  int stride = n1pad*n2pad;

  float H1m2,H1m1,H1c,H1p1;
  float H2m2,H2m1,H2c,H2p1;
  float D1H2,D1H3,D2H1,D2H3,D3H1,D3H2;
  float tmp1,tmp2;
  int j1,k1,j2,k2,j3,k3,i3;
  int out_idx1, out_idx2, out_idx3;//index for pML memory variables

  bool validrw = false;//valid for reading from and writing into global memory
  bool validr = false;
  int i1min=2;
  int i1max=n1pad-2;
  int i2min=2;
  int i2max=n2pad-2;	
  int i3min=airwave?nbe:2;
  int i3max=n3pad-2;
  if(i1>=i1min && i1<=i1max && i2>=i2min && i2<=i2max) validrw = true;
  in_idx += stride*(i3min-2);
  if(i1<n1pad && i2<n2pad){
    validr = true;
    //slice i3min-2
    H1m1 = H1[in_idx];  
    H2m1 = H2[in_idx];  
    //slice i3min-1
    in_idx += stride;
    out_idx = in_idx;  
    H1c = H1[in_idx];  
    H2c = H2[in_idx];
    //slice i3min
    in_idx += stride;
    H1p1 = H1[in_idx];  
    H2p1 = H2[in_idx];  
  }

  for(i3=i3min; i3<=i3max; i3++){//i3-2>=0 && i3+1<n3pad
    //-------------------------------------------------------
    if(validr){
      //increase slice counter
      in_idx += stride;//slice i3min+1 when i3=i3min
      out_idx += stride;//slice i3min when i3=i3min
      //slice i3-2 (minus 2): m2<--m1
      H1m2 = H1m1;     
      H2m2 = H2m1;
      //slice i3-1 (minus 1): m1<--c
      H1m1 = H1c;
      H2m1 = H2c;
      //slice i3 (current): c<--p1
      H1c = H1p1;       
      H2c = H2p1;
      //slice i3+1 (plus 1): p1<--frontmost slice
      H1p1 = H1[in_idx]; 
      H2p1 = H2[in_idx];
    }
    __syncthreads();

    if(threadIdx.x<2){//left halo: threadIdx.x<2
      //threadIdx.x=t1-2, [t2,t1-2]=t1-2+n1pad*t2=out_idx-2
      if(blockIdx.x>0){//not first block
	sH1[t2][threadIdx.x] = H1[out_idx-2];
	sH2[t2][threadIdx.x] = H2[out_idx-2];
	sH3[t2][threadIdx.x] = H3[out_idx-2];
      }else{//first block, no left halo
	sH1[t2][threadIdx.x] = 0.0;
	sH2[t2][threadIdx.x] = 0.0;
	sH3[t2][threadIdx.x] = 0.0;
      }
    }
    if(threadIdx.x<1){//right halo: threadIdx.x+BlockSize1+2 <BlockSize1+3
      //threadIdx.x+BlockSize1+2=t1+BlockSize1, [t2,t1+BlockSize1]=t1+BlockSize1+n1pad*t2=out_idx+BlockSize1
      if(blockIdx.x<gridDim.x-1){//not last block
	sH1[t2][threadIdx.x+BlockSize1+2] = H1[out_idx+BlockSize1]; 
	sH2[t2][threadIdx.x+BlockSize1+2] = H2[out_idx+BlockSize1]; 
	sH3[t2][threadIdx.x+BlockSize1+2] = H3[out_idx+BlockSize1]; 
      }else{//last block, no right halo
	sH1[t2][threadIdx.x+BlockSize1+2] = 0.0;
	sH2[t2][threadIdx.x+BlockSize1+2] = 0.0;
	sH3[t2][threadIdx.x+BlockSize1+2] = 0.0;
      }
    }
    if(threadIdx.y<2){//top halo: threadIdx.y<2
      //threadIdx.y=t2-2, [t2-2,t1]=t1+n1pad*(t2-2)=out_idx-2*n1pad
      if(blockIdx.y>0){//not first block
	sH1[threadIdx.y][t1] = H1[out_idx-2*n1pad];
	sH2[threadIdx.y][t1] = H2[out_idx-2*n1pad];
	sH3[threadIdx.y][t1] = H3[out_idx-2*n1pad];
      }else{//first block, no top halo
	sH1[threadIdx.y][t1] = 0.0;
	sH2[threadIdx.y][t1] = 0.0;
	sH3[threadIdx.y][t1] = 0.0;
      }
    }
    if(threadIdx.y<1){//bottom halo: threadIdx.y+BlockSize2+2<BlockSize2+3
      //threadIdx.y+BlockSize2+2=t2+BlockSize2, [t2+BlockSize2,t1]=t1+n1pad*(t2+BlockSize2)=out_idx+n1pad*BlockSize2
      if(blockIdx.y<gridDim.x-1){//not last block
	sH1[threadIdx.y+BlockSize2+2][t1] = H1[out_idx+BlockSize2*n1pad];
	sH2[threadIdx.y+BlockSize2+2][t1] = H2[out_idx+BlockSize2*n1pad];
	sH3[threadIdx.y+BlockSize2+2][t1] = H3[out_idx+BlockSize2*n1pad];
      }else{//last block, no bottom halo
	sH1[threadIdx.y+BlockSize2+2][t1] = 0.0;
	sH2[threadIdx.y+BlockSize2+2][t1] = 0.0;
	sH3[threadIdx.y+BlockSize2+2][t1] = 0.0;
      }
    }
    if(validr){
      //[i3,i2,i1]: i1+n1pad*i2+n1pad*n2pad*i3=out_idx
      sH1[t2][t1] = H1c;//current slice
      sH2[t2][t1] = H2c;//current slice
      sH3[t2][t1] = H3[out_idx];//current slice
    }
    __syncthreads();

    if(validrw){
      // D2H3 = c12*(H3[i3][i2][i1]-H3[i3][i2-1][i1]) + c22*(H3[i3][i2+1][i1]-H3[i3][i2-2][i1]);
      // D3H2 = c13*(H2[i3][i2][i1]-H2[i3-1][i2][i1]) + c23*(H2[i3+1][i2][i1]-H2[i3-2][i2][i1]);
      // D3H1 = c13*(H1[i3][i2][i1]-H1[i3-1][i2][i1]) + c23*(H1[i3+1][i2][i1]-H1[i3-2][i2][i1]);
      // D1H3 = c11*(H3[i3][i2][i1]-H3[i3][i2][i1-1]) + c21*(H3[i3][i2][i1+1]-H3[i3][i2][i1-2]);
      // D1H2 = c11*(H2[i3][i2][i1]-H2[i3][i2][i1-1]) + c21*(H2[i3][i2][i1+1]-H2[i3][i2][i1-2]);
      // D2H1 = c12*(H1[i3][i2][i1]-H1[i3][i2-1][i1]) + c22*(H1[i3][i2+1][i1]-H1[i3][i2-2][i1]);
      D1H2 = c11*(sH2[t2][t1] - sH2[t2][t1-1]) + c21*(sH2[t2][t1+1] - sH2[t2][t1-2]);
      D1H3 = c11*(sH3[t2][t1] - sH3[t2][t1-1]) + c21*(sH3[t2][t1+1] - sH3[t2][t1-2]);
      D2H1 = c12*(sH1[t2][t1] - sH1[t2-1][t1]) + c22*(sH1[t2+1][t1] - sH1[t2-2][t1]);
      D2H3 = c12*(sH3[t2][t1] - sH3[t2-1][t1]) + c22*(sH3[t2+1][t1] - sH3[t2-2][t1]);
      D3H1 = c13*(H1c - H1m1)                  + c23*(H1p1 - H1m2);
      D3H2 = c13*(H2c - H2m1)                  + c23*(H2p1 - H2m2);

      //CPML: mem=memory variable
      if(i1<nb){
	//   memD1H3[i3][i2][i1] = b1[i1]*memD1H3[i3][i2][i1] + a1[i1]*D1H3;
	//   memD1H2[i3][i2][i1] = b1[i1]*memD1H2[i3][i2][i1] + a1[i1]*D1H2;
	//   D1H3 += memD1H3[i3][i2][i1];
	//   D1H2 += memD1H2[i3][i2][i1];
	out_idx1 = i1+2*nb*(i2+n2pad*i3);
	tmp1 = memD1H3[out_idx1];
	tmp2 = memD1H2[out_idx1];
	tmp1 = b1[i1]*tmp1 + a1[i1]*D1H3;
	tmp2 = b1[i1]*tmp2 + a1[i1]*D1H2;
	memD1H3[out_idx1] = tmp1;
	memD1H2[out_idx1] = tmp2;
	D1H3 += tmp1;
	D1H2 += tmp2;
      }
      if(i1>n1pad-1-nb){
	j1 = n1pad-1-i1;
	k1 = j1+nb;
	//   memD1H3[i3][i2][k1] = b1[j1]*memD1H3[i3][i2][k1] + a1[j1]*D1H3;
	//   memD1H2[i3][i2][k1] = b1[j1]*memD1H2[i3][i2][k1] + a1[j1]*D1H2;
	//   D1H3 += memD1H3[i3][i2][k1];
	//   D1H2 += memD1H2[i3][i2][k1];
	out_idx1 = k1+2*nb*(i2+n2pad*i3);//[i3,i2,k1]
	tmp1 = memD1H3[out_idx1];
	tmp2 = memD1H2[out_idx1];
	tmp1 = b1[j1]*tmp1 + a1[j1]*D1H3;
	tmp2 = b1[j1]*tmp2 + a1[j1]*D1H2;
	memD1H3[out_idx1] = tmp1;
	memD1H2[out_idx1] = tmp2;
	D1H3 += tmp1;
	D1H2 += tmp2;
      }
      if(i2<nb){
	// memD2H3[i3][i2][i1] = b2[i2]*memD2H3[i3][i2][i1] + a2[i2]*D2H3;
	// memD2H1[i3][i2][i1] = b2[i2]*memD2H1[i3][i2][i1] + a2[i2]*D2H1;
	// D2H3 += memD2H3[i3][i2][i1];
	// D2H1 += memD2H1[i3][i2][i1];
	out_idx2 = i1+n1pad*(i2+2*nb*i3);//[i3,k2,i1]
	tmp1 = memD2H3[out_idx2];
	tmp2 = memD2H1[out_idx2];
	tmp1 = b2[i2]*tmp1 + a2[i2]*D2H3;
	tmp2 = b2[i2]*tmp2 + a2[i2]*D2H1;
	memD2H3[out_idx2] = tmp1;
	memD2H1[out_idx2] = tmp2;
	D2H3 += tmp1;
	D2H1 += tmp2;
      }else if(i2>n2pad-1-nb){
	j2 = n2pad-1-i2;
	k2 = j2+nb;
	// memD2H3[i3][k2][i1] = b2[j2]*memD2H3[i3][k2][i1] + a2[j2]*D2H3;
	// memD2H1[i3][k2][i1] = b2[j2]*memD2H1[i3][k2][i1] + a2[j2]*D2H1;
	// D2H3 += memD2H3[i3][k2][i1];
	// D2H1 += memD2H1[i3][k2][i1];
	out_idx2 = i1+n1pad*(k2+2*nb*i3);//[i3,k2,i1]
	tmp1 = memD2H3[out_idx2];
	tmp2 = memD2H1[out_idx2];
	tmp1 = b2[j2]*tmp1 + a2[j2]*D2H3;
	tmp2 = b2[j2]*tmp2 + a2[j2]*D2H1;
	memD2H3[out_idx2] = tmp1;
	memD2H1[out_idx2] = tmp2;
	D2H3 += tmp1;
	D2H1 += tmp2;
      }
      if(i3<nb){
	//   memD3H2[i3][i2][i1] = b3[i3]*memD3H2[i3][i2][i1] + a3[i3]*D3H2;
	//   memD3H1[i3][i2][i1] = b3[i3]*memD3H1[i3][i2][i1] + a3[i3]*D3H1;
	//   D3H2 += memD3H2[i3][i2][i1];
	//   D3H1 += memD3H1[i3][i2][i1];
	//out_idx3 = i1+n1pad*(i2+n2pad*i3);
	out_idx3 = out_idx;
	tmp1 = memD3H2[out_idx3];
	tmp2 = memD3H1[out_idx3];
	tmp1 = b3[i3]*tmp1 + a3[i3]*D3H2;
	tmp2 = b3[i3]*tmp2 + a3[i3]*D3H1;
	memD3H2[out_idx3] = tmp1;
	memD3H1[out_idx3] = tmp2;
	D3H2 += tmp1;
	D3H1 += tmp2;
      }else if(i3>n3pad-1-nb){
	j3 = n3pad-1-i3;
	k3 = j3+nb;
	//   memD3H2[k3][i2][i1] = b3[j3]*memD3H2[k3][i2][i1] + a3[j3]*D3H2;
	//   memD3H1[k3][i2][i1] = b3[j3]*memD3H1[k3][i2][i1] + a3[j3]*D3H1;
	//   D3H2 += memD3H2[k3][i2][i1];
	//   D3H1 += memD3H1[k3][i2][i1];
	out_idx3 = i1+n1pad*(i2+n2pad*k3);//[k3,i2,i1]
	tmp1 = memD3H2[out_idx3];
	tmp2 = memD3H1[out_idx3];
	tmp1 = b3[j3]*tmp1 + a3[j3]*D3H2;
	tmp2 = b3[j3]*tmp2 + a3[j3]*D3H1;
	memD3H2[out_idx3] = tmp1;
	memD3H1[out_idx3] = tmp2;
	D3H2 += tmp1;
	D3H1 += tmp2;
      }

      curlH1[out_idx] = D2H3 - D3H2;
      curlH2[out_idx] = D3H1 - D1H3;
      curlH3[out_idx] = D1H2 - D2H1;
    }
  }//end for i3
}

//cuda_fdtd_update_H<<<dimg,dimb>>>()
//dimb.x = BlockSize1, dimb.y = BlockSize2
//grid.x = (n1pad+dimb.x-1)/dimb.x, grid.y = (n2pad+dimb.y-1)/dimb.y
__global__ void cuda_fdtd_curlE(float *E1, float *E2, float *E3,
				float *curlE1, float *curlE2, float *curlE3,
				float *a1, float *b1, float *a2, float *b2, float *a3, float *b3,
				float *memD1E2, float *memD1E3, float *memD2E1,
				float *memD2E3, float *memD3E1, float *memD3E2, 
				float c11, float c21, float c12, float c22, float c13, float c23,
				int n1pad, int n2pad, int n3pad, int nb, int nbe, int airwave)
{
  const int i1 = threadIdx.x + blockDim.x*blockIdx.x;
  const int i2 = threadIdx.y + blockDim.y*blockIdx.y;
  const int t1 = threadIdx.x + 1;
  const int t2 = threadIdx.y + 1;

  __shared__ float sE1[BlockSize2+3][BlockSize1+3];
  __shared__ float sE2[BlockSize2+3][BlockSize1+3];
  __shared__ float sE3[BlockSize2+3][BlockSize1+3];

  int in_idx = i1+n1pad*i2;
  int out_idx = 0;
  int stride = n1pad*n2pad;//n1pad+

  float E1m1,E1c,E1p1,E1p2;
  float E2m1,E2c,E2p1,E2p2;
  float D1E2,D1E3,D2E1,D2E3,D3E1,D3E2;
  float tmp1,tmp2;
  int j1,k1,j2,k2,j3,k3,i3;
  int out_idx1, out_idx2, out_idx3;//index for pML memory variables

  bool validrw = false;//true=valid for reading from and writing into global memory
  bool validr = false;
  int i1min=1;
  int i1max=n1pad-3;
  int i2min=1;
  int i2max=n2pad-3;	
  int i3min=airwave?nbe:1;
  int i3max=n3pad-3;

  if(i1>=i1min && i1<=i1max && i2>=i2min && i2<=i2max) validrw = true;
  in_idx += stride*(i3min-1);
  if(i1<n1pad && i2<n2pad){
    validr = true;
    //slice i3min-1
    out_idx = in_idx;
    E1c = E1[in_idx];  
    E2c = E2[in_idx];  
    //slice i3min
    in_idx += stride;
    E1p1 = E1[in_idx];  
    E2p1 = E2[in_idx];  
    //slice i3min+1
    in_idx += stride;
    E1p2 = E1[in_idx]; 
    E2p2 = E2[in_idx];  
  }

  
  for(i3=i3min; i3<=i3max; i3++){//i3-1>=0 && i3+2<n2pad
    //-------------------------------------------------------
    if(validr){
      //increase slice counter
      in_idx += stride;//slice i3min+2 when i3=i3min
      out_idx += stride;//slice i3min when i3=i3min
      //slice i3-1 (minus 1): m1<--c
      E1m1 = E1c;
      E2m1 = E2c;
      //slice i3 (current): c<--p1
      E1c = E1p1;       
      E2c = E2p1;
      //slice i3+1 (plus 1): p1<--p2
      E1p1 = E1p2;     
      E2p1 = E2p2;
      //slice i3+2 (plus 2): p2<--frontmost slice
      E1p2 = E1[in_idx]; 
      E2p2 = E2[in_idx];
    }
    __syncthreads();

    if(threadIdx.x<1){//left halo: threadIdx.x<1
      //threadIdx.x=t1-1, [t2,t1-1]=t1-1+n1pad*t2=out_idx-1
      if(blockIdx.x>0){//not first block
  	sE1[t2][threadIdx.x] = E1[out_idx-1];
  	sE2[t2][threadIdx.x] = E2[out_idx-1];
  	sE3[t2][threadIdx.x] = E3[out_idx-1];
      }else{//first block, no left halo
  	sE1[t2][threadIdx.x] = 0.0;
  	sE2[t2][threadIdx.x] = 0.0;
  	sE3[t2][threadIdx.x] = 0.0;
      }
    }
    if(threadIdx.x<2){//right halo: threadIdx.x+BlockSize1+1<BlockSize1+3
      //threadIdx.x+BlockSize1+1=t1+BlockSize1, [t2,t1+BlockSize1]=t1+BlockSize1+n1pad*t2=out_idx+BlockSize1
      if(blockIdx.x<gridDim.x-1){//not last block
  	sE1[t2][threadIdx.x+BlockSize1+1] = E1[out_idx+BlockSize1]; 
  	sE2[t2][threadIdx.x+BlockSize1+1] = E2[out_idx+BlockSize1]; 
  	sE3[t2][threadIdx.x+BlockSize1+1] = E3[out_idx+BlockSize1]; 
      }else{//last block, no right halo
  	sE1[t2][threadIdx.x+BlockSize1+1] = 0.0;
  	sE2[t2][threadIdx.x+BlockSize1+1] = 0.0;
  	sE3[t2][threadIdx.x+BlockSize1+1] = 0.0;
      }
    }
    if(threadIdx.y<1){//top halo: threadIdx.y<1
      //threadIdx.y=t2-1, [t2-1,t1]=t1+n1pad*(t2-1)=out_idx-n1pad
      if(blockIdx.y>0){//not first block
  	sE1[threadIdx.y][t1] = E1[out_idx-n1pad];
  	sE2[threadIdx.y][t1] = E2[out_idx-n1pad];
  	sE3[threadIdx.y][t1] = E3[out_idx-n1pad];
      }else{//first block, no top halo
  	sE1[threadIdx.y][t1] = 0.0;
  	sE2[threadIdx.y][t1] = 0.0;
  	sE3[threadIdx.y][t1] = 0.0;
      }
    }
    if(threadIdx.y<2){//bottom halo: threadIdx.y+BlockSize2+1<BlockSize2+3
      //threadIdx.y+BlockSize2+1=t2+BlockSize2, [t2+BlockSize2,t1]=t1+n1pad*(t2+BlockSize2)=out_idx+n1pad*BlockSize2
      if(blockIdx.y<gridDim.y-1){//not last block
  	sE1[threadIdx.y+BlockSize2+1][t1] = E1[out_idx+BlockSize2*n1pad];
  	sE2[threadIdx.y+BlockSize2+1][t1] = E2[out_idx+BlockSize2*n1pad];
  	sE3[threadIdx.y+BlockSize2+1][t1] = E3[out_idx+BlockSize2*n1pad];
      }else{//last block, no bottom halo
  	sE1[threadIdx.y+BlockSize2+1][t1] = 0.0;
  	sE2[threadIdx.y+BlockSize2+1][t1] = 0.0;
  	sE3[threadIdx.y+BlockSize2+1][t1] = 0.0;
      }
    }
    if(validr){
      //[i3,i2,i1]= i1+n1pad*(i2+n2pad*i3)=out_idx
      sE1[t2][t1] = E1c;//current slice
      sE2[t2][t1] = E2c;//current slice
      sE3[t2][t1] = E3[out_idx];//current slice
    }
    __syncthreads();

    if(validrw){
      // D2E3 = c12*(E3[i3][i2+1][i1]-E3[i3][i2][i1]) + c22*(E3[i3][i2+2][i1]-E3[i3][i2-1][i1]);
      // D3E2 = c13*(E2[i3+1][i2][i1]-E2[i3][i2][i1]) + c23*(E2[i3+2][i2][i1]-E2[i3-1][i2][i1]);
      // D3E1 = c13*(E1[i3+1][i2][i1]-E1[i3][i2][i1]) + c23*(E1[i3+2][i2][i1]-E1[i3-1][i2][i1]);
      // D1E3 = c11*(E3[i3][i2][i1+1]-E3[i3][i2][i1]) + c21*(E3[i3][i2][i1+2]-E3[i3][i2][i1-1]);
      // D1E2 = c11*(E2[i3][i2][i1+1]-E2[i3][i2][i1]) + c21*(E2[i3][i2][i1+2]-E2[i3][i2][i1-1]);
      // D2E1 = c12*(E1[i3][i2+1][i1]-E1[i3][i2][i1]) + c22*(E1[i3][i2+2][i1]-E1[i3][i2-1][i1]);
      D1E2 = c11*(sE2[t2][t1+1]-sE2[t2][t1]) + c21*(sE2[t2][t1+2] - sE2[t2][t1-1]);
      D1E3 = c11*(sE3[t2][t1+1]-sE3[t2][t1]) + c21*(sE3[t2][t1+2] - sE3[t2][t1-1]);
      D2E1 = c12*(sE1[t2+1][t1]-sE1[t2][t1]) + c22*(sE1[t2+2][t1] - sE1[t2-1][t1]);
      D2E3 = c12*(sE3[t2+1][t1]-sE3[t2][t1]) + c22*(sE3[t2+2][t1] - sE3[t2-1][t1]);
      D3E1 = c13*(E1p1 - E1c)                + c23*(E1p2 - E1m1);
      D3E2 = c13*(E2p1 - E2c)                + c23*(E2p2 - E2m1);

      //CPML: mem=memory variable
      if(i1<nb){
  	//   memD1E3[i3][i2][i1] = b1[i1]*memD1E3[i3][i2][i1] + a1[i1]*D1E3;
  	//   memD1E2[i3][i2][i1] = b1[i1]*memD1E2[i3][i2][i1] + a1[i1]*D1E2;
  	//   D1E3 += memD1E3[i3][i2][i1];
  	//   D1E2 += memD1E2[i3][i2][i1];
  	out_idx1 = i1+2*nb*(i2+n2pad*i3);
  	tmp1 = memD1E3[out_idx1];
  	tmp2 = memD1E2[out_idx1];
  	tmp1 = b1[i1]*tmp1 + a1[i1]*D1E3;
  	tmp2 = b1[i1]*tmp2 + a1[i1]*D1E2;
  	memD1E3[out_idx1] = tmp1;
  	memD1E2[out_idx1] = tmp2;
  	D1E3 += tmp1;
  	D1E2 += tmp2;
      }
      if(i1>n1pad-1-nb){
  	j1 = n1pad-1-i1;
  	k1 = j1+nb;
  	//   memD1E3[i3][i2][k1] = b1[j1]*memD1E3[i3][i2][k1] + a1[j1]*D1E3;
  	//   memD1E2[i3][i2][k1] = b1[j1]*memD1E2[i3][i2][k1] + a1[j1]*D1E2;
  	//   D1E3 += memD1E3[i3][i2][k1];
  	//   D1E2 += memD1E2[i3][i2][k1];
  	out_idx1 = k1+2*nb*(i2+n2pad*i3);//[i3,i2,k1]
  	tmp1 = memD1E3[out_idx1];
  	tmp2 = memD1E2[out_idx1];
  	tmp1 = b1[j1]*tmp1 + a1[j1]*D1E3;
  	tmp2 = b1[j1]*tmp2 + a1[j1]*D1E2;
  	memD1E3[out_idx1] = tmp1;
  	memD1E2[out_idx1] = tmp2;
  	D1E3 += tmp1;
  	D1E2 += tmp2;
      }
      if(i2<nb){
  	// memD2E3[i3][i2][i1] = b2[i2]*memD2E3[i3][i2][i1] + a2[i2]*D2E3;
  	// memD2E1[i3][i2][i1] = b2[i2]*memD2E1[i3][i2][i1] + a2[i2]*D2E1;
  	// D2E3 += memD2E3[i3][i2][i1];
  	// D2E1 += memD2E1[i3][i2][i1];
  	out_idx2 = i1+n1pad*(i2+2*nb*i3);//[i3,k2,i1]
  	tmp1 = memD2E3[out_idx2];
  	tmp2 = memD2E1[out_idx2];
  	tmp1 = b2[i2]*tmp1 + a2[i2]*D2E3;
  	tmp2 = b2[i2]*tmp2 + a2[i2]*D2E1;
  	memD2E3[out_idx2] = tmp1;
  	memD2E1[out_idx2] = tmp2;
  	D2E3 += tmp1;
  	D2E1 += tmp2;
      }else if(i2>n2pad-1-nb){
  	j2 = n2pad-1-i2;
  	k2 = j2+nb;
  	// memD2E3[i3][k2][i1] = b2[j2]*memD2E3[i3][k2][i1] + a2[j2]*D2E3;
  	// memD2E1[i3][k2][i1] = b2[j2]*memD2E1[i3][k2][i1] + a2[j2]*D2E1;
  	// D2E3 += memD2E3[i3][k2][i1];
  	// D2E1 += memD2E1[i3][k2][i1];
  	out_idx2 = i1+n1pad*(k2+2*nb*i3);//[i3,k2,i1]
  	tmp1 = memD2E3[out_idx2];
  	tmp2 = memD2E1[out_idx2];
  	tmp1 = b2[j2]*tmp1 + a2[j2]*D2E3;
  	tmp2 = b2[j2]*tmp2 + a2[j2]*D2E1;
  	memD2E3[out_idx2] = tmp1;
  	memD2E1[out_idx2] = tmp2;
  	D2E3 += tmp1;
  	D2E1 += tmp2;
      }
      if(i3<nb){
  	//   memD3E2[i3][i2][i1] = b3[i3]*memD3E2[i3][i2][i1] + a3[i3]*D3E2;
  	//   memD3E1[i3][i2][i1] = b3[i3]*memD3E1[i3][i2][i1] + a3[i3]*D3E1;
  	//   D3E2 += memD3E2[i3][i2][i1];
  	//   D3E1 += memD3E1[i3][i2][i1];
  	//out_idx3 = i1+n1pad*(i2+n2pad*i3);
  	out_idx3 = out_idx;
  	tmp1 = memD3E2[out_idx3];
  	tmp2 = memD3E1[out_idx3];
  	tmp1 = b3[i3]*tmp1 + a3[i3]*D3E2;
  	tmp2 = b3[i3]*tmp2 + a3[i3]*D3E1;
  	memD3E2[out_idx3] = tmp1;
  	memD3E1[out_idx3] = tmp2;
  	D3E2 += tmp1;
  	D3E1 += tmp2;
      }else if(i3>n3pad-1-nb){
  	j3 = n3pad-1-i3;
  	k3 = j3+nb;
  	//   memD3E2[k3][i2][i1] = b3[j3]*memD3E2[k3][i2][i1] + a3[j3]*D3E2;
  	//   memD3E1[k3][i2][i1] = b3[j3]*memD3E1[k3][i2][i1] + a3[j3]*D3E1;
  	//   D3E2 += memD3E2[k3][i2][i1];
  	//   D3E1 += memD3E1[k3][i2][i1];
  	out_idx3 = i1+n1pad*(i2+n2pad*k3);//[k3,i2,i1]
  	tmp1 = memD3E2[out_idx3];
  	tmp2 = memD3E1[out_idx3];
  	tmp1 = b3[j3]*tmp1 + a3[j3]*D3E2;
  	tmp2 = b3[j3]*tmp2 + a3[j3]*D3E1;
  	memD3E2[out_idx3] = tmp1;
  	memD3E1[out_idx3] = tmp2;
  	D3E2 += tmp1;
  	D3E1 += tmp2;
      }

      curlE1[out_idx] = (D2E3-D3E2);
      curlE2[out_idx] = (D3E1-D1E3);
      curlE3[out_idx] = (D1E2-D2E1);
    }
  }//end for i3
  
}


//cuda_fdtd_update_E<<<dimg,dimb>>>()
//dimb.x = BlockSize1, dimb.y = BlockSize2
//grid.x = (n1pad+dimb.x-1)/dimb.x, grid.y = (n2pad+dimb.y-1)/dimb.y
__global__ void cuda_fdtd_curlH_nugrid(float *H1, float *H2, float *H3,
				       float *curlH1, float *curlH2, float *curlH3,
				       float *a1, float *b1, float *a2, float *b2, float *a3, float *b3,
				       float *memD1H2, float *memD1H3, float *memD2H1,
				       float *memD2H3, float *memD3H1, float *memD3H2, 
				       float *inveps11, float *inveps22, float *inveps33,
				       float c11, float c21, float c12, float c22, float *v3s,
				       int n1pad, int n2pad, int n3pad, int nb, int nbe, int airwave)
		
{
  const int i1 = threadIdx.x + blockDim.x*blockIdx.x;
  const int i2 = threadIdx.y + blockDim.y*blockIdx.y;
  const int t1 = threadIdx.x + 2;//t1>=2 && t1<BlockSize1+2
  const int t2 = threadIdx.y + 2;//t2>=2 && t2<BlockSize2+2

  __shared__ float sH1[BlockSize2+3][BlockSize1+3];
  __shared__ float sH2[BlockSize2+3][BlockSize1+3];
  __shared__ float sH3[BlockSize2+3][BlockSize1+3];

  int in_idx = i1+n1pad*i2;
  int out_idx = 0;
  int stride = n1pad*n2pad;

  float H1m2,H1m1,H1c,H1p1;
  float H2m2,H2m1,H2c,H2p1;
  float D1H2,D1H3,D2H1,D2H3,D3H1,D3H2;
  float tmp1,tmp2;
  int j1,k1,j2,k2,j3,k3,i3;
  int out_idx1, out_idx2, out_idx3;//index for pML memory variables

  bool validrw = false;//valid for reading from and writing into global memory
  bool validr = false;
  int i1min=2;
  int i1max=n1pad-2;
  int i2min=2;
  int i2max=n2pad-2;	
  int i3min=airwave?nbe:2;
  int i3max=n3pad-2;
  if(i1>=i1min && i1<=i1max && i2>=i2min && i2<=i2max) validrw = true;
  in_idx += stride*(i3min-2);
  if(i1<n1pad && i2<n2pad){
    validr = true;
    //slice i3min-2
    H1m1 = H1[in_idx];  
    H2m1 = H2[in_idx];  
    //slice i3min-1
    in_idx += stride;
    out_idx = in_idx;  
    H1c = H1[in_idx];  
    H2c = H2[in_idx];
    //slice i3min
    in_idx += stride;
    H1p1 = H1[in_idx];  
    H2p1 = H2[in_idx];  
  }

  for(i3=i3min; i3<=i3max; i3++){//i3-2>=0 && i3+1<n3pad
    //-------------------------------------------------------
    if(validr){
      //increase slice counter
      in_idx += stride;//slice i3min+1 when i3=i3min
      out_idx += stride;//slice i3min when i3=i3min
      //slice i3-2 (minus 2): m2<--m1
      H1m2 = H1m1;     
      H2m2 = H2m1;
      //slice i3-1 (minus 1): m1<--c
      H1m1 = H1c;
      H2m1 = H2c;
      //slice i3 (current): c<--p1
      H1c = H1p1;       
      H2c = H2p1;
      //slice i3+1 (plus 1): p1<--frontmost slice
      H1p1 = H1[in_idx]; 
      H2p1 = H2[in_idx];
    }
    __syncthreads();

    if(threadIdx.x<2){//left halo: threadIdx.x<2
      //threadIdx.x=t1-2, [t2,t1-2]=t1-2+n1pad*t2=out_idx-2
      if(blockIdx.x>0){//not first block
	sH1[t2][threadIdx.x] = H1[out_idx-2];
	sH2[t2][threadIdx.x] = H2[out_idx-2];
	sH3[t2][threadIdx.x] = H3[out_idx-2];
      }else{//first block, no left halo
	sH1[t2][threadIdx.x] = 0.0;
	sH2[t2][threadIdx.x] = 0.0;
	sH3[t2][threadIdx.x] = 0.0;
      }
    }
    if(threadIdx.x<1){//right halo: threadIdx.x+BlockSize1+2 <BlockSize1+3
      //threadIdx.x+BlockSize1+2=t1+BlockSize1, [t2,t1+BlockSize1]=t1+BlockSize1+n1pad*t2=out_idx+BlockSize1
      if(blockIdx.x<gridDim.x-1){//not last block
	sH1[t2][threadIdx.x+BlockSize1+2] = H1[out_idx+BlockSize1]; 
	sH2[t2][threadIdx.x+BlockSize1+2] = H2[out_idx+BlockSize1]; 
	sH3[t2][threadIdx.x+BlockSize1+2] = H3[out_idx+BlockSize1]; 
      }else{//last block, no right halo
	sH1[t2][threadIdx.x+BlockSize1+2] = 0.0;
	sH2[t2][threadIdx.x+BlockSize1+2] = 0.0;
	sH3[t2][threadIdx.x+BlockSize1+2] = 0.0;
      }
    }
    if(threadIdx.y<2){//top halo: threadIdx.y<2
      //threadIdx.y=t2-2, [t2-2,t1]=t1+n1pad*(t2-2)=out_idx-2*n1pad
      if(blockIdx.y>0){//not first block
	sH1[threadIdx.y][t1] = H1[out_idx-2*n1pad];
	sH2[threadIdx.y][t1] = H2[out_idx-2*n1pad];
	sH3[threadIdx.y][t1] = H3[out_idx-2*n1pad];
      }else{//first block, no top halo
	sH1[threadIdx.y][t1] = 0.0;
	sH2[threadIdx.y][t1] = 0.0;
	sH3[threadIdx.y][t1] = 0.0;
      }
    }
    if(threadIdx.y<1){//bottom halo: threadIdx.y+BlockSize2+2<BlockSize2+3
      //threadIdx.y+BlockSize2+2=t2+BlockSize2, [t2+BlockSize2,t1]=t1+n1pad*(t2+BlockSize2)=out_idx+n1pad*BlockSize2
      if(blockIdx.y<gridDim.x-1){//not last block
	sH1[threadIdx.y+BlockSize2+2][t1] = H1[out_idx+BlockSize2*n1pad];
	sH2[threadIdx.y+BlockSize2+2][t1] = H2[out_idx+BlockSize2*n1pad];
	sH3[threadIdx.y+BlockSize2+2][t1] = H3[out_idx+BlockSize2*n1pad];
      }else{//last block, no bottom halo
	sH1[threadIdx.y+BlockSize2+2][t1] = 0.0;
	sH2[threadIdx.y+BlockSize2+2][t1] = 0.0;
	sH3[threadIdx.y+BlockSize2+2][t1] = 0.0;
      }
    }
    if(validr){
      //[i3,i2,i1]: i1+n1pad*i2+n1pad*n2pad*i3=out_idx
      sH1[t2][t1] = H1c;//current slice
      sH2[t2][t1] = H2c;//current slice
      sH3[t2][t1] = H3[out_idx];//current slice
    }
    __syncthreads();

    if(validrw){
      // D2H3 = c12*(H3[i3][i2][i1]-H3[i3][i2-1][i1]) + c22*(H3[i3][i2+1][i1]-H3[i3][i2-2][i1]);
      // D3H2 = c13*(H2[i3][i2][i1]-H2[i3-1][i2][i1]) + c23*(H2[i3+1][i2][i1]-H2[i3-2][i2][i1]);
      // D3H1 = c13*(H1[i3][i2][i1]-H1[i3-1][i2][i1]) + c23*(H1[i3+1][i2][i1]-H1[i3-2][i2][i1]);
      // D1H3 = c11*(H3[i3][i2][i1]-H3[i3][i2][i1-1]) + c21*(H3[i3][i2][i1+1]-H3[i3][i2][i1-2]);
      // D1H2 = c11*(H2[i3][i2][i1]-H2[i3][i2][i1-1]) + c21*(H2[i3][i2][i1+1]-H2[i3][i2][i1-2]);
      // D2H1 = c12*(H1[i3][i2][i1]-H1[i3][i2-1][i1]) + c22*(H1[i3][i2+1][i1]-H1[i3][i2-2][i1]);
      D1H2 = c11*(sH2[t2][t1] - sH2[t2][t1-1]) + c21*(sH2[t2][t1+1] - sH2[t2][t1-2]);
      D1H3 = c11*(sH3[t2][t1] - sH3[t2][t1-1]) + c21*(sH3[t2][t1+1] - sH3[t2][t1-2]);
      D2H1 = c12*(sH1[t2][t1] - sH1[t2-1][t1]) + c22*(sH1[t2+1][t1] - sH1[t2-2][t1]);
      D2H3 = c12*(sH3[t2][t1] - sH3[t2-1][t1]) + c22*(sH3[t2+1][t1] - sH3[t2-2][t1]);
      D3H1 = v3s[0 + 4*i3]*H1m2 + v3s[1 + 4*i3]*H1m1 + v3s[2 + 4*i3]*H1c + v3s[3 + 4*i3]*H1p1;
      D3H2 = v3s[0 + 4*i3]*H2m2 + v3s[1 + 4*i3]*H2m1 + v3s[2 + 4*i3]*H2c + v3s[3 + 4*i3]*H2p1;

      //CPML: mem=memory variable
      if(i1<nb){
	//   memD1H3[i3][i2][i1] = b1[i1]*memD1H3[i3][i2][i1] + a1[i1]*D1H3;
	//   memD1H2[i3][i2][i1] = b1[i1]*memD1H2[i3][i2][i1] + a1[i1]*D1H2;
	//   D1H3 += memD1H3[i3][i2][i1];
	//   D1H2 += memD1H2[i3][i2][i1];
	out_idx1 = i1+2*nb*(i2+n2pad*i3);
	tmp1 = memD1H3[out_idx1];
	tmp2 = memD1H2[out_idx1];
	tmp1 = b1[i1]*tmp1 + a1[i1]*D1H3;
	tmp2 = b1[i1]*tmp2 + a1[i1]*D1H2;
	memD1H3[out_idx1] = tmp1;
	memD1H2[out_idx1] = tmp2;
	D1H3 += tmp1;
	D1H2 += tmp2;
      }
      if(i1>n1pad-1-nb){
	j1 = n1pad-1-i1;
	k1 = j1+nb;
	//   memD1H3[i3][i2][k1] = b1[j1]*memD1H3[i3][i2][k1] + a1[j1]*D1H3;
	//   memD1H2[i3][i2][k1] = b1[j1]*memD1H2[i3][i2][k1] + a1[j1]*D1H2;
	//   D1H3 += memD1H3[i3][i2][k1];
	//   D1H2 += memD1H2[i3][i2][k1];
	out_idx1 = k1+2*nb*(i2+n2pad*i3);//[i3,i2,k1]
	tmp1 = memD1H3[out_idx1];
	tmp2 = memD1H2[out_idx1];
	tmp1 = b1[j1]*tmp1 + a1[j1]*D1H3;
	tmp2 = b1[j1]*tmp2 + a1[j1]*D1H2;
	memD1H3[out_idx1] = tmp1;
	memD1H2[out_idx1] = tmp2;
	D1H3 += tmp1;
	D1H2 += tmp2;
      }
      if(i2<nb){
	// memD2H3[i3][i2][i1] = b2[i2]*memD2H3[i3][i2][i1] + a2[i2]*D2H3;
	// memD2H1[i3][i2][i1] = b2[i2]*memD2H1[i3][i2][i1] + a2[i2]*D2H1;
	// D2H3 += memD2H3[i3][i2][i1];
	// D2H1 += memD2H1[i3][i2][i1];
	out_idx2 = i1+n1pad*(i2+2*nb*i3);//[i3,k2,i1]
	tmp1 = memD2H3[out_idx2];
	tmp2 = memD2H1[out_idx2];
	tmp1 = b2[i2]*tmp1 + a2[i2]*D2H3;
	tmp2 = b2[i2]*tmp2 + a2[i2]*D2H1;
	memD2H3[out_idx2] = tmp1;
	memD2H1[out_idx2] = tmp2;
	D2H3 += tmp1;
	D2H1 += tmp2;
      }else if(i2>n2pad-1-nb){
	j2 = n2pad-1-i2;
	k2 = j2+nb;
	// memD2H3[i3][k2][i1] = b2[j2]*memD2H3[i3][k2][i1] + a2[j2]*D2H3;
	// memD2H1[i3][k2][i1] = b2[j2]*memD2H1[i3][k2][i1] + a2[j2]*D2H1;
	// D2H3 += memD2H3[i3][k2][i1];
	// D2H1 += memD2H1[i3][k2][i1];
	out_idx2 = i1+n1pad*(k2+2*nb*i3);//[i3,k2,i1]
	tmp1 = memD2H3[out_idx2];
	tmp2 = memD2H1[out_idx2];
	tmp1 = b2[j2]*tmp1 + a2[j2]*D2H3;
	tmp2 = b2[j2]*tmp2 + a2[j2]*D2H1;
	memD2H3[out_idx2] = tmp1;
	memD2H1[out_idx2] = tmp2;
	D2H3 += tmp1;
	D2H1 += tmp2;
      }
      if(i3<nb){
	//   memD3H2[i3][i2][i1] = b3[i3]*memD3H2[i3][i2][i1] + a3[i3]*D3H2;
	//   memD3H1[i3][i2][i1] = b3[i3]*memD3H1[i3][i2][i1] + a3[i3]*D3H1;
	//   D3H2 += memD3H2[i3][i2][i1];
	//   D3H1 += memD3H1[i3][i2][i1];
	//out_idx3 = i1+n1pad*(i2+n2pad*i3);
	out_idx3 = out_idx;
	tmp1 = memD3H2[out_idx3];
	tmp2 = memD3H1[out_idx3];
	tmp1 = b3[i3]*tmp1 + a3[i3]*D3H2;
	tmp2 = b3[i3]*tmp2 + a3[i3]*D3H1;
	memD3H2[out_idx3] = tmp1;
	memD3H1[out_idx3] = tmp2;
	D3H2 += tmp1;
	D3H1 += tmp2;
      }else if(i3>n3pad-1-nb){
	j3 = n3pad-1-i3;
	k3 = j3+nb;
	//   memD3H2[k3][i2][i1] = b3[j3]*memD3H2[k3][i2][i1] + a3[j3]*D3H2;
	//   memD3H1[k3][i2][i1] = b3[j3]*memD3H1[k3][i2][i1] + a3[j3]*D3H1;
	//   D3H2 += memD3H2[k3][i2][i1];
	//   D3H1 += memD3H1[k3][i2][i1];
	out_idx3 = i1+n1pad*(i2+n2pad*k3);//[k3,i2,i1]
	tmp1 = memD3H2[out_idx3];
	tmp2 = memD3H1[out_idx3];
	tmp1 = b3[j3]*tmp1 + a3[j3]*D3H2;
	tmp2 = b3[j3]*tmp2 + a3[j3]*D3H1;
	memD3H2[out_idx3] = tmp1;
	memD3H1[out_idx3] = tmp2;
	D3H2 += tmp1;
	D3H1 += tmp2;
      }

      curlH1[out_idx] = D2H3 - D3H2;
      curlH2[out_idx] = D3H1 - D1H3;
      curlH3[out_idx] = D1H2 - D2H1;
    }
  }//end for i3
}

//cuda_fdtd_update_H<<<dimg,dimb>>>()
//dimb.x = BlockSize1, dimb.y = BlockSize2
//grid.x = (n1pad+dimb.x-1)/dimb.x, grid.y = (n2pad+dimb.y-1)/dimb.y
__global__ void cuda_fdtd_curlE_nugrid(float *E1, float *E2, float *E3,
				       float *curlE1, float *curlE2, float *curlE3,
				       float *a1, float *b1, float *a2, float *b2, float *a3, float *b3,
				       float *memD1E2, float *memD1E3, float *memD2E1,
				       float *memD2E3, float *memD3E1, float *memD3E2, 
				       float c11, float c21, float c12, float c22, float *v3,
				       int n1pad, int n2pad, int n3pad, int nb, int nbe, int airwave)
{
  const int i1 = threadIdx.x + blockDim.x*blockIdx.x;
  const int i2 = threadIdx.y + blockDim.y*blockIdx.y;
  const int t1 = threadIdx.x + 1;
  const int t2 = threadIdx.y + 1;

  __shared__ float sE1[BlockSize2+3][BlockSize1+3];
  __shared__ float sE2[BlockSize2+3][BlockSize1+3];
  __shared__ float sE3[BlockSize2+3][BlockSize1+3];

  int in_idx = i1+n1pad*i2;
  int out_idx = 0;
  int stride = n1pad*n2pad;//n1pad+

  float E1m1,E1c,E1p1,E1p2;
  float E2m1,E2c,E2p1,E2p2;
  float D1E2,D1E3,D2E1,D2E3,D3E1,D3E2;
  float tmp1,tmp2;
  int j1,k1,j2,k2,j3,k3,i3;
  int out_idx1, out_idx2, out_idx3;//index for pML memory variables

  bool validrw = false;//true=valid for reading from and writing into global memory
  bool validr = false;
  int i1min=1;
  int i1max=n1pad-3;
  int i2min=1;
  int i2max=n2pad-3;	
  int i3min=airwave?nbe:1;
  int i3max=n3pad-3;

  if(i1>=i1min && i1<=i1max && i2>=i2min && i2<=i2max) validrw = true;
  in_idx += stride*(i3min-1);
  if(i1<n1pad && i2<n2pad){
    validr = true;
    //slice i3min-1
    out_idx = in_idx;
    E1c = E1[in_idx];  
    E2c = E2[in_idx];  
    //slice i3min
    in_idx += stride;
    E1p1 = E1[in_idx];  
    E2p1 = E2[in_idx];  
    //slice i3min+1
    in_idx += stride;
    E1p2 = E1[in_idx]; 
    E2p2 = E2[in_idx];  
  }

  
  for(i3=i3min; i3<=i3max; i3++){//i3-1>=0 && i3+2<n2pad
    //-------------------------------------------------------
    if(validr){
      //increase slice counter
      in_idx += stride;//slice i3min+2 when i3=i3min
      out_idx += stride;//slice i3min when i3=i3min
      //slice i3-1 (minus 1): m1<--c
      E1m1 = E1c;
      E2m1 = E2c;
      //slice i3 (current): c<--p1
      E1c = E1p1;       
      E2c = E2p1;
      //slice i3+1 (plus 1): p1<--p2
      E1p1 = E1p2;     
      E2p1 = E2p2;
      //slice i3+2 (plus 2): p2<--frontmost slice
      E1p2 = E1[in_idx]; 
      E2p2 = E2[in_idx];
    }
    __syncthreads();

    if(threadIdx.x<1){//left halo: threadIdx.x<1
      //threadIdx.x=t1-1, [t2,t1-1]=t1-1+n1pad*t2=out_idx-1
      if(blockIdx.x>0){//not first block
  	sE1[t2][threadIdx.x] = E1[out_idx-1];
  	sE2[t2][threadIdx.x] = E2[out_idx-1];
  	sE3[t2][threadIdx.x] = E3[out_idx-1];
      }else{//first block, no left halo
  	sE1[t2][threadIdx.x] = 0.0;
  	sE2[t2][threadIdx.x] = 0.0;
  	sE3[t2][threadIdx.x] = 0.0;
      }
    }
    if(threadIdx.x<2){//right halo: threadIdx.x+BlockSize1+1<BlockSize1+3
      //threadIdx.x+BlockSize1+1=t1+BlockSize1, [t2,t1+BlockSize1]=t1+BlockSize1+n1pad*t2=out_idx+BlockSize1
      if(blockIdx.x<gridDim.x-1){//not last block
  	sE1[t2][threadIdx.x+BlockSize1+1] = E1[out_idx+BlockSize1]; 
  	sE2[t2][threadIdx.x+BlockSize1+1] = E2[out_idx+BlockSize1]; 
  	sE3[t2][threadIdx.x+BlockSize1+1] = E3[out_idx+BlockSize1]; 
      }else{//last block, no right halo
  	sE1[t2][threadIdx.x+BlockSize1+1] = 0.0;
  	sE2[t2][threadIdx.x+BlockSize1+1] = 0.0;
  	sE3[t2][threadIdx.x+BlockSize1+1] = 0.0;
      }
    }
    if(threadIdx.y<1){//top halo: threadIdx.y<1
      //threadIdx.y=t2-1, [t2-1,t1]=t1+n1pad*(t2-1)=out_idx-n1pad
      if(blockIdx.y>0){//not first block
  	sE1[threadIdx.y][t1] = E1[out_idx-n1pad];
  	sE2[threadIdx.y][t1] = E2[out_idx-n1pad];
  	sE3[threadIdx.y][t1] = E3[out_idx-n1pad];
      }else{//first block, no top halo
  	sE1[threadIdx.y][t1] = 0.0;
  	sE2[threadIdx.y][t1] = 0.0;
  	sE3[threadIdx.y][t1] = 0.0;
      }
    }
    if(threadIdx.y<2){//bottom halo: threadIdx.y+BlockSize2+1<BlockSize2+3
      //threadIdx.y+BlockSize2+1=t2+BlockSize2, [t2+BlockSize2,t1]=t1+n1pad*(t2+BlockSize2)=out_idx+n1pad*BlockSize2
      if(blockIdx.y<gridDim.y-1){//not last block
  	sE1[threadIdx.y+BlockSize2+1][t1] = E1[out_idx+BlockSize2*n1pad];
  	sE2[threadIdx.y+BlockSize2+1][t1] = E2[out_idx+BlockSize2*n1pad];
  	sE3[threadIdx.y+BlockSize2+1][t1] = E3[out_idx+BlockSize2*n1pad];
      }else{//last block, no bottom halo
  	sE1[threadIdx.y+BlockSize2+1][t1] = 0.0;
  	sE2[threadIdx.y+BlockSize2+1][t1] = 0.0;
  	sE3[threadIdx.y+BlockSize2+1][t1] = 0.0;
      }
    }
    if(validr){
      //[i3,i2,i1]= i1+n1pad*(i2+n2pad*i3)=out_idx
      sE1[t2][t1] = E1c;//current slice
      sE2[t2][t1] = E2c;//current slice
      sE3[t2][t1] = E3[out_idx];//current slice
    }
    __syncthreads();

    if(validrw){
      // D2E3 = c12*(E3[i3][i2+1][i1]-E3[i3][i2][i1]) + c22*(E3[i3][i2+2][i1]-E3[i3][i2-1][i1]);
      // D3E2 = c13*(E2[i3+1][i2][i1]-E2[i3][i2][i1]) + c23*(E2[i3+2][i2][i1]-E2[i3-1][i2][i1]);
      // D3E1 = c13*(E1[i3+1][i2][i1]-E1[i3][i2][i1]) + c23*(E1[i3+2][i2][i1]-E1[i3-1][i2][i1]);
      // D1E3 = c11*(E3[i3][i2][i1+1]-E3[i3][i2][i1]) + c21*(E3[i3][i2][i1+2]-E3[i3][i2][i1-1]);
      // D1E2 = c11*(E2[i3][i2][i1+1]-E2[i3][i2][i1]) + c21*(E2[i3][i2][i1+2]-E2[i3][i2][i1-1]);
      // D2E1 = c12*(E1[i3][i2+1][i1]-E1[i3][i2][i1]) + c22*(E1[i3][i2+2][i1]-E1[i3][i2-1][i1]);
      D1E2 = c11*(sE2[t2][t1+1]-sE2[t2][t1]) + c21*(sE2[t2][t1+2] - sE2[t2][t1-1]);
      D1E3 = c11*(sE3[t2][t1+1]-sE3[t2][t1]) + c21*(sE3[t2][t1+2] - sE3[t2][t1-1]);
      D2E1 = c12*(sE1[t2+1][t1]-sE1[t2][t1]) + c22*(sE1[t2+2][t1] - sE1[t2-1][t1]);
      D2E3 = c12*(sE3[t2+1][t1]-sE3[t2][t1]) + c22*(sE3[t2+2][t1] - sE3[t2-1][t1]);
      D3E1 = v3[0 + 4*i3]*E1m1 + v3[1 + 4*i3]*E1c + v3[2 + 4*i3]*E1p1 + v3[3 + 4*i3]*E1p2;
      D3E2 = v3[0 + 4*i3]*E2m1 + v3[1 + 4*i3]*E2c + v3[2 + 4*i3]*E2p1 + v3[3 + 4*i3]*E2p2;

      //CPML: mem=memory variable
      if(i1<nb){
  	//   memD1E3[i3][i2][i1] = b1[i1]*memD1E3[i3][i2][i1] + a1[i1]*D1E3;
  	//   memD1E2[i3][i2][i1] = b1[i1]*memD1E2[i3][i2][i1] + a1[i1]*D1E2;
  	//   D1E3 += memD1E3[i3][i2][i1];
  	//   D1E2 += memD1E2[i3][i2][i1];
  	out_idx1 = i1+2*nb*(i2+n2pad*i3);
  	tmp1 = memD1E3[out_idx1];
  	tmp2 = memD1E2[out_idx1];
  	tmp1 = b1[i1]*tmp1 + a1[i1]*D1E3;
  	tmp2 = b1[i1]*tmp2 + a1[i1]*D1E2;
  	memD1E3[out_idx1] = tmp1;
  	memD1E2[out_idx1] = tmp2;
  	D1E3 += tmp1;
  	D1E2 += tmp2;
      }
      if(i1>n1pad-1-nb){
  	j1 = n1pad-1-i1;
  	k1 = j1+nb;
  	//   memD1E3[i3][i2][k1] = b1[j1]*memD1E3[i3][i2][k1] + a1[j1]*D1E3;
  	//   memD1E2[i3][i2][k1] = b1[j1]*memD1E2[i3][i2][k1] + a1[j1]*D1E2;
  	//   D1E3 += memD1E3[i3][i2][k1];
  	//   D1E2 += memD1E2[i3][i2][k1];
  	out_idx1 = k1+2*nb*(i2+n2pad*i3);//[i3,i2,k1]
  	tmp1 = memD1E3[out_idx1];
  	tmp2 = memD1E2[out_idx1];
  	tmp1 = b1[j1]*tmp1 + a1[j1]*D1E3;
  	tmp2 = b1[j1]*tmp2 + a1[j1]*D1E2;
  	memD1E3[out_idx1] = tmp1;
  	memD1E2[out_idx1] = tmp2;
  	D1E3 += tmp1;
  	D1E2 += tmp2;
      }
      if(i2<nb){
  	// memD2E3[i3][i2][i1] = b2[i2]*memD2E3[i3][i2][i1] + a2[i2]*D2E3;
  	// memD2E1[i3][i2][i1] = b2[i2]*memD2E1[i3][i2][i1] + a2[i2]*D2E1;
  	// D2E3 += memD2E3[i3][i2][i1];
  	// D2E1 += memD2E1[i3][i2][i1];
  	out_idx2 = i1+n1pad*(i2+2*nb*i3);//[i3,k2,i1]
  	tmp1 = memD2E3[out_idx2];
  	tmp2 = memD2E1[out_idx2];
  	tmp1 = b2[i2]*tmp1 + a2[i2]*D2E3;
  	tmp2 = b2[i2]*tmp2 + a2[i2]*D2E1;
  	memD2E3[out_idx2] = tmp1;
  	memD2E1[out_idx2] = tmp2;
  	D2E3 += tmp1;
  	D2E1 += tmp2;
      }else if(i2>n2pad-1-nb){
  	j2 = n2pad-1-i2;
  	k2 = j2+nb;
  	// memD2E3[i3][k2][i1] = b2[j2]*memD2E3[i3][k2][i1] + a2[j2]*D2E3;
  	// memD2E1[i3][k2][i1] = b2[j2]*memD2E1[i3][k2][i1] + a2[j2]*D2E1;
  	// D2E3 += memD2E3[i3][k2][i1];
  	// D2E1 += memD2E1[i3][k2][i1];
  	out_idx2 = i1+n1pad*(k2+2*nb*i3);//[i3,k2,i1]
  	tmp1 = memD2E3[out_idx2];
  	tmp2 = memD2E1[out_idx2];
  	tmp1 = b2[j2]*tmp1 + a2[j2]*D2E3;
  	tmp2 = b2[j2]*tmp2 + a2[j2]*D2E1;
  	memD2E3[out_idx2] = tmp1;
  	memD2E1[out_idx2] = tmp2;
  	D2E3 += tmp1;
  	D2E1 += tmp2;
      }
      if(i3<nb){
  	//   memD3E2[i3][i2][i1] = b3[i3]*memD3E2[i3][i2][i1] + a3[i3]*D3E2;
  	//   memD3E1[i3][i2][i1] = b3[i3]*memD3E1[i3][i2][i1] + a3[i3]*D3E1;
  	//   D3E2 += memD3E2[i3][i2][i1];
  	//   D3E1 += memD3E1[i3][i2][i1];
  	//out_idx3 = i1+n1pad*(i2+n2pad*i3);
  	out_idx3 = out_idx;
  	tmp1 = memD3E2[out_idx3];
  	tmp2 = memD3E1[out_idx3];
  	tmp1 = b3[i3]*tmp1 + a3[i3]*D3E2;
  	tmp2 = b3[i3]*tmp2 + a3[i3]*D3E1;
  	memD3E2[out_idx3] = tmp1;
  	memD3E1[out_idx3] = tmp2;
  	D3E2 += tmp1;
  	D3E1 += tmp2;
      }else if(i3>n3pad-1-nb){
  	j3 = n3pad-1-i3;
  	k3 = j3+nb;
  	//   memD3E2[k3][i2][i1] = b3[j3]*memD3E2[k3][i2][i1] + a3[j3]*D3E2;
  	//   memD3E1[k3][i2][i1] = b3[j3]*memD3E1[k3][i2][i1] + a3[j3]*D3E1;
  	//   D3E2 += memD3E2[k3][i2][i1];
  	//   D3E1 += memD3E1[k3][i2][i1];
  	out_idx3 = i1+n1pad*(i2+n2pad*k3);//[k3,i2,i1]
  	tmp1 = memD3E2[out_idx3];
  	tmp2 = memD3E1[out_idx3];
  	tmp1 = b3[j3]*tmp1 + a3[j3]*D3E2;
  	tmp2 = b3[j3]*tmp2 + a3[j3]*D3E1;
  	memD3E2[out_idx3] = tmp1;
  	memD3E1[out_idx3] = tmp2;
  	D3E2 += tmp1;
  	D3E1 += tmp2;
      }

      curlE1[out_idx] = (D2E3-D3E2);
      curlE2[out_idx] = (D3E1-D1E3);
      curlE3[out_idx] = (D1E2-D2E1);
    }
  }//end for i3
  
}



__global__ void cuda_fdtd_update_E(float *E1, float *E2, float *E3, float *curlH1, float *curlH2, float *curlH3, float *inveps11, float *inveps22, float *inveps33, int n1pad, int n2pad, int n3pad, float dt)
{
  const int i1 = threadIdx.x + blockDim.x*blockIdx.x;
  const int i2 = threadIdx.y + blockDim.y*blockIdx.y;
  int i3, id;
  
  for(i3=0; i3<n3pad; i3++){
    if(i1<n1pad && i2<n2pad){
      id = i1+n1pad*(i2+ n2pad*i3);
      E1[id] += dt*inveps11[id]*curlH1[id];
      E2[id] += dt*inveps22[id]*curlH2[id];
      E3[id] += dt*inveps33[id]*curlH3[id];
    }
  }
}

__global__ void cuda_fdtd_update_H(float *H1, float *H2, float *H3, float *curlE1, float *curlE2, float *curlE3, float dt, int n1pad, int n2pad, int n3pad)
{
  const int i1 = threadIdx.x + blockDim.x*blockIdx.x;
  const int i2 = threadIdx.y + blockDim.y*blockIdx.y;
  int i3,id;

  const float factor = dt*invmu0;
  
  for(i3=0; i3<n3pad; i3++){
    if(i1<n1pad && i2<n2pad){
      id = i1+n1pad*(i2+ n2pad*i3);

      H1[id] -= factor*curlE1[id];
      H2[id] -= factor*curlE2[id];
      H3[id] -= factor*curlE3[id];
    }
  }
}


//<<<dimGrid,dimBlock>>>
//dimBlock.x = BlockSize1; 
//dimBlock.y = BlockSize2
//dimGrid.x = (n1fft+BlockSize1-1)/BlockSize1; 
//dimBlock.y = (n2fft+BlockSize2-1)/BlockSize2
//tmp_air: n1fft*n2fft; emf_air=&d_E3[n1pad*n2pad*nbe]: n1pad*n2pad
__global__ void cuda_airwave_bc_copy(cuFloatComplex *emfft, float *emf, int n1pad, int n2pad, int n1fft, int n2fft)
{
  const int i1 = threadIdx.x + blockDim.x*blockIdx.x;
  const int i2 = threadIdx.y + blockDim.y*blockIdx.y;
  int id_fft = i1+n1fft*i2;
  int id_airwave;

  if(i1<n1pad && i2<n2pad) {
    id_airwave = i1+n1pad*i2;
    emfft[id_fft].x = emf[id_airwave];
    emfft[id_fft].y = 0.0;
  }else if(i1<n1fft && i2<n2fft) {
    emfft[id_fft].x = 0.0;
    emfft[id_fft].y = 0.0;
  }
}


//<<<dimGrid,dimBlock>>>, emf=H,E
// normalization = 1./(n1fft*n2fft);//cufft did not include normalization factor
__global__ void cuda_airwave_bc_back2emf(float *emf, cuFloatComplex *emfft, int n1pad, int n2pad, int n1fft, float normalization)
{
  const int i1 = threadIdx.x + blockDim.x*blockIdx.x;
  const int i2 = threadIdx.y + blockDim.y*blockIdx.y;

  if(i1<n1pad && i2<n2pad) {
    int id_air = i1+n1pad*i2;
    int id_fft = i1+n1fft*i2;
    emf[id_air] = emfft[id_fft].x*normalization;
  }

}

//do it for both H1 and H2
__global__ void cuda_airwave_bc_scale_FH(cuFloatComplex *emfft, cuFloatComplex *sHkxky, int n1fft, int n2fft)
//__global__ void cuda_airwave_bc_H(cuFloatComplex *FH2, cuFloatComplex *sH2kxky, int n1fft, int n2fft)
{
  const int i1 = threadIdx.x + blockDim.x*blockIdx.x;
  const int i2 = threadIdx.y + blockDim.y*blockIdx.y;

  if(i1<n1fft && i2<n2fft){
    int id = i1+n1fft*i2;
    float a = emfft[id].x;
    float b = emfft[id].y;
    //(a+I*b)(c+I*d)=(ac-bd) + I*(ad+bc)
    emfft[id].x = a*sHkxky[id].x - b*sHkxky[id].y;
    emfft[id].y = a*sHkxky[id].y + b*sHkxky[id].x;
  }
}

__global__ void cuda_airwave_bc_scale_FE(cuFloatComplex *emfft, float *sEkxky, int n1fft, int n2fft)
//__global__ void cuda_airwave_bc_H(cuFloatComplex *FH2, cuFloatComplex *sH2kxky, int n1fft, int n2fft)
{
  const int i1 = threadIdx.x + blockDim.x*blockIdx.x;
  const int i2 = threadIdx.y + blockDim.y*blockIdx.y;

  if(i1<n1fft && i2<n2fft){
    int id = i1+n1fft*i2;
    emfft[id].x *= sEkxky[id];
    emfft[id].y *= sEkxky[id];
  }
}


//pointwise multiplication with a factor determined by complex frequency
__global__ void cuda_dtft_emf(cuFloatComplex *fwd_E1, cuComplex *expfactor, float *E1,
			      int nb, int n123pad, int n1pad, int n2pad, int n3pad, int nfreq)

{
  const int i1 = threadIdx.x + blockDim.x*blockIdx.x;
  const int i2 = threadIdx.y + blockDim.y*blockIdx.y;

  int idx2d = i1+n1pad*i2;
  int stride = n1pad*n2pad;
  int i3,ifreq,idx3d,id;

  int i1min = nb;
  int i1max = n1pad-1-nb;
  int i2min = nb;
  int i2max = n2pad-1-nb;
  int i3min = nb;
  int i3max = n3pad-1-nb;

  for(ifreq=0; ifreq<nfreq; ++ifreq){
    idx3d = idx2d + stride*i3min;
    id = idx3d + n123pad*ifreq;
    for(i3=i3min; i3<=i3max; ++i3, idx3d += stride, id += stride){

      if(i1>=i1min && i1<=i1max && i2>=i2min && i2<=i2max) {

	fwd_E1[id].x += E1[idx3d]*expfactor[ifreq].x;
	fwd_E1[id].y += E1[idx3d]*expfactor[ifreq].y;
      }
    }
  }
}

//<<<(nchsrc*nsrc*nsubsrc+BlockSize-1)/BlockSize, BlockSize>>>
__global__ void cuda_inject_electric_source(int *rg_src_i1, int *rg_src_i2, int *rg_src_i3,
					    float *rg_src_w1, float *rg_src_w2, float *rg_src_w3,
					    int *sg_src_i1, int *sg_src_i2, int *sg_src_i3,
					    float *sg_src_w1, float *sg_src_w2, float *sg_src_w3,
					    float *inveps11, float *inveps22, float *inveps33,
					    float *curlH1, float *curlH2, float *curlH3, int *chsrc, 
					    float src_it, float d1, float d2, float d3, 
					    int nchsrc, int nsrc, int nsubsrc, 
					    int n1pad, int n2pad, int n3pad, int nbe, int radius)
/*< inject a source time function into EM field >*/
{
  const int id = threadIdx.x + blockDim.x*blockIdx.x;
  int ic,isubsrc,i1,i2,i3,ix1,ix2,ix3,i1_,i2_,i3_,idx3d,ishift;
  float w1,w2,w3,s;

  s = src_it/(d1*d2*d3);/* source normalized by volume */
  s /= (float)nsubsrc; /*since one source is distributed over many points */

  if(id<nchsrc*nsrc*nsubsrc){
    //id = isub+nsubsrc*(isrc+nsrc*ic)=isubsrc+nsubsrc*nsrc*ic
    // int isub = id%nsubsrc;
    // int isrc = id/nsubsrc%nsrc;
    isubsrc = id%(nsubsrc*nsrc);
    ic = id/(nsubsrc*nsrc);
    ishift = radius-1+2*radius*isubsrc;

    if(chsrc[ic]==1){//(strcmp(chsrc[ic],"Ex") == 0){
      /* staggered grid: E1[i1,i2,i3] = Ex[i1+0.5,i2,i3] */
      ix1 = sg_src_i1[isubsrc];
      ix2 = rg_src_i2[isubsrc];
      ix3 = rg_src_i3[isubsrc];
      for(i3=-radius+1; i3<=radius; i3++){
    	w3 = rg_src_w3[i3+ishift];
    	i3_ = ix3+i3;
    	for(i2=-radius+1; i2<=radius; i2++){
    	  w2 = rg_src_w2[i2+ishift];
    	  i2_ = ix2+i2;
    	  for(i1=-radius+1; i1<=radius; i1++){
    	    w1 = sg_src_w1[i1+ishift];
    	    i1_ = ix1+i1;

    	    idx3d = i1_+n1pad*(i2_+n2pad*i3_);
    	    curlH1[idx3d] -= s*w1*w2*w3;
    	  }/* end for i1 */
    	}/* end for i2 */
      }/* end for i3 */
    }else if(chsrc[ic]==2){//(strcmp(chsrc[ic],"Ey") == 0){
      /* staggered grid: E2[i1,i2,i3] = Ey[i1,i2+0.5,i3] */
      ix1 = rg_src_i1[isubsrc];
      ix2 = sg_src_i2[isubsrc];
      ix3 = rg_src_i3[isubsrc];
      for(i3=-radius+1; i3<=radius; i3++){
      	w3 = rg_src_w3[i3+ishift];
      	i3_ = ix3+i3;
      	for(i2=-radius+1; i2<=radius; i2++){
      	  w2 = sg_src_w2[i2+ishift];
      	  i2_ = ix2+i2;
      	  for(i1=-radius+1; i1<=radius; i1++){
      	    w1 = rg_src_w1[i1+ishift];
      	    i1_ = ix1+i1;

      	    idx3d = i1_+n1pad*(i2_+n2pad*i3_);
      	    curlH2[idx3d] -= s*w1*w2*w3;
      	  }
      	}
      }
    }else if(chsrc[ic]==3){//(strcmp(chsrc[ic],"Ez") == 0){
      /* staggered grid: E3[i1,i2,i3] = Ez[i1,i2,i3+0.5] */
      ix1 = rg_src_i1[isubsrc];
      ix2 = rg_src_i2[isubsrc];
      ix3 = sg_src_i3[isubsrc];
      for(i3=-radius+1; i3<=radius; i3++){
      	w3 = sg_src_w3[i3+ishift];
      	i3_ = ix3+i3;
      	for(i2=-radius+1; i2<=radius; i2++){
      	  w2 = rg_src_w2[i2+ishift];
      	  i2_ = ix2+i2;
      	  for(i1=-radius+1; i1<=radius; i1++){
      	    w1 = rg_src_w1[i1+ishift];
      	    i1_ = ix1+i1;

      	    idx3d = i1_+n1pad*(i2_+n2pad*i3_);
      	    curlH3[idx3d] -= s*w1*w2*w3;
      	  }
      	}
      }
    }//end if Ex/Ey/Ez

  }//end if id

}


//<<<(nchsrc*nsrc*nsubsrc+BlockSize-1)/BlockSize, BlockSize>>>
__global__ void cuda_inject_magnetic_source(int *rg_src_i1, int *rg_src_i2, int *rg_src_i3,
					    float *rg_src_w1, float *rg_src_w2, float *rg_src_w3,
					    int *sg_src_i1, int *sg_src_i2, int *sg_src_i3,
					    float *sg_src_w1, float *sg_src_w2, float *sg_src_w3,
					    float *curlE1, float *curlE2, float *curlE3, int *chsrc, 
					    float src_it, float d1, float d2, float d3, 
					    int nchsrc, int nsrc, int nsubsrc, 
					    int n1pad, int n2pad, int n3pad, int nbe, int radius)
{
  const int id = threadIdx.x + blockDim.x*blockIdx.x;
  int ic,isubsrc,i1,i2,i3,ix1,ix2,ix3,i1_,i2_,i3_,idx3d,ishift;
  float w1,w2,w3,s;

  s = src_it/(d1*d2*d3);/* source normalized by volume */
  s /= (float)nsubsrc; /*since one source is distributed to many points */

  if(id<nchsrc*nsrc*nsubsrc){
    //id = isub+nsubsrc*(isrc+nsrc*ic)=isubsrc+nsubsrc*nsrc*ic
    // int isub = id%nsubsrc;
    // int isrc = id/nsubsrc%nsrc;
    isubsrc = id%(nsubsrc*nsrc);
    ic = id/(nsubsrc*nsrc);
    ishift = radius-1+2*radius*isubsrc;

    if(chsrc[ic]==4){//(strcmp(chsrc[ic],"Ex") == 0){
      /* staggered grid: H1[i1,i2,i3] = Hx[i1,i2+0.5,i3+0.5] */
      ix1 = rg_src_i1[isubsrc];
      ix2 = sg_src_i2[isubsrc];
      ix3 = sg_src_i3[isubsrc];
      for(i3=-radius+1; i3<=radius; i3++){
	w3 = sg_src_w3[i3+ishift];
	i3_ = ix3+i3;
	for(i2=-radius+1; i2<=radius; i2++){
	  w2 = sg_src_w2[i2+ishift];
	  i2_ = ix2+i2;
	  for(i1=-radius+1; i1<=radius; i1++){
	    w1 = rg_src_w1[i1+ishift];
	    i1_ = ix1+i1;

	    idx3d = i1_+n1pad*(i2_+n2pad*i3_);
	    curlE1[idx3d] -= s*w1*w2*w3;
	  }/* end for i1 */
	}/* end for i2 */
      }/* end for i3 */
    }else if(chsrc[ic]==5){//(strcmp(chsrc[ic],"Hy") == 0){
      /* staggered grid: H2[i1,i2,i3] = Hy[i1,i2+0.5,i3] */
      ix1 = sg_src_i1[isubsrc];
      ix2 = rg_src_i2[isubsrc];
      ix3 = sg_src_i3[isubsrc];
      for(i3=-radius+1; i3<=radius; i3++){
      	w3 = sg_src_w3[i3+ishift];
      	i3_ = ix3+i3;
      	for(i2=-radius+1; i2<=radius; i2++){
      	  w2 = rg_src_w2[i2+ishift];
      	  i2_ = ix2+i2;
      	  for(i1=-radius+1; i1<=radius; i1++){
      	    w1 = sg_src_w1[i1+ishift];
      	    i1_ = ix1+i1;

      	    idx3d = i1_+n1pad*(i2_+n2pad*i3_);
      	    curlE2[idx3d] -= s*w1*w2*w3;
      	  }
      	}
      }
    }else if(chsrc[ic]==6){//(strcmp(chsrc[ic],"Hz") == 0){
      /* staggered grid: H3[i1,i2,i3] = Hz[i1,i2,i3+0.5] */
      ix1 = sg_src_i1[isubsrc];
      ix2 = sg_src_i2[isubsrc];
      ix3 = rg_src_i3[isubsrc];
      for(i3=-radius+1; i3<=radius; i3++){
      	w3 = rg_src_w3[i3+ishift];
      	i3_ = ix3+i3;
      	for(i2=-radius+1; i2<=radius; i2++){
      	  w2 = sg_src_w2[i2+ishift];
      	  i2_ = ix2+i2;
      	  for(i1=-radius+1; i1<=radius; i1++){
      	    w1 = sg_src_w1[i1+ishift];
      	    i1_ = ix1+i1;

      	    idx3d = i1_+n1pad*(i2_+n2pad*i3_);
      	    curlE3[idx3d] -= s*w1*w2*w3;
      	  }
      	}
      }

    }//end if Hx/Hy/Hz

  }//end if id

}

