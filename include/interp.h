#ifndef interp_h
#define interp_h

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

#endif
