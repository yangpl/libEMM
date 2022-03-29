#ifndef acqui_h
#define acqui_h

typedef struct {
  int nsrc;/* number of sources on each processor, default=1 */
  int nrec;/* number of receivers on each processor */
  int nsubsrc;/* number of distributed subpoints for each source */
  int nsubrec;/* number of distributed subpoints for each receiver */
  float lensrc;/* length of source antenna */
  float lenrec; /* length of receiver antenna */
  float *src_x1, *src_x2, *src_x3, *src_azimuth, *src_dip;
  float *rec_x1, *rec_x2, *rec_x3, *rec_azimuth, *rec_dip;
  float x1min, x1max, x2min, x2max, x3min, x3max;/* coordinate bounds */

  int nsrc_total;
  int nrec_total;
  
  int *shot_idx;
} acqui_t;/* type of acquisition geometry */

#endif
