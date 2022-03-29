/* generate nonuniform (NU) grid and perform medium homogenization
 * Reference: 
 *    [1] Davydycheva, S., V. Druskin, and T. Habashy, 2003, An efficient 
 *        finite-difference scheme for electromagnetic logging in 3D 
 *        anisotropic inhomogeneous media: Geophysics, 68, 1525–1536
 *    [2] Pengliang Yang and Rune Mittet, 2022, Controlled-source 
 *        electromagnetics modelling using high order finite-difference 
 *        time-domain method on a nonuniform grid, Geophysics
 *--------------------------------------------------------------------
 *
 *   Copyright (c) 2020, Harbin Institute of Technology, China
 *   Author: Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *--------------------------------------------------------------------*/
#include "cstd.h"

#define PI 3.141592653589793238462643

float create_nugrid(int n, float len, float dx, float *x);

/*----------------------------------------------------------------*/
void create_1d_model()
{
  int iz, ix, iy;
  float zi, r, tmp;
  FILE *fp;

  float xmin = -10000;
  float xmax = 10000;
  float ymin = -10000;
  float ymax = 10000;
  float zmin = 0;
  float zmax = 5000;
  
  int nx = 101;
  int ny = 101;
  int nz = 101;

  float dx = 200;
  float dy = 200;
  float dz = 50;

  float *x1nu = alloc1float(nx);
  float *x2nu = alloc1float(ny);
  float *x3nu = alloc1float(nz);

  /*----------------------------------------------------*/
  /* step 1: generate NU grid coordinates along x and y */
  /*----------------------------------------------------*/
  float xs1 = 0;
  float xs2 = 0;
  int n1left = (xs1-xmin)/(xmax-xmin)*(nx-1) +1;
  int n2left = (xs2-ymin)/(ymax-ymin)*(ny-1) +1;
  int n1right = nx - n1left+1;
  int n2right = ny - n2left+1;

  float *x1left, *x1right, *x2left, *x2right;
  
  x1left = alloc1float(n1left);
  x2left = alloc1float(n2left);
  x1right = alloc1float(n1right);
  x2right = alloc1float(n2right);

  r = create_nugrid(n1left-1, xs1-xmin, dx, x1left);
  printf("r1left=%g\n", r);
  r = create_nugrid(n2left-1, xs2-ymin, dy, x2left);  
  printf("r2left=%g\n", r);
  r = create_nugrid(n1right-1, xmax-xs1, dx, x1right);
  printf("r1right=%g\n", r);
  r = create_nugrid(n2right-1, ymax-xs2, dy, x2right);
  printf("r2right=%g\n", r);

  
  for(ix=0; ix<n1left; ix++)    x1nu[n1left-1-ix] = xs1 - x1left[ix];
  for(ix=0; ix<n1right; ix++)   x1nu[n1left-1+ix] = xs1 + x1right[ix];
  for(iy=0; iy<n2left; iy++)    x2nu[n2left-1-iy] = xs2 - x2left[iy];
  for(iy=0; iy<n2right; iy++)   x2nu[n2left-1+iy] = xs2 + x2right[iy];
  for(iz=0; iz<nz; iz++)        x3nu[iz] = iz*dz + zmin;

  /* reset the end points */
  x1nu[0] = xmin;
  x1nu[nx-1] = xmax;
  x2nu[0] = ymin;
  x2nu[ny-1] = ymax;
  x3nu[0] = zmin;
  x3nu[nz-1] = zmax;


  /* output NU grid for each axis */
  fp = fopen("x1nu", "wb");
  fwrite(x1nu, nx*sizeof(float), 1, fp);
  fclose(fp);

  fp = fopen("x2nu", "wb");
  fwrite(x2nu, ny*sizeof(float), 1, fp);
  fclose(fp);

  fp = fopen("x3nu", "wb");
  fwrite(x3nu, nz*sizeof(float), 1, fp);
  fclose(fp);
  
  /*-----------------------------------------------*/
  /* step2: asign values to rho11, rho22 and rho33 */
  /*-----------------------------------------------*/
  float ***rho = alloc3float(nx, ny, nz);
  float ***rho11 = alloc3float(nx, ny, nz);
  float ***rho22 = alloc3float(nx, ny, nz);
  float ***rho33 = alloc3float(nx, ny, nz);
  for(iz=0; iz<nz; iz++){
    for(iy=0; iy<ny; iy++){
      for(ix=0; ix<nx; ix++){
	zi = x3nu[iz];
	/*--------------------------------------------------*/
	if(zi<825) tmp = 0.3125;
	else if(zi<1525) tmp = 1.5;
	else if(zi<1625) tmp = 50;
	else tmp = 2;
	rho[iz][iy][ix] = tmp;
      }
    }
  }
  
  /*------- homogenization, easier for uniform grid ----*/
  for(iz=0; iz<nz; iz++){
    for(iy=0; iy<ny; iy++){
      for(ix=0; ix<nx-1; ix++){
	tmp = 0.5*(1./rho[iz][iy][ix] + 1./rho[iz][iy][ix+1]);
	rho11[iz][iy][ix] = 1./tmp;
      }
      ix = nx-1;
      rho11[iz][iy][ix] = rho[iz][iy][ix];
    }
  }

  for(iz=0; iz<nz; iz++){
    for(ix=0; ix<nx; ix++){
      for(iy=0; iy<ny-1; iy++){
	tmp = 0.5*(1./rho[iz][iy][ix] + 1./rho[iz][iy+1][ix]);
	rho22[iz][iy][ix] = 1./tmp;
      }
      iy = ny-1;
      rho22[iz][iy][ix] = rho[iz][iy][ix];
    }
  }
  
  for(iy=0; iy<ny; iy++){
    for(ix=0; ix<nx; ix++){
      for(iz=0; iz<nz-1; iz++){
	rho33[iz][iy][ix] = 0.5*(rho[iz][iy][ix] + rho[iz+1][iy][ix]);
      }
      rho33[nz-1][iy][ix] = rho[nz-1][iy][ix];
    }
  }

  /*------------ output the models -----------------*/
  fp = fopen("rho11", "wb");
  fwrite(&rho11[0][0][0], nz*nx*ny*sizeof(float), 1, fp);
  fclose(fp);
  fp = fopen("rho22", "wb");
  fwrite(&rho22[0][0][0], nz*nx*ny*sizeof(float), 1, fp);
  fclose(fp);
  fp = fopen("rho33", "wb");
  fwrite(&rho33[0][0][0], nz*nx*ny*sizeof(float), 1, fp);
  fclose(fp);

  free3float(rho);
  free3float(rho11);
  free3float(rho22);
  free3float(rho33);

  free1float(x1nu);
  free1float(x2nu);
  free1float(x3nu);

}


/*----------------------------------------------------------------*/
void create_3d_model_with_bathymetry()
{
  int iz, ix, iy, i, nx_, ny_, nz_;
  float xi, yi, zi, wd, tmp, sx;
  float r;
  FILE *fp;

  /* dimension of the physical domain for modelling */
  float xmin = -10000;
  float xmax = 10000;
  float ymin = -10000;
  float ymax = 10000;
  float zmin = 0;
  float zmax = 4000;
  float Lx = xmax-xmin;

  /* size of the model and grid spacing on coarse grid */
  int nx = 101;
  int ny = 101;
  int nz = 101;
  float dx = 200;
  float dy = 200;
  float dz = 20;

  float *x1nu = alloc1float(nx);
  float *x2nu = alloc1float(ny);
  float *x3nu = alloc1float(nz);

  /*--------------------------------------------------------------*/
  /* step 1: generate NU grid coordinates along x, y, z           */
  /*  NU grid for x and y are suppressed due to degrated accuracy */
  /*--------------------------------------------------------------*/
  float xs1 = 0;
  float xs2 = 0;
  int n1left = (xs1-xmin)/(xmax-xmin)*(nx-1) +1;
  int n2left = (xs2-ymin)/(ymax-ymin)*(ny-1) +1;
  int n1right = nx - n1left+1;
  int n2right = ny - n2left+1;

  float *x1left = alloc1float(n1left);
  float *x2left = alloc1float(n2left);
  float *x1right = alloc1float(n1right);
  float *x2right = alloc1float(n2right);
  r = create_nugrid(n1left-1, xs1-xmin, dx, x1left);
  printf("r1left=%g\n", r);
  r = create_nugrid(n2left-1, xs2-ymin, dy, x2left);  
  printf("r2left=%g\n", r);
  r = create_nugrid(n1right-1, xmax-xs1, dx, x1right);
  printf("r1right=%g\n", r);
  r = create_nugrid(n2right-1, ymax-xs2, dy, x2right);
  printf("r2right=%g\n", r);

  
  for(ix=0; ix<n1left; ix++)    x1nu[n1left-1-ix] = xs1 - x1left[ix];
  for(ix=0; ix<n1right; ix++)   x1nu[n1left-1+ix] = xs1 + x1right[ix];
  for(iy=0; iy<n2left; iy++)    x2nu[n2left-1-iy] = xs2 - x2left[iy];
  for(iy=0; iy<n2right; iy++)   x2nu[n2left-1+iy] = xs2 + x2right[iy];
  for(iz=0; iz<nz; iz++)        x3nu[iz] = iz*dz + zmin;
  
  float zstretch = 1200;
  iz = zstretch/dz;
  int nh = nz-iz;
  float *x3right = alloc1float(nh);
  r = create_nugrid(nh-1, zmax-zstretch, dz, x3right);
  printf("stretching factor=%g\n", r);
  for(i=0; i<nh; i++) {
    x3nu[iz+i] = x3nu[iz] + x3right[i];//over write uniform grid by nonuniform grid
    //printf("z=%g\n", x3nu[iz+i]);
  }

  /* reset the end points */
  x1nu[0] = xmin;
  x1nu[nx-1] = xmax;
  x2nu[0] = ymin;
  x2nu[ny-1] = ymax;
  x3nu[0] = zmin;
  x3nu[nz-1] = zmax;

  /* output NU grid for each axis */
  fp = fopen("x1nu", "wb");
  fwrite(x1nu, nx*sizeof(float), 1, fp);
  fclose(fp);

  fp = fopen("x2nu", "wb");
  fwrite(x2nu, ny*sizeof(float), 1, fp);
  fclose(fp);

  fp = fopen("x3nu", "wb");
  fwrite(x3nu, nz*sizeof(float), 1, fp);
  fclose(fp);

  /* grid spacing and size of model on fine grid */
  float dx_ = 10;
  float dy_ = 10;
  float dz_ = 2;
  nx_ = (xmax-xmin)/dx_ + 1;
  ny_ = (ymax-ymin)/dy_ + 1;
  nz_ = (zmax-zmin)/dz_ + 1;
  printf("nx_=%d\n", nx_);
  printf("ny_=%d\n", ny_);
  printf("nz_=%d\n", nz_);
  printf("dx_=%g\n", dx_);
  printf("dy_=%g\n", dy_);
  printf("dz_=%g\n", dz_);

  /*-----------------------------------------------*/
  /* step2: asign values to rho11, rho22 and rho33 */
  /*-----------------------------------------------*/
  float ***intx_sigma = alloc3float(nx_, ny, nz);//integral_x rho(x)dx
  float ***inty_sigma = alloc3float(nx, ny_, nz);//integral_y rho(y)dy
  float ***intz_rho = alloc3float(nx, ny, nz_);//integral_z rho(z)dz
  float ***rho11 = alloc3float(nx, ny, nz);
  float ***rho22 = alloc3float(nx, ny, nz);
  float ***rho33 = alloc3float(nx, ny, nz);
  float s;
  int ix_, iy_, iz_;
  int ix_m1, iy_m1, iz_m1;

  //======================================================
  printf("-----------homogenization for rho11----------\n");
  for(iz=0; iz<nz; iz++){
    zi = x3nu[iz];
    for(iy=0; iy<ny; iy++){
      yi = x2nu[iy];
      for(ix=0; ix<nx_; ix++){
	xi = xmin + ix*dx_;

	sx = sin(2.*PI*1.5*xi/Lx) + 0.5*sin(2*PI*2.5*xi/Lx);
	wd = 600 + 100.*sx;
	/*--------------------------------------------------*/
	if(zi<=wd) tmp = 0.3;
	else if(zi<=1000) tmp = 1;
	else tmp = 5;
	if(zi>1800 && zi<=2000 && fabs(xi)<=3000) tmp = 100.;
	
	intx_sigma[iz][iy][ix] = 1./tmp;
      }
    }
  }
  printf("sigma11 created!\n");

  for(iz=0; iz<nz; iz++) {
    for(iy=0; iy<ny; iy++){
      s = 0;
      for(ix=0; ix<nx_; ix++){//integral along x direction
	s += intx_sigma[iz][iy][ix];
	intx_sigma[iz][iy][ix] = s;
      }
    }
  }
  printf("integration along x, done!\n");

  for(iz=0; iz<nz; iz++){
    for(iy=0; iy<ny; iy++){
      for(ix=0; ix<nx; ix++){
	xi = 0.5*(x1nu[ix] + x1nu[MIN(ix+1,nx-1)])-xmin;//current grid point
	ix_ = MIN(NINT(xi/dx_), nx_-1);
	
	xi = 0.5*(x1nu[ix] + x1nu[MAX(ix-1,0)])-xmin;//previous grid point 
	ix_m1 = MIN(NINT(xi/dx_), nx_-1);

	//average over sigma_h
	tmp = (intx_sigma[iz][iy][ix_]-intx_sigma[iz][iy][ix_m1])/(ix_-ix_m1);
	rho11[iz][iy][ix] = 1./tmp;
      }
    }
  }
  printf("homogenization for rho11, done!\n");

  //======================================================
  printf("-----------homogenization for rho22----------\n");
  for(iz=0; iz<nz; iz++){
    zi = x3nu[iz];
    for(iy=0; iy<ny_; iy++){
      yi = ymin + iy*dy_;
      for(ix=0; ix<nx; ix++){
	xi = x1nu[ix];

	sx = sin(2.*PI*1.5*xi/Lx) + 0.5*sin(2*PI*2.5*xi/Lx);
	wd = 600 + 100.*sx;
	/*--------------------------------------------------*/
	if(zi<=wd) tmp = 0.3;
	else if(zi<=1000) tmp = 1;
	else tmp = 5;
	if(zi>1800 && zi<=2000 && fabs(xi)<=3000) tmp = 100.;
	
	inty_sigma[iz][iy][ix] = 1./tmp;
      }
    }
  }
  printf("sigma22 created!\n");

  for(iz=0; iz<nz; iz++) {
    for(ix=0; ix<nx; ix++){
      s = 0;
      for(iy=0; iy<ny_; iy++){//integral along y direction
	s += inty_sigma[iz][iy][ix];
	inty_sigma[iz][iy][ix] = s;
      }
    }
  }  
  printf("integration along y, done!\n");

  for(iz=0; iz<nz; iz++){
    for(iy=0; iy<ny; iy++){
      yi = 0.5*(x2nu[iy] + x2nu[MIN(iy+1,ny-1)])-ymin;
      iy_ = MIN(NINT(yi/dy_), ny_-1);
      yi = 0.5*(x2nu[iy] + x2nu[MAX(iy-1,0)])-ymin;
      iy_m1 = MIN(NINT(yi/dy_), ny_-1);
      for(ix=0; ix<nx; ix++){

	//average over sigma_h
	tmp = (inty_sigma[iz][iy_][ix]-inty_sigma[iz][iy_m1][ix])/(iy_-iy_m1);
	rho22[iz][iy][ix] = 1./tmp;
      }
    }
  }
  printf("homogenization for rho22, done!\n");

  //======================================================
  printf("-----------homogenization for rho33----------\n");
  for(iz=0; iz<nz_; iz++){
    zi = zmin + iz*dz_;
    for(iy=0; iy<ny; iy++){
      yi = x2nu[iy];
      for(ix=0; ix<nx; ix++){
	xi = x1nu[ix];

	sx = sin(2.*PI*1.5*xi/Lx) + 0.5*sin(2*PI*2.5*xi/Lx);
	wd = 600 + 100.*sx;
	/*--------------------------------------------------*/
	if(zi<=wd) tmp = 0.3;
	else if(zi<=1000) tmp = 1;
	else tmp = 5;
	if(zi>1800 && zi<=2000 && fabs(xi)<=3000) tmp = 100.;

	intz_rho[iz][iy][ix] = tmp;
      }
    }
  }
  printf("rho33 created!\n");

  for(iy=0; iy<ny; iy++){
    for(ix=0; ix<nx; ix++){
      s = 0;
      for(iz=0; iz<nz_; iz++) {//integral along z direction
	s += intz_rho[iz][iy][ix];
	intz_rho[iz][iy][ix] = s;
      }
    }
  }
  printf("integration along z, done!\n");

  for(iz=0; iz<nz; iz++){
    zi = 0.5*(x3nu[iz]+x3nu[MIN(iz+1,nz-1)])-zmin;
    iz_ = MIN(NINT(zi/dz_), nz_-1);
    zi = 0.5*(x3nu[iz]+x3nu[MAX(iz-1,0)])-zmin;
    iz_m1 = MIN(NINT(zi/dz_), nz_-1);
    for(iy=0; iy<ny; iy++){
      for(ix=0; ix<nx; ix++){

	//average over rho_v
	tmp = (intz_rho[iz_][iy][ix]-intz_rho[iz_m1][iy][ix])/(iz_-iz_m1);
	rho33[iz][iy][ix] = tmp;
      }
    }
  }
  printf("homogenization for rho33, done!\n");

  
  /* output the models */
  fp = fopen("rho11", "wb");
  fwrite(&rho11[0][0][0], nz*nx*ny*sizeof(float), 1, fp);
  fclose(fp);
  fp = fopen("rho22", "wb");
  fwrite(&rho22[0][0][0], nz*nx*ny*sizeof(float), 1, fp);
  fclose(fp);
  fp = fopen("rho33", "wb");
  fwrite(&rho33[0][0][0], nz*nx*ny*sizeof(float), 1, fp);
  fclose(fp);

  /* print out seafloor topography, top and bottom of resistor for MARE2DEM */
  fp = fopen("topo.txt", "w");
  dx = 10;
  nx = (xmax-xmin)/dx + 1;
  for(ix=0; ix<nx; ix++){
    xi = xmin + ix*dx;
    sx = sin(2.*PI*1.5*xi/Lx) + 0.5*sin(2*PI*2.5*xi/Lx);
    wd = 600 + 100.*sx;
    fprintf(fp, "%e \t %e\n", xi, wd);
  }
  fclose(fp);

  fp = fopen("resistor_top.txt", "w");
  for(ix=0; ix<nx; ix++){
    xi = xmin + ix*dx;
    zi = 1800;
    if(fabs(xi)<3000) fprintf(fp, "%e \t %e\n", xi, zi);
  }
  fclose(fp);

  fp = fopen("resistor_bottom.txt", "w");
  for(ix=0; ix<nx; ix++){
    xi = xmin + ix*dx;
    zi = 2000;
    if(fabs(xi)<3000) fprintf(fp, "%e \t %e\n", xi, zi);
  }
  fclose(fp);

  free3float(intx_sigma);
  free3float(inty_sigma);
  free3float(intz_rho);
  free3float(rho11);
  free3float(rho22);
  free3float(rho33);

  free1float(x1nu);
  free1float(x2nu);
  free1float(x3nu);

}

int main(int argc, char *argv[])
{
  int opt = 2;

  if(opt==1) create_1d_model();
  if(opt==2) create_3d_model_with_bathymetry();

}
