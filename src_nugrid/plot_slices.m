graphics_toolkit('gnuplot')
set(0, "defaulttextfontname", "Helvetica")
set(0, "defaultaxesfontname", "Helvetica")

clear all

fid = fopen('x1nu');
x1 = fread(fid, 'float');
fclose(fid);

fid = fopen('x2nu');
x2 = fread(fid, 'float');
fclose(fid);

fid = fopen('x3nu');
x3 = fread(fid, 'float');
fclose(fid);

fid = fopen('rho11');
rho = fread(fid, 'float');
fclose(fid);




[xx, yy, zz] = meshgrid(x1, x2, x3);
nx = length(x1);
ny = length(x2);
nz = length(x3);
V=reshape(rho, [nx ny nz]);

xslice = 0;
yslice = 0;
zslice = 2100;

figure,clf
slice(xx, yy, zz, V, xslice, yslice, zslice);%shading flat
set(gca,'Zdir','reverse')
%light('Position',[-1 0 0],'Style','local')
lighting gouraud %flat
xlabel('X[m]')
ylabel('Y[m]')
zlabel('Z[m]')
print -dpng figure.png
