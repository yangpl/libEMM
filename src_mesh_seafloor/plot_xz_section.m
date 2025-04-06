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

nx = length(x1);
ny = length(x2);
nz = length(x3);
V=reshape(rho, [nx ny nz]);

ix = int32(length(x1)/2);
iy = int32(length(x2)/2);
iz = 58; %63, 54


figure(1),clf

[xx, zz] = meshgrid(x1, x3);
surf(xx, zz, squeeze(V(:,iy,:)).'); shading flat %interp

set(gca,'Ydir','reverse')
view(0,90)
colormap(jet(2000))

xlim([-10000,10000])
xlabel('X (m)')
ylabel('Z (m)')
%axis tight
set(gca, 'Position', [0.1 0.1 0.9 0.8])

%text(0,2500, sprintf('(b) Y=%g m',x2(iy)))
caxis([0.3 10])
colorbar()
grid off



print -dpng rhoxz.png
