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
iz = 66; %63, 54


figure(1),clf

subplot(221)
[xx, yy] = meshgrid(x1, x2);
surf(xx, yy, V(:,:,iz).'); shading flat % interp
hold on
plot(x1, x2(iy)*ones(size(x1)), 'k')
plot(x1(ix)*ones(size(x2)), x2, 'k')


view(0,90)
colormap(jet(2000))
%colorbar('northoutside')
xlabel('X[m]')
ylabel('Y[m]')
axis tight
set(gca,'XTickLabel',[]);
set(gca, 'Position', [0.1 0.5 0.4 0.4])
caxis([0.3 20])
grid off


subplot(223)
[xx, zz] = meshgrid(x1, x3);
surf(xx, zz, squeeze(V(:,iy,:)).'); shading flat %interp
hold on
plot(x1(ix)*ones(size(x3)), x3, 'k')
plot(x1, x3(iz)*ones(size(x1)), 'k')

set(gca,'Ydir','reverse')
view(0,90)
colormap(jet(2000))

xlabel('X[m]')
ylabel('Z[m]')
axis tight
set(gca, 'Position', [0.1 0.1 0.4 0.4])
%text(0,2500, sprintf('(b) Y=%g m',x2(iy)))
caxis([0.3 20])
grid off

subplot(224)
[yy, zz] = meshgrid(x2, x3);
surf(yy, zz, squeeze(V(ix,:,:)).'); shading flat %interp
hold on
plot(x2(iy)*ones(size(x3)), x3, 'k')
plot(x2, x3(iz)*ones(size(x2)), 'k')


set(gca,'Ydir','reverse')
view(0,90)
colormap(jet(2000))
xlabel('Y[m]')
%ylabel('Z[m]')
axis tight
set(gca,'YTickLabel',[]);
set(gca, 'Position', [0.5 0.1 0.4 0.4])
caxis([0.3 20])
grid off

print -dpng rho3d.png
