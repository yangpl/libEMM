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

% fid = fopen('rho11');
% rho = fread(fid, 'float');
% fclose(fid);

% nx = length(x1);
% ny = length(x2);
% nz = length(x3);
% V=reshape(rho, [nx ny nz]);



[xx, zz] = meshgrid(x1, x3);
%topo = load('topo.txt');
% rec = load('receivers.txt');
% src = load('sources.txt');
Lx = x1(end)-x1(1);
topo = 600. + 100*(sin(2.*pi*1.5*x1/Lx) + 0.5*sin(2.*pi*2.5*x1/Lx));

figure(2),clf
plot(xx, zz, 'k')
hold on
plot(xx.', zz.', 'k')
hold on
plot(x1, topo, 'b')
% hold on
% plot(rec(:,1), rec(:,3),'b.')
% hold on
% plot(src(:,1), src(:,3),'r*')


set(gca,'Ydir','reverse')
axis tight
xlabel('X (m)')
ylabel('Z (m)')

#print -deps mesh.eps
print -dpng mesh.png



