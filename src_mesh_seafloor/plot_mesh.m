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
topo = load('topo.txt');
% rec = load('receivers.txt');
% src = load('sources.txt');

figure(2),clf
plot(xx, zz, 'k')
hold on
plot(xx.', zz.', 'k')
hold on
plot(topo(:,1), topo(:,2), 'k-')
% hold on
% plot(rec(:,1), rec(:,3),'b.')
% hold on
% plot(src(:,1), src(:,3),'r*')


set(gca,'Ydir','reverse')
axis tight
xlabel('X[m]')
ylabel('Z[m]')

#print -deps mesh.eps
print -dpng mesh.png



