%% Project 3
% Task G
clear all;close all;clc

%Model data
mu = 50;
sigma = 4;
phiM = 0.2;

t = linspace(10,80,141)';

%Experimental data
tb = [19.4, 29.7, 36.1,50.7, 71.9,40.7]'; % last point is the additional observation for the augmented dataset.
xb = [50.1,39.1,54.7,42.1,40.9,49.7]'; 

% compute mean vectors and covariance matrices
% of the process at the 141 points

[S,H] = covmat(t,t,sigma,phiM);
L = chol(S)';
realizations = 30;
z = randn(141,realizations);
x = mu*ones(1,realizations) + L*z;

figure(1)
imagesc(H)
colorbar
title('Distance matrix')
fig = gcf;
set(fig,'position',[0,0,560,420])


figure(2)
imagesc(S)
title('Covariance matrix')
colorbar
fig = gcf;
set(fig,'position',[0,0,560,420])


figure(3)
plot(t,x,'k-')
title('Unconditional realizations'), xlabel('temperature')
ylabel('Product quality')
fig = gcf;
set(fig,'position',[0,0,560,420])

% Using covmat.m to calculate covariance-matrices
Sa = covmat(t,t,sigma,phiM);
Sb = covmat(tb,tb,sigma,phiM);
Sab = covmat(t,tb,sigma,phiM);
Sba = covmat(tb,t,sigma,phiM);

mub = mu*ones(numel(xb),1);

%calculating conditional mean and covariance
mcond = mu*ones(141,1) + Sab*(Sb\(xb-mub));
Scond = Sa-Sab*(Sb\Sba);
Lcond = chol(Scond)';
Xcond = mcond*ones(1,realizations) + Lcond*randn(141,realizations);

figure(4)
plot(t,Xcond,'k-','handlevisibility','off')
title('Conditional realizations'), xlabel('temperature')
ylabel('Product quality')
hold on
for itit=1:size(tb,1),
   plot([tb(itit) tb(itit)],[35 xb(itit)],'b-','handlevisibility','off') 
end
plot(tb,xb,'bo','markersize',5)
legend('Experimental')
fig = gcf;
set(fig,'position',[0,0,560,420])
%saveas(fig,'G1condreal3.png')
hold off

figure(5)
plot(t,mcond,'k-')
hold on
plot(tb,xb,'bo','markersize',5)
plot(t,mcond-1.64*sqrt(diag(Scond)),'r--','handlevisibility','off');
plot(t,mcond+1.64*sqrt(diag(Scond)),'r--');

for itit=1:size(tb,1),
   plot([tb(itit) tb(itit)],[35 xb(itit)],'b-','handlevisibility','off') 
end

hold off
title('Conditional mean with 90% prediction intervals')
xlabel('Temperature'), ylabel('Product quality')
legend('Conditional mean','Experimental','Prediction interval')
fig = gcf;
set(fig,'position',[0,0,560,420])
%saveas(fig,'G1903.png')


% P(X(t) > 57)
cumprob = normcdf(57,mcond,sqrt(diag(Scond)));
probabove57 = 1-cumprob;

figure(6)
plot(t,probabove57(:,1))
title('Probability of product quality greater than 57')
xlabel('Temperature')
ylabel('P(X(t) > 57)')

fig = gcf;
set(fig,'position',[0,0,560,420])
%saveas(fig,'Gprob.png')

