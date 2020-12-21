function [Sab,Hab] = covmat(ta,tb,sigma,phi);
    TTa = size(ta,1);
    TTb = size(tb,1);
    Hab =abs(ta*ones(1,TTb)-ones(TTa,1)*tb');
    Sab = sigma^2*(1+phi*Hab).*exp(-phi*Hab);
end