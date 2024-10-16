% Chapter 14 Support Vector Machine
clear, close all
global xmin xmax ymin ymax
[X,y]=myData;        % get the training data

% your code here to define variables for the QP problem
k=size(X,2);
Q=(y'*y).*(X'*X);
A=y;
c=-ones(k,1);
b=0;

[alpha,mu,lambda]=IPmethod(Q,A,c,b,alpha);
                     % solve QP for alpha (interior point method)
% your code here do compute support vectors Xsv, ysv and w, bias
Xsv=X(:,I);
ysv=y(:,I);
w=sum(repmat(ysv.*asv,m,1).*Xsv,2); 
bias=mean(ysv-w'*Xsv);

b = zeros(1,k);
x=xmin:0.01:xmax;
fprintf("Support vectors:\n")
for i=1:k
    b(i)=ysv(i)-w'*Xsv(:,i);
    fprintf('%d\talpha=%.2f\tx=[%.2f,%.2f]\ty=%d\n',i,asv(i),Xsv(1,i),Xsv(2,i),ysv(i));
    if ysv(i)>0
        scatter(Xsv(1,i),Xsv(2,i),'b','filled');
    else
        scatter(Xsv(1,i),Xsv(2,i),'r','filled');
    end
    plot(x,-(b(i)+w(1)*x)./w(2)); hold on
    plot(x,-(b(i)+1+w(1)*x)./w(2),'r'); hold on
    plot(x,-(b(i)-1+w(1)*x)./w(2),'b'); hold on
end
fprintf('w = [%.2f,%.2f],\tb = %.2f\n',w(1),w(2),bias)
xlim([xmin xmax])
ylim([ymin ymax])
hold off

function [x,mu,lambda]=IPmethod(Q,A,c,b,x)
% your code here to implete the interior point method

end

function [X,y]=myData
global xmin xmax ymin ymax
rng(2)
K0=60;  K1=60;
m0=[3.5; 0]; m1=[0; 3.5];
%    m0=[2.8; 0]; m1=[0; 2.8];
S0=eye(2);   S0(1,2)=0.5; S0(2,1)=0.5;
S1=eye(2);

X0=mvnrnd(m0,S0,K0)';
X1=mvnrnd(m1,S1,K1)';
y0=-ones(1,K0);
y1=ones(1,K1);
X=[X0 X1];
y=[y0 y1];
xmax=max(X(1,:));
xmin=min(X(1,:));
ymax=max(X(2,:));
ymin=min(X(2,:));
dx=xmax-xmin;
dy=ymax-ymin;
M=50;
d=dx/M;
N=round(dy/d);

v=linspace(xmin-dx,xmax+dx,3*M);
u=linspace(ymin-dy,ymax+dy,3*N);
[U,V]=meshgrid(u,v);
z=mvnpdf([U(:) V(:)],m0',S0);
z0=reshape(z,length(v),length(u));
z=mvnpdf([U(:) V(:)],m1',S1);
z1=reshape(z,length(v),length(u));

figure(1)
w=[0.01 0.025 0.05 0.1 0.15 0.2 0.25];
contour(u,v,z0,w); hold on
contour(u,v,z1,w); hold on
scatter(X0(1,:),X0(2,:),'r'); hold on
scatter(X1(1,:),X1(2,:),'b'); hold on
xlim([xmin xmax])
ylim([ymin ymax])
end
