% Chapter 15 Clustering Analysis
% Apply the K-means and GMM method to the 4-D Iris dataset.
% Chapter 15 Clustering Analysis
clear, close all
global xmin xmax

rng(1)

[X,Ki]=Irisdata;
[N,d]=size(X);

K=3;                    % number of clusters

xmin=min(X)';
xmax=max(X)';
dx=xmax-xmin;
mk = zeros(d,K);
for k=1:K
    mk(:,k)=xmin+rand*dx;        % initial means
end

figure(1)
Kmeans(X,Ki,K,mk)                % Kmeans

figure(2)
GMM(X,Ki,K,mk)                   % GMM

%% 1) Kmeans
function Kmeans(X,Ki,K,mk)
[N,d]=size(X);
id=zeros(N,1);
Ck=zeros(d,d,K);
for k=1:K
    Ck(:,:,k)=eye(d);
end
done=0;
it=0;
while ~done
    it=it+1;
    ip=12;
    if it<12
        ip=it;
    end
    subaxis(3,4,ip, 'Spacing', 0.01, 'Padding', 0.01, 'Margin', 0.01);
    myPlot(X,Ki,mk,Ck);
    Nk=zeros(K,1);
    Mk=zeros(d,K);
    for n=1:N
        x=X(n,:)';
        dmin=inf;
        for k=1:K
            dist=norm(x-mk(:,k));
            if dist<dmin
                dmin=dist;
                ik=k;
            end
        end
        Nk(ik)=Nk(ik)+1;
        Mk(:,ik)=Mk(:,ik)+x;
        id(n)=ik;
    end
    er=0;
    for k=1:K
        if Nk(k)>0
            % your code here to compute mean Mk
            Mk(:,k)=Mk(:,k)/Nk(k);
        end
        er=er+norm(mk(:,k)-Mk(:,k));
    end
    if er<10^(-4)
        done=1;
    else
        mk=Mk;
    end
end
subtitle('K-means method')
K0=length(Ki);          % ground truth number of clusters
confussion=zeros(K0,K);
kmax=0;
n=0;
for k=1:K0
    for i=1:Ki(k)
        n=n+1;
        confussion(k,id(n))=confussion(k,id(n))+1;
    end
    kmax=kmax+max(confussion(k,:));
end
fprintf("Iteration times = %d\n",it)
fprintf("Cm = \n")
disp(confussion)
fprintf('Percentage error: %.2f%%\n\n',100*(1-kmax/N))
end

%% 2) GMM
function GMM(X,Ki,K,mk)
[N,d]=size(X);
Mk=zeros(d,K);
Ck=zeros(d,d,K);
P=zeros(N,K);
rk=zeros(N,K);
pk=zeros(K,1);
Nk=zeros(1,K);
xmin=min(X);
xmax=max(X);
m=200;
dx=(xmax(1)-xmin(1))/m;
n=ceil((xmax(2)-xmin(2))/dx);
for k=1:K
    Ck(:,:,k)=eye(d);
    pk(k)=1/K;
end
done=0;
it=0;
while ~done
    it=it+1;
    ip=12;
    if it<12
        ip=it;
    end
    subaxis(3,4,ip, 'Spacing', 0.01, 'Padding', 0.01, 'Margin', 0.01);
    myPlot(X,Ki,mk,Ck);
    for k=1:K
        if det(Ck(:,:,k))>10^(-9)       % Ck is full rank
            % your code here to compute weighted posterier P
            P(:,k)=pk(k)*mvnpdf(X,mk(:,k)',Ck(:,:,k));
        else
            P(:,k)=0;
        end
    end
    er=0;
    D=sum(P,2);
    for k=1:K
        rk(:,k)=P(:,k)./D;
        Nk(k)=sum(rk(:,k));
        pk(k)=Nk(k)/N;
        if Nk(k)>0
            % your code here to compute mean Mk and covariance Ck
            Mk(:,k)=sum(X.*repmat(rk(:,k),[1,d]),1)/Nk(k);
            Xm=X-repmat(Mk(:,k)',[N,1]);
            Ck(:,:,k)=Xm'*(Xm.*repmat(rk(:,k),[1,d]))/Nk(k);
        end
        er=er+norm(mk(:,k)-Mk(:,k));
    end
    if er<10^(-4)
        done=1;
    else
        mk=Mk;
    end
end
subtitle('EM method')
for n=1:N
    [~,id(n)]=max(rk(n,:));
end
K0=length(Ki);
confussion=zeros(K0,K);
kmax=0;
n=0;
for k=1:K0
    for i=1:Ki(k)
        n=n+1;
        confussion(k,id(n))=confussion(k,id(n))+1;
    end
    kmax=kmax+max(confussion(k,:));
end
fprintf("Iteration times = %d\n",it)
fprintf("Cm = \n")
disp(confussion)
fprintf('Percentage error: %.2f%%\n\n',100*(1-kmax/N))
end

%% Utilities
function colors=colorTable
colors=zeros(9,3);
colors(1,:)=[1 0 0];
colors(2,:)=[0 1 0];
colors(3,:)=[0 0 1];
colors(4,:)=[1 1 0];
colors(5,:)=[0 1 1];
colors(6,:)=[1 0 1];
colors(7,:)=[.3 0 .7];
colors(8,:)=[0 .3 .7];
colors(9,:)=[.7 .3 0];
colors(10,:)=[.3 .3 .3];
end

function [X,Ki]=Irisdata
   data_path = 'iris.txt';
   data = load(data_path);
   X = data(:,1:4);
   Ki = [50 50 50];
   end

function myPlot(X,Ki,mk,Ck)
[N,d]=size(X);
K0=length(Ki);
K=size(mk,2);
if d>2
    [V,D]=eig(cov(X));
    T=V(:,end:-1:1);
    X=X*T;
    if exist('mk') && exist('Ck')
        mk=T'*mk;
        for k=1:K
            Ck2(:,:,k)=T'*Ck(:,:,k);
        end
    end
end
xmin=min(X);
xmax=max(X);
m=200;
dx=(xmax(1)-xmin(1))/m;
n=ceil((xmax(2)-xmin(2))/dx);
colors=colorTable;
Xcolors=[];
for k=1:K0
    Xcolors=[Xcolors; repmat(colors(k,:),[Ki(k),1])];
end
scatter(X(:,1),X(:,2),5,Xcolors);
if exist('mk') && exist('Ck')
    hold on
    scatter(mk(1,:),mk(2,:),30,'k','filled');
    hold on
    w=[0.005 0.02 0.05 0.1 0.3 0.6 0.9];
    %        w=[2];
    u=linspace(xmin(1)-4*dx,xmax(1)+4*dx,m);
    v=linspace(xmin(2)-4*dx,xmax(2)+4*dx,n);
    [U,V]=meshgrid(u,v);
    for k=1:K
        if det(Ck(1:2,1:2,k))>10^(-9)
            f=mvnpdf([U(:) V(:)],mk(1:2,k)',Ck(1:2,1:2,k));
        end
        y=reshape(f,length(v),length(u));
        contour(u,v,y,w,'red'); hold on
    end
    hold off
end
axis equal
end
