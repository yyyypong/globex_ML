% Chapter 18 Perceptron
% You should solve the following problems.
% 1) Perform kernel perceptron on different datasets
% You may refer to pages 439-447 of our textbook.

close all;

rng(2023);
onehot=1;
%% Use different datasets in Perceptron algorithm, 1 for linear seperable 
% dataset, 2 for XOR dataset and 3 for Iris dataset
idata=input('Dataset: linear (1), XOR (2), Iris (3) :');
kernel = "RBF";
switch idata
    case 1
        [X y]=Data;             % linearly saperable data
    case 2
        [X y]=XOR;
    case 3
        [X,y]=IrisData;
end
[d N]=size(X);
K=length(unique(y));
fprintf('d=%d\tN=%d\tK=%d\n',d,N,K);
X=[ones(1,N); X];               % data augmentation
Y=Yencoder(y,onehot);

itrain=sort(randperm(N,N/2));   % random indices for half of the data
itest=setdiff(1:N,itrain);
Xtrain=X(:,itrain);    Xtest=X(:,itest);
Ytrain=Y(:,itrain);    Ytest=Y(:,itest);
ytrain=y(itrain);

A=kernelPerceptron(Xtrain,Ytrain,kernel);
ytest=testKernel(Xtest,Xtrain,Ytrain,A,kernel);

fprintf('Testing:\n')
Cm=zeros(K);
for n=1:length(ytrain)
    i=ytrain(n);
    j=ytest(n);
    Cm(i,j)=Cm(i,j)+1;
end

fprintf('Cm = \n')
disp(Cm)
fprintf('Error rate is: %.4f\n', 1-trace(Cm)/length(ytrain))


function K=RBFKernel(X,Y)
    sigma=1;
    m=size(X,2);
    n=size(Y,2);
    K=zeros(m,n);
    for i=1:m
        for j=1:n
            % Your code here to calculate RBF kernel
            K(i,j)=exp(-(norm(X(:,i)-Y(:,j)))^2/(2*sigma^2));  % RBF
        end
    end
end


function K=LinearKernel(X,Y)
    sigma=1;
    m=size(X,2);
    n=size(Y,2);
    K=zeros(m,n);
    for i=1:m
        for j=1:n
            % Your code here to calculate linear kernel
            K(i,j)=X(:,i)'*Y(:,j);  % linear kernel
        end
    end
end


function S=sgmd(X)
    S=2./(1+exp(-X))-1;     % sigmoid function between -1 and 1
end


function A=kernelPerceptron(X,Y,kernel)
    [d N]=size(X);
    m=size(Y,1);                % number of output nodes
    nt=10^5;                    % maximum number of iterations
    eta=1;
    ie=0;
    A=zeros(m,N);
    if kernel == "RBF"
        K=RBFKernel(X,X);    % RBFKernel matrix of data
    elseif kernel == "linear"
        K=LinearKernel(X,X);
    end
	for it=1:nt       
        i=randi([1 N]);         % random index
        x=X(:,i);               % pick a training sample     
        y=Y(:,i);               % and its label
        % Your code here to calculate yhat for the first iteration
        yhat=sign((A.*Y)*K(:,i));     
        idx=find(y-yhat~=0); 	% indecies of mismatched output nodes
        A(idx,i)=A(idx,i)+eta;    % update alphas for mismatched nodes
        if ~mod(it,N)           % test for every epoch
            ie=ie+1;
            if kernel == "RBF"
                K=RBFKernel(X,X);
            elseif kernel == "linear"
                K=LinearKernel(X,X);
            end
            % Your code here to calculate Yhat
            Yhat= sign((A.*Y)*K);
            [Cm er0 er1]=ConfusionMatrix(Y,Yhat);
            if er0<10^(-3) && er1<10^(-2)        
                break
            end
        end        
    end
end


function ytest=testKernel(Xtest,X,Y,A,kernel)
    N=size(Xtest,2);                % number of test samples
    Yunique=unique(Y','rows','stable')';     % unique class labels
    if kernel == "RBF"
        S=sgmd((A.*Y)*RBFKernel(X,Xtest));	% sigmoid functions of all test samples
    elseif kernel == "linear"
        S=sgmd((A.*Y)*LinearKernel(X,Xtest));
    end
    for n=1:N                       % for all test samples
        yhat=S(:,n);
        [d,k]=min(sum(abs(Yunique-yhat)));	% find class with min distance
        ytest(n)=k;                    % nth sample classified to kth class
    end                          
end


function [Cm er0 er1]=ConfusionMatrix(Y,Yhat)
    N=size(Y,2);                    % number of samples
    Yunique=unique(Y','rows','stable')';     % unique labels 
    K=length(Yunique);              % number of classes
    m=length(Yunique);              % number of output nodes
    Cm=zeros(K);
    No=0;
    for n=1:N
        [l i]=ismember(Y(:,n)',Yunique','rows');     % class of y
        [l j]=ismember(Yhat(:,n)',Yunique','rows');  % class of yhat
        if l                                
            Cm(i,j)=Cm(i,j)+1; 
        else 
            No=No+1;                % yhat not one of the K class
        end
    end
    er0=1-trace(Cm)/N;              % percentage misclassified
    er1=No/N;                       % percentage unclassified
end


function Visualize(X,Y)
    [d N]=size(X);
    Yunique=unique(Y','rows','stable')';    % unique labels 
    xmax=max(X(1,:));
    xmin=min(X(1,:));
    ymax=max(X(2,:));
    ymin=min(X(2,:));
    if N==3
        zmax=max(X(3,:));
        zmin=min(X(3,:));
    end      
    CT=colorTable;
    c=zeros(N,3);
    
    for n=1:N                       % for each of the N data points
        k=1;
        Y(:,n);
        while norm(Y(:,n)-Yunique(:,k))>0
            k=k+1;
        end
        color(n,:)=CT(k,:);
    end
    if d==2
        scatter(X(1,:),X(2,:),20*ones(1,n),color,'filled','o');
    elseif d==3
        scatter3(X(1,:),X(2,:),X(3,:),20*ones(1,n),color,'filled','o');
    end
    hold on
end   



function CT=colorTable
    CT=zeros(8,3);
    g=1;
    CT(1,:)=[g 0 0];
    CT(2,:)=[0 0 g];
    CT(3,:)=[0 g 0];
    CT(4,:)=[g g 0];
    CT(5,:)=[0 g g];
    CT(6,:)=[g 0 g];
    CT(7,:)=[0 0 0];
    CT(8,:)=[g/2 g/2 g/2];    
end


function [X,y]=Data
    d=3;                    % number of dimensions
    K=8;                    % number of classes
    Nk(1:K)=50;             % number of samples per class
    N=sum(Nk);              % total number of samples
    Means=[ -1 -1 -1 -1 1 1 1 1;
        -1 -1 1 1 -1 -1 1 1;
        -1 1 -1 1 -1 1 -1 1];    
    X=[];
    y=[];
    s=0.22;
    for k=1:K               % for each of the K classes
        Xk=Means(:,k)+s*randn(d,Nk(k));
        y=[y repelem(k,Nk(k))];
        X=[X Xk];
    end
    Visualize(X,y)   
end

function [X,y]=XOR
    k1=40;  k2=60; k3=60;  k4=30;
	m1=[2; 2]; 
    m2=[-2; -2];
    m3=[2; -2];
    m4=[-2; 2];    
    d=[99; 33];
    m1=m1+d;
    m2=m2+d;
    m3=m3+d;
    m4=m4+d;
    
    c=1.5;
    S1=c*eye(2); S2=c*eye(2); S3=c*eye(2); S4=c*eye(2);   
    X1=mvnrnd(m1,S1,k1)'; X2=mvnrnd(m2,S2,k2)'; X3=mvnrnd(m3,S3,k3)'; X4=mvnrnd(m4,S4,k4)';
    X=[X1 X2 X3 X4];
    y=[repelem(1,1,k1+k2) repelem(2,k3+k4)];
    X0=[X1 X2];     
    X1=[X3 X4];     

	[m n]=size(X); 

	xmax=max(X')';  xmin=min(X')';  
    dx=xmax(1)-xmin(1);
    dy=xmax(2)-xmin(2);
    M=100;
    d=dx/M;
    N=round(dy/d);

    u=linspace(xmin(1)-4*d,xmax(1)+4*d,M);
    v=linspace(xmin(2)-4*d,xmax(2)+4*d,N);
    [U,V]=meshgrid(u,v);
    z=mvnpdf([U(:) V(:)],m1',S1);
    z1=reshape(z,length(v),length(u));
    z=mvnpdf([U(:) V(:)],m2',S2);
    z2=reshape(z,length(v),length(u));
    z=mvnpdf([U(:) V(:)],m3',S3);
    z3=reshape(z,length(v),length(u));
    z=mvnpdf([U(:) V(:)],m4',S4);
    z4=reshape(z,length(v),length(u));

    w=[0.01 0.025 0.05 0.1 0.15 0.2 0.25];
    [c,h1]=contour(u,v,z1,w,'red'); hold on
    [c,h2]=contour(u,v,z2,w,'red'); hold on    
    [c,h3]=contour(u,v,z3,w,'blue'); hold on
    [c,h4]=contour(u,v,z4,w,'blue'); hold on    
    h5=scatter(X0(1,:),X0(2,:),'r'); hold on
    h6=scatter(X1(1,:),X1(2,:),'b'); hold on
    axis([xmin(1) xmax(1) xmin(2) xmax(2)])
    hold off 
end


function [X,y]=IrisData
    data_path = 'iris.txt';
    data = load(data_path);
    X = data(:,1:4).';
    y=[repelem(1,1,50) repelem(2,1,50) repelem(3,1,50)];
end
  

function Y=Yencoder(y,onehot)   % onehot or binary encode labels y=1,2,...,K
    N=length(y);                % number of samples
    yunique=unique(y);          % unique labels
    K=length(yunique);          % number of classes                     
	if onehot
        Y=-ones(K,N);
        for n=1:N
            Y(y(n),n)=1;
        end
    else  
        b=ceil(log2(K));      	% number of bits
        Y=[];
        for n=1:N
            z=2*de2bi(y(n)-1,b)-1;
            Y=[Y,z'];
        end
    end
end