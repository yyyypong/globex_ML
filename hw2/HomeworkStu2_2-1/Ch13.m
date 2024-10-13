% Chapter 13 Statistic Classification
% Naive Bayes Classification
% You will solve the following problems.
% 1) Carry out classification using both the iris and the handwritten digit datasets by Naive Bayes method.
% 2) In each case, cross validate your algorithm by using 50% randomly chosen samples 
% in the dataset for training and the other 50% for testing. 
% 3) Show your classification results in the confusion matrix together with the error rate, 
% the sum of all off-diagonal components of the confusion matrix divided by the total number of samples.
clear, close all
rng('default')

flg = 1;
if flg==1
    [X,y]=iris_dataset; % Read in iris flower dataset
elseif flg==2
    [X,y]=data0to9;     % Read in handwritten digit dataset
end
K=length(unique(y));
fprintf("K = %d\n\n",K)

N=size(X,2);
itrain=sort(randperm(N,N/2));             % random indices for half of the data
itest=setdiff(1:N,itrain);
Xtrain=X(:,itrain);    Xtest=X(:,itest);
ytrain=y(itrain);      ytest=y(itest);

[M, S, P]=NBtraining(Xtrain,ytrain);      % training ML classifier to get means and covariances

ys=NBtesting(Xtrain,M,S,P);
Cm=ConfusionMatrix(ys,ytrain);
fprintf("Cm = \n")
disp(Cm)

ys=NBtesting(Xtest,M,S,P);       
Cm=ConfusionMatrix(ys,ytest);
fprintf("Cm = \n")
disp(Cm)


function [M, S, P]=NBtraining(X,y) % naive Bayes training
[d, N]=size(X);                    % dimensions of dataset
K=length(unique(y));               % number of classes
M=zeros(d,K);                      % mean vectors
S=zeros(d,d,K);                    % covariance matrices
P=zeros(1,K);                      % prior probabilities
% your code here to compute M, S, P
for k=1:K                          % for each of K classes
    
end
end

function yhat=NBtesting(X,M,S,P)   % naive Bayes testing
[d, N]=size(X);                    % dimensions of dataset
K=length(P);                       % number of classes
InvS = zeros(d,d,K);
Det = zeros(1,K);
yhat = zeros(1,N);
for k=1:K
    InvS(:,:,k)=inv(S(:,:,k));     % inverse of covariance
    Det(k)=det(S(:,:,k));          % determinant of covariance
end
for n=1:N                          % for each of N samples
    x=X(:,n);
    dmax=-inf;
    for k=1:K
        % your code here to compute the discriminant function 
        % defined by equation 13.16 in textbook
        d = ;
        if d>dmax
            yhat(n)=k;             % assign nth sample to kth class
            dmax=d;
        end
    end
end
end

function [Cm, er]=ConfusionMatrix(yhat,ytrain)
N=length(ytrain);                  % number of test samples
K=length(unique(ytrain));          % number of classes
Cm=zeros(K);                       % the confusion matrix
for n=1:N
    i=ytrain(n);
    j=yhat(n);
    Cm(i,j)=Cm(i,j)+1;
end
% your code here to compute the number of samples that are misclassified
r = ;
% your code here to compute error percentage
er = ;
fprintf('error rate: %d/%d=%.4f\n',r,N,er);
end

function [X, y] = iris_dataset
% Read in iris flower dataset
data_path = 'iris.txt';
data = load(data_path);
X = data(:,1:4).';
y = data(:,5).';
end

function [X, y] = data0to9
% Read in handwritten digit dataset
data=load('data0to9.txt','data'); % read ASCII file
d=size(data,1)-1;    % dimensionality of data
X=data(1:d,:);       % all N d-dimensional sample vectors
y=data(d+1,:) + 1;   % and their labelings
% carry out a KLT to reduce the dimensionality
[V, D]=eig(cov(X'));
[d, idx]=sort(diag(D),'descend');
V=V(:,idx);
V=V(:,1:10);
X=V'*X;
end