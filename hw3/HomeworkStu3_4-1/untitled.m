%% Ch18 Back propagation network for classification
% You may find reference in the textbook we used till 2023/07/20 from 447 
% to 457 or new text book available on https://pages.hmc.edu/ruye/book2/Book.pdf
% from 449 to 459.
% The code for backPropagate function is fulfilled in the textbook, but you
% need to complete the missing code to realize this function and try
% different learning rate to see how learning rate affects the training
% process including training time, convergence and error rate

rng(2023)
onehot=1;


[X y]=Data;             % linearly saperable data
[d N]=size(X);
size(y);

K=length(unique(y));
fprintf('d=%d\tN=%d\tK=%d\n',d,N,K);

Y=Yencoder(y,onehot);                       % class encoding
Yunique=unique(Y','rows','stable')';        % unique class labels

itrain=sort(randperm(N,N/2));               % random indices for half of the data
itest=setdiff(1:N,itrain);
Xtrain=X(:,itrain);    Xtest=X(:,itest);
Ytrain=Y(:,itrain);    Ytest=Y(:,itest);
ytrain=y(itrain);      ytest=y(itest);

fprintf('\nTraining:\n\n')
% Specify eta from 0.001, 0.01, 0.1 and 1
eta = 1; % learning rate or step size (0,1)
tic;
[W,g]=backPropagate(Xtrain,Ytrain,ytrain, eta);
toc;

fprintf('\nTesting:\n\n')
Xtest=[ones(1,length(Xtest)); Xtest];
yhat=test(Xtest,W,Yunique,g);
[Cm er]=ConfusionMatrix(yhat,ytest);
fprintf('Cm = \n')
disp(Cm)
fprintf('Testing: er=%.4f\n', er);


function  [W,g]= backPropagate(X, Y, y, eta)
    % Input:
    % X: d x N matrix of training data vectors
    % Y: M x N matrix of desired outputs (encoded)
    % y: class labeling (from 1 to K)
    % Output:
    % W: weights at all learning layers
    % g: sigmoid function
	            
    [d N]=size(X);
    X=[ones(1,N); X];               % data augmentation    
    Yunique=unique(Y','rows','stable')';        % unique class labels
    syms x 
    g=1/(1+exp(-x));                % Sigmoid activation function
    dg=diff(g);    
    g=matlabFunction(g);
    dg=matlabFunction(dg);              
    
    H=3;                            % number of layers (excluding input)
    L=30*ones(1,H);                 % number of nodes per hidden layer
    L(H)=size(Y,1);                 % number of output nodes     
    W={1-2*rand(L(1),d+1)};         % initial weights for first layer
    for h=2:H
        W{h}=1-2*rand(L(h),L(h-1)+1);  % initializing weights for all layers
    end

    er=1;                            % Initialize er as 1 to start loop
    convergence = "true";
    it=0;
    d={};
    while er > 0.001    
        it=it+1;
        er=0;        
        I=randperm(N);                      % random order of training samples
        for n=1:N
            % Your code here to complete following six missing lines
            z={;X(:,I(n))};                  % pick a random training sample
            a={W{1}*z{1}};                  % activation of first layer
            for h=2:H                       % forward pass
                z{h}=[1;g(a{h-1})];         % input to layer h
                a{h}=W{h}*z{h};             % activation of layer h
            end
            yhat=g(a{H});                   % output from last layer
            delta=Y(:,I(n))-yhat;
            er=er+norm(delta)/N;
            % Your code here to complete following four missing lines
            d{H}=delta.*dg(a{H});
            W{H}=W{H}+eta*d{H}*z{H}';
            for h=H-1:-1:1                  % backward pass
                d{h}=(W{h+1}(:,2:end)'*d{h+1}).*dg(a{h});
                W{h}=W{h}+eta*d{h}*z{h}';
            end  
        end
        if ~mod(it,100)
            fprintf('epoch %d: er=%.4f\n', it,er);  
        end
        if it > 100000
            convergence = "false";
            break
        end
    end
    fprintf("Converged in 100000 iterations: %s\n", convergence)
	yhat=test(X,W,Yunique,g);
	[Cm er]=ConfusionMatrix(yhat,y);
end


function ytest=test(Xtest,W,Yunique,g)      % given Wh, Wo and g, classify Xtest
    H=size(W,2);
    K=length(Yunique);                      % number of classes
    N=length(Xtest);                        % number of test samples
    Z=g(W{1}*Xtest); 
    for h=2:H                               % forward pass
        Z=g(W{h}*[ones(1,N);Z]);            % output from layer h
    end
	Yhat=g(Z);
    for n=1:N
        yhat=Yhat(:,n);
        [d,k]=min(sum(abs(Yunique-yhat)));      % find class with min distance
        ytest(n)=k;
    end    
end


function [Cm er]=ConfusionMatrix(yhat,ytest)
    K=length(unique(ytest));         % number of classes
    N=length(ytest);                 % number of test samples
    Cm=zeros(K);
    for n=1:N
        i=ytest(n);
        j=yhat(n);
        Cm(i,j)=Cm(i,j)+1;                
    end
    if sum(sum(Cm)) ~= N
        fprintf('%d ~= %d\n',sum(sum(Cm)),N)
    end
    er=1-sum(diag(Cm))/N;
    fprintf('%d/%d\n',N-sum(diag(Cm)),N)
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
end


function Y=Yencoder(y,onehot)   % onehot or binary encode labels y=1,2,...,K
    N=length(y);                % number of samples
    yunique=unique(y,'stable');          % unique labels
    K=length(yunique);          % number of classes                     
	if onehot
        Y=zeros(K,N);
        for n=1:N
            Y(y(n),n)=1;
        end
    else  
        b=ceil(log2(K));      	% number of bits
        Y=[];
        for n=1:N
            z=de2bi(y(n),b)';
            Y=[Y,z];
        end
    end
end