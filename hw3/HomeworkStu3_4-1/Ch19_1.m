% Chapter 19 SOM
% Implement SOM to the dataset 'ColorData'
clear, close all

X=ColorData;
[N,K]=size(X);
for i=1:K
    X(:,i)=X(:,i)/norm(X(:,i));
end
M=200;
W=rand(M,M,N);
for i=1:M
    for j=1:M
        w=reshape(W(i,j,:),[N 1]);
        W(i,j,:)=W(i,j,:)/norm(w);
    end
end

subaxis(1,3,1, 'Spacing', 0.01, 'Padding', 0.01, 'Margin', 0.01);
size(X)
title('Before competitive learning')
imshow(W);

% your code here to implement SOM
eta=0.9;                              % initial learning rate
sgm=2*M;                        % width of Gaussian
decay=0.999;                      % decay rate
% find appropriate training iterations
% you can try nt = 1000
nt=1000;
for it=1:nt                      % training iterations
    n = randi([1 K]);
    x = X(:,n);
    dmax = -inf;
    for i = 1:M
        for j = 1:M
            w = reshape(W(i,j,:), [N 1]);
            d = w' * x; % compute the inner product
            if d > dmax
                dmax = d;
                wi = i;
                wj = j;
            end
        end
    end
    % Update the weights of the neighborhood
    for i = 1:M
        for j = 1:M
            % Compute the distance from the current neuron to the winner neuron
            dist = (i-wi)^2+(j-wj)^2;
            % Compute the neighborhood function
            c = exp(-dist/sgm);
            % Update the weights of the current neuron
            w = reshape(W(i,j,:), [N 1]);
            w = w+eta*c*(x-w);
            w = w/norm(w);
            W(i,j,:) = w;
        end
    end
    % Update the learning rate and neighborhood width
    eta = eta * decay;
    sgm = sgm * decay;

    fprintf('%d, winner=(%d %d), eta=%.3f, sgm=%.3f\n',it,wi,wj,eta,sgm);
end

subaxis(1,3,2, 'Spacing', 0.01, 'Padding', 0.01, 'Margin', 0.01);
title('After competitive learning')
imshow(W);

V=zeros(M,M,N);
for k=1:K
    k
    x=X(:,k);
    dmax=-9e9;
    for i=1:M                        % find winner in all M output nodes
        for j=1:M
            w=reshape(W(i,j,:),[N 1]);
            d=x'*w;
            if d>dmax
                dmax=d;  wi=i;  wj=j;   % get winner
            end
        end
    end
    V(wi,wj,:)=x;
end
subaxis(1,3,3, 'Spacing', 0.01, 'Padding', 0.01, 'Margin', 0.01);
imshow(V);

function X=ColorData
L=20;
l=0;
v=255/(L-1);
for i=1:L
    for j=1:L
        for k=1:L
            l=l+1;
            X(1,l)=(i-1)*v;
            X(2,l)=(j-1)*v;
            X(3,l)=(k-1)*v;
        end
    end
end
end