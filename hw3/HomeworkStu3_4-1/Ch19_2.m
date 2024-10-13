% Chapter 19 SOM
% Implement SOM to the dataset 'LorenzData'
clear, close all
rng(5);

X=LorenzData;
[N,K]=size(X);
xmin=min(X(1,:)); xmax=max(X(1,:)); dx=xmax-xmin;
ymin=min(X(2,:)); ymax=max(X(2,:)); dy=ymax-ymin;
zmin=min(X(3,:)); zmax=max(X(3,:)); dz=zmax-zmin;
fprintf('[%.0f %.0f], [%.0f %.0f], [%.0f %.0f]\n',xmin,xmax,ymin,ymax,zmin,zmax);
M=24;
W=rand(N,M,M);
for i=1:M
    for j=1:M
        W(1,i,j)=xmin+dx*rand;
        W(2,i,j)=ymin+dy*rand;
        W(3,i,j)=zmin+dz*rand;
    end
end

figure(1)
subaxis(1,2,1, 'Spacing', 0.02, 'Padding', 0.02, 'Margin', 0.02);
DisplayWGrid(W,X)
title('Before competitive learning')

figure(2)
subaxis(1,2,1, 'Spacing', 0.02, 'Padding', 0.02, 'Margin', 0.02);
DisplayW(W,X)
title('Before competitive learning')

% Your code here to implement SOM.
% Note that here you should use minimum Euclidean distance instead of inner product
% to find the winner so that the weights W do not need to be normlized here.
% For more information, refer to Example 19.4 in the textbook.
eta=0.8;                          % initial learning rate
sgm=M;                            % width of Gaussian
decay=0.999;                      % decay rate
l = 0;
% find appropriate training iterations
% you can try nt = 1e4;
nt=1e4;                              
for it=1:nt                      % training iterations
    % Find the winner neuron for each input vector
    k=randi(K);
    x = X(:,k);
    dmin = inf;
    for i = 1:M
        for j = 1:M
            w = reshape(W(:,i,j), [3 1]);
            d = norm(x - w);
            if d < dmin
                dmin = d;
                wi = i;
                wj = j;
            end
        end
    end   
    % Update the weights of the neighborhood
    for i = 1:M
        for j = 1:M
            % Compute the distance from the current neuron to the winner neuron
            d = norm([i-wi, j-wj]);
            % Compute the neighborhood function
            h = exp(-(d^2)/sgm);
            % Update the weights of the current neuron
            W(:,i,j) = W(:,i,j)+eta*h*(x-W(:,i,j));
        end
    end
    % Update the learning rate and neighborhood width
    eta = eta * decay;
    sgm = sgm * decay;
    
    if it<9
        figure(3)
        subaxis(2,4,mod(it-1,8)+1, 'Spacing', 0.01, 'Padding', 0.01, 'Margin', 0.01);
        DisplayWGrid(W,X); 
    end        
    if ~mod(it,100) && l<8
        figure(4)
        l=l+1;
        subaxis(2,4,mod(l-1,8)+1, 'Spacing', 0.01, 'Padding', 0.01, 'Margin', 0.01);
        % fprintf('%d, k=%d, winner=(%d %d), eta=%.3f, sgm=%.3f\n',it,k,wi,wj,eta,sgm); 
        DisplayWGrid(W,X);
        % pause
    end
end

figure(1)
subaxis(1,2,2, 'Spacing', 0.02, 'Padding', 0.02, 'Margin', 0.02);
DisplayWGrid(W,X);
title('After competitive learning')

figure(2)
subaxis(1,2,2, 'Spacing', 0.02, 'Padding', 0.02, 'Margin', 0.02);
DisplayW(W,X)
title('Before competitive learning')


function DisplayWGrid(W,X)
[n,M,N]=size(W);
[D,K] = size(X);
x = X(:,1);
dmin=inf;
iter = 0;
for i=1:M
    for j=1:M
        w=reshape(W(:,i,j),[3 1]);
        d=norm(x-w);
        if d<dmin
            dmin=d;  wi=i;  wj=j;
        end
    end
end
w_old = [wi wj];
for k = 1:K
    x = X(:,k);
    dmin=inf;
    for i=1:M
        for j=1:M
            w=reshape(W(:,i,j),[3 1]);
            d=norm(x-w);
            if d<dmin
                dmin=d;  wi=i;  wj=j;
            end
        end
    end
    w = [wi wj];
    if w_old ~= w
        plot([w_old(1) w(1)],[w_old(2) w(2)]); hold on
        w_old = w;
        iter = iter + 1;
    end
end
for i=1:M
    for j=1:M
        w=reshape(W(:,i,j),[3 1]);
    end
end
hold off
end

function DisplayW(W,X)
[n,M,N]=size(W);
[D,K] = size(X);
x = X(:,1);
dmin=inf;
iter = 0;
for i=1:M
    for j=1:M
        w=reshape(W(:,i,j),[3 1]);
        d=norm(x-w);
        if d<dmin
            dmin=d;  wi=i;  wj=j;
        end
    end
end
w_old=reshape(W(:,wi,wj),[3 1]);
for k = 1:K
    x = X(:,k);
    dmin=inf;
    for i=1:M
        for j=1:M
            w=reshape(W(:,i,j),[3 1]);
            d=norm(x-w);
            if d<dmin
                dmin=d;  wi=i;  wj=j;
            end
        end
    end
    w=reshape(W(:,wi,wj),[3 1]);
    if w_old ~= w
        plot3([w_old(1) w(1)],[w_old(2) w(2)],[w_old(3) w(3)]); hold on
        w_old = w;
        iter = iter + 1;
    end
end
for i=1:M
    for j=1:M
        w=reshape(W(:,i,j),[3 1]);
        plot3(w(1),w(2),w(3),'*'), hold on
    end
end
hold off
end

function X=LorenzData
data_path = 'Lorenz.txt';
X = load(data_path);
X = X.';
X = X(:,2:10:end);
end