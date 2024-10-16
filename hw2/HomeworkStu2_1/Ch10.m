 % Chapter 10 PCA for Feature Extraction 
% handwritten digit dataset
% You will solve the following problems.
% 1) Plot dynamic energy or information contained in each data component
% and after KLT
% 2) Visualize the handwritten digit dataset in 2-D space
% 3) Calculate Bhattacharyya distance of class centers
% 4) Use PCA on handwritten digits
% You may refer to pages 252-254 of our textbook.
clear, close all
[X, y] = data0to9;
[d, N] = size(X);      % dimensionality and number of samples
K=length(unique(y));   % number of classes
Nk = zeros(1, K);      % number of samples in each class

%% 1) Carry out KLT for the handwritten digit dataset.
% Plot their diagonal components (representing the dynamic energy or 
% information contained in each data component), same as in Fig. 10.16.
mu = zeros(d,K);       % mean vectors for all classes
Cm = zeros(d,d,K);     % covariance matrices for all classes
% specify mean vector for each class
for k = 1:K
    idx = (y==k-1);     % indices of samples in class k
    Nk(k) = sum(idx);   % number of samples in class k
    Xk = X(:,idx);      % collect all samples in class k
    mu(:,k) = mean(Xk,2); % mean vector of class k
    Cm(:,:,k) = cov(Xk.'); % covariance matrix of class k
end
% Find the total scatter matrices ST
ST = zeros(d,d);
mm = mean(mu,2);
for k = 1:K
    % your code here to compute ST
    idx = (y==k);       
    Xk = X(:,idx);
    ST = ST + (1/N)*(Xk-mm)*(Xk-mm).';
end

v = zeros(d, 1);
for i = 1:d
    v(i) = ST(i,i);
end
% Plot the original energy distributuion
figure(1)
subaxis(3,1,1, 'Spacing', 0.04, 'Padding', 0.04, 'Margin', 0.04);
plot(1:d,v);    
xlim([1 d])
title('Energy distribution')

[V, D] = KLT(X, ST);
for i = 1:d
    v(i) = D(i,i);
end
% Plot the energy distributuion after KLT based on ST
figure(1)
subaxis(3,1,2, 'Spacing', 0.04, 'Padding', 0.04, 'Margin', 0.04);
plot(1:d,v);    
xlim([1 d])
title('Energy distribution after KLT based on ST')

SB = zeros(d,d);
for k = 1:K
    % your code here to compute SB
    SB = SB + (1/N)*(Nk(k)*((mu(:,k)-mm)*(mu(:,k)-mm).'));
end
[V, D] = KLT(X, SB);
for i = 1:d
    v(i) = D(i,i);
end
% Plot the energy distributuion after KLT based on SB
figure(1)
subaxis(3,1,3, 'Spacing', 0.04, 'Padding', 0.04, 'Margin', 0.04);
plot(1:d,v);    
xlim([1 d])
title('Energy distribution after KLT based on SB')


%% 2) Visualize the handwritten digit dataset in 2-D space spanned by the first two principal components. 
% Color code all data points according to their class identities
N0 = 2;
[V, D] = KLT(X, ST);    % KLT based on ST
for i = 1:d
    v(i) = D(i,i);
end
Y = V(:,1:N0)'*X;       % KLT transform
figure(2)
gscatter(Y(1,:),Y(2,:),y)
title('2-D space spanned')

N0 = 3;
[V, D] = KLT(X, ST);    % KLT based on ST
for i = 1:d
    v(i) = D(i,i);
end
Y = V(:,1:N0)'*X;       % KLT transform
figure(3), hold on
for k = 1:K
    idx = (y==k-1);     % indices of samples in class k
    scatter3(Y(1,idx),Y(2,idx),Y(3,idx),'.')
end
view(3), grid on
legend('0','1','2','3','4','5','6','7','8','9')
title('3-D space spanned')


%% 3) Find all pair-wise Bhattacharyya distances between any two classes,
% and display them in a lower triangular distance matrix,
% and you will see that all the covariance matrices are non-invertable.
dB = zeros(K,K);      % Bhattacharyya distances
for i = 1:K
    for j = 1:i-1
        % your code here to compute Bhattacharyya distances.
        % Note that all the covariance matrices are non-invertable.
        C1 = Cm(:,:,i);
        C2 = Cm(:,:,j);
        mu1 = mu(:,i);
        mu2 = mu(:,j);
        C_avg = (C1 + C2)/2;
        term1 = (1/4)*(mu1-mu2).'/(C_avg)*(mu1 - mu2);
        term2 = log(det(C_avg) / sqrt(det(C1)*det(C2)));
        dB(i,j) = term1 + term2;
    end
end
fprintf('dB = \n')
disp(dB)

%% 4) Carry out PCA to the handwritten digit dataset.
% Observe the resulting eigen-digits to see how many eigen values we need
% to keep the original digit distinguishable
Y = X(:,1);
figure(4)
for i = 1:16
    subplot(4,4,i)
    digit = reshape(V(:,1:i*16)*V(:,1:i*16).'*Y,[16 16]).';
    imshow(digit)
    if i<16
        title("First "+16*i)
    else
        title("Original")
    end
end

function [V, D] = KLT(X, Cm)
    % Compute KLT transform matrix
    [V, D] = eig(Cm);
end

function [X, y] = data0to9
    % Read in handwritten digit dataset
    data=load('data0to9.txt','data'); % read ASCII file
    d=size(data,1)-1;    % dimensionality of data
    X=data(1:d,:);       % all N d-dimensional sample vectors
    y=data(d+1,:);       % and their labelings
end