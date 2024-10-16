% Chapter 9 Feacture selection 
% iris flower dataset 
% You will solve the following problems.
% 1) Visualize iris dataset using the first two features versus the classes
% 2) Calculate distances of class centers using 1,2 and infinite norm
% 3) Calculate Bhattacharyya distances of class centers
% 4) Calculate between-class, within-class and total scatter matrices
% You may refer to pages 224-226 of our textbook.
clear, close all
X = iris_dataset;      % read in iris flower dataset
[d, N] = size(X);      % dimensionality and number of samples
y = [ones(1,50) 2*ones(1,50) 3*ones(1,50)]; % data labelings
K = 3;                 % number of classes
Nk = [50 50 50];       % number of samples in each class
    
%% 1) Visualize the first two dimensions of the dataset
figure(1)
gscatter(X(1,:),X(2,:),y)
legend('class 1','class 2','class 3')

%% 2) find all K(K-1)/2 pair-wise distances {dp(mi, mj)} between the mean
% vectors of any two of different classes for each value of p = 1, 2, ∞. 
% Display your results as the lower triangular matrix 
% (all components below the main diagonal of a K × K matrix)
mu = zeros(d,K);       % mean vectors for all classes
Cm = zeros(d,d,K);     % covariance matrices for all classes
% specify mean vector for each class
for k = 1:K
    idx = (y==k);       % indices of samples in class k
    Xk = X(:,idx);      % collect all samples in class k
    mu(:,k) = mean(Xk,2); % mean vector of class k
    Cm(:,:,k) = cov(Xk.'); % covariance matrix of class k
end
% find all K(K-1)/2 pair-wise distances between mean vectors stored in 
% the form of lower triangular matrix for each value of p = 1, 2, ∞.
dp1 = zeros(K,K);     % p = 1
dp2 = zeros(K,K);     % p = 2
dpinf = zeros(K,K);   % p = ∞

for i = 1:K
    for j = 1:i-1
        % compute dp1(i,j), dp2(i,j) and dpinf(i,j)
        % your code here to compute dp1, dp2 and dpinf
        dp1(i,j) = norm(mu(:,i)-mu(:,j),1);
        dp2(i,j) = norm(mu(:,i)-mu(:,j),2);
        dpinf(i,j) = norm(mu(:,i)-mu(:,j),inf);
    end
end
fprintf('d1 = \n')
disp(dp1)
fprintf('d2 = \n')
disp(dp2)
fprintf('dinf = \n')
disp(dpinf)
    
%% 3) Find all pair-wise Bhattacharyya distances between any two classes, 
% and show them as the lower triangular matrix, 
% and compare them with those obtained in the previous problem.
dB = zeros(K,K);      % Bhattacharyya distances

for i = 1:K
    for j = 1:i-1
        % compute dB(i,j); refer to equation 9.16 in the textbook
        % your code here to compute Bhattacharyya distances
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

%% 4) Find the between-class, within-class and total scatter matrices 
% and verify they do satisfy SB + SW = ST.
SB = zeros(d,d);
SW = zeros(d,d);
ST = zeros(d,d);
mm = mean(mu,2);
for k = 1:K
    % your code here to compute SB
    SB = SB + (1/N)*(Nk(k)*(mu(:,k) -mm)*(mu(:,k)-mm)');
    idx = (y==k);       % indices of samples in class k
    Xk = X(:,idx);      % collect all samples in class k
    % your code here to compute SW and ST
    SW = SW +(1/N)*((Xk - mu(:,k))*(Xk - mu(:,k))');
    ST = ST + (1/N)*((Xk - mm)*(Xk - mm)');
end
% Verify SB + SW = ST
res = sum(sum(SB + SW - ST));
tol = 1e-6;
if res < tol
    fprintf('SB + SW = ST verified.\n')
end
    
function X = iris_dataset
    % Read in iris flower dataset
    data_path = 'iris.txt';
    data = load(data_path);
    X = data(:,1:4).';
end