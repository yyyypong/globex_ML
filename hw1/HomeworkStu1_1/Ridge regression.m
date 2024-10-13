% Problem 4 in Chapter 5

clear, close all
% Set random seed
rng(2023);

% Generate data
x1 = [2.4, 3.1, 3.8, 2.3, 2.0, 3.7, 3.2, 3.0, 2.8]';
y = [4.6, 6.1, 7.7, 4.9, 4.1, 7.4, 6.3, 5.8, 5.5]';
y_mean = mean(y);

N = size(x1,1);
x1_mean = mean(x1);

% Specify a like 1, 10, 100, 1000
a = 1;
x2 = x1 + a*rand(N,1);
x2_mean = mean(x2);

% Calculate coefficient of x1 and x2
rho=corr2(x1,x2);
fprintf('Coefficient: %.4f\n', rho);

% Calculate eigenvalues of XX^T to see how ill-conditioned the problem is
X = [x1,x2];
e = eig(X'*X);
fprintf('Eigenvalues are %.16f, %.16f\n', e(1), e(2));

% Perform ridge regression on X, y
new_x=(X*X'+lambda*eye(d));
% Compute w, yhat, ybar
w=pinv(new_x)*y;
yhat=X*w;
ybar=y_mean*ones(N,1);
% Compute TSS, ESS
TSS=(y-ybar)'*(y-ybar);
ESS=(ybar-yhat)'*(ybar-yhat);
% Report R-square
R2=ESS/TSS;

% Specify lambda
lambda = 0.1;
old_x = X;
X=[ones(N,1) X];        % augmented data array
d = size(X,2);



fprintf('R-square is %.4f\n', R2);