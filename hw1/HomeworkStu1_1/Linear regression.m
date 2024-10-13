% Problem 3 in Chapter 5

clear, close all
% Load data
data_path = "dataLS1.txt";
data = load(data_path);

% Specify X, y for dataLS1 - this data has two columns, first column is X,
% the second column is target. So it's a simple regression.
X = data(:,1);
y = data(:,2);
[N,d] = size(X);

% 3.a 1) Report coefficient rho
% Please calculate the mean of data X and target y as in equation 5.28
xmean=mean(X);
ymean=mean(y);
% Please calculate the variation of data X and target y as in equation 5.27
xvar=1/(N-1)*(X-xmean)'*(X-xmean);
yvar=1/(N-1)*(y-xmean)'*(y-xmean);
% Please calculate the coefficient as in equation 5.26
xyvar=1/(N-1)*(X-xmean)'*(y-ymean);     % calculate the covariance
rho=xyvar/sqrt(xvar*yvar);

fprintf('rho=%.4f\n',rho);

% Perform linear regression on X, y
x = X;
X=[ones(N,1) X];         % augmented data array
% Compute w, yhat, ybar
w=pinv(X)*y;
yhat=X*w;
ybar=ymean*ones(N,1);
% Report TSS, ESS, RSS, R-square, and rho
TSS=(y-ybar)'*(y-ybar);
ESS=(ybar-yhat)'*(ybar-yhat);
RSS=(y-yhat)'*(y-yhat);
R2=ESS/TSS;

fprintf('ESS=%.4f\tRSS=%.4f\tTSS=%.4f\tR2=%.4f\n',ESS,RSS,TSS,R2);

% Plot
plot(x,y,'o',x,yhat,'-',x,ybar,'-');
box on
xlim([min(x)-0.2 max(x)+0.2]);
legend({'{\bf y}','$\hat{\bf y}$','$\bar{\bf y}$'},'Interpreter','latex','Location','northwest');
