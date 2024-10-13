% Section 8.1 Gaussian Process Regression
% Complete the missing code, and add appropriate comments 
% to key steps and formulas of the algorithm.
rng(9);
ns=12;                              % number of samples drawn from posterior
xmin=0;
xmax=9;
xstar=xmin:0.1:xmax;                % locations for all samples
n=length(xstar);                    % number of test samples

sigma_n=0.2;                        % noise
noise=randn(1,n)*sigma_n^2;

f1 = 0.2;                           % frequency
fx = cos(2*pi*f1*xstar);
ystar=(fx+noise)';                  % training samples of y(x)

% You should try different values for the hyperparameter
a = 1.5;                              % hyperparameter
Kss=Kernel(xstar,xstar,a);
fmean = zeros(n,1);

figure(1)
subaxis(4,4,1, 'Spacing', 0.01, 'Padding', 0.01, 'Margin', 0.03);
plot(xstar,fmean,'r',xstar,fmean+diag(Kss),'--b',xstar,fmean-diag(Kss),'--b');
xlim([xmin xmax])
ylim([-1.5 1.5])
subaxis(4,4,9, 'Spacing', 0.01, 'Padding', 0.01, 'Margin', 0.03);
L=(SVDsqrt(Kss))';
for i=1:ns
    % your code here to draw samples from prior
    % Note: you CAN NOT copy it from the textbook because it is WRONG!
    plot(xstar, L'*(randn(n,1)))
    hold on
end
xlim([xmin xmax])
ylim([-2.5 2.5])
hold off

id = [randi(n,1,1)];    % pick the first random training data point
for i=2:8               % regression with increasing number of training samples
    x=xstar(id);        % input
    f=ystar(id);
    Kxx=Kernel(x,x,a)+sigma_n^2*eye(length(x));
    Kxs=Kernel(x,xstar,a);
    Ksx=Kxs';
    fmean=Ksx*(Kxx\f);
    fcov=Kss-Ksx*(Kxx\Kxs);

    subaxis(4,4,i, 'Spacing', 0.01, 'Padding', 0.01, 'Margin', 0.03);
    stem(x,f,'filled','k')
    hold on
    plot(xstar,fmean,'r',xstar,fmean+diag(fcov),'--b',xstar,fmean-diag(fcov),'--b');
    xlim([xmin xmax])
    ylim([-1.5 1.5])
    hold off

    id=sort([id randi(n,1,1)]);

    subaxis(4,4,8+i, 'Spacing', 0.01, 'Padding', 0.01, 'Margin', 0.03);
    stem(x,f,'filled','k')
    hold on
    L=(SVDsqrt(fcov))';
    for j=1:ns
        % your code here to draw samples from posterior
        % Note: you CAN NOT copy it from the textbook because it is WRONG!
        plot(xstar, L'*(randn(n,1))+fmean)
        hold on
    end
    xlim([xmin xmax])
    ylim([-2.5 2.5])
    hold off
end
figure(2)
stem(x,f,'filled','k')
hold on
plot(xstar,fx,xstar,ystar)
xlim([xmin xmax])
ylim([-1.5 1.5])
hold off

function K=Kernel(x,y,a)
% your code here to complete the Kernel function
n=length(x);
m=length(y);
K=zeros(n,m);
for i=1:n
    for j=1:m
        r=norm(x(:,i)-y(:,j));
        K(i,j)=exp(-(r/a)^2);
    end
end
end

function Asqrt=SVDsqrt(A)
% A = Asqrt * Asqrt'
if det(A-A') > 10^(-9)
    disp('matrix is not symmetric')
    return
else
    [u, s, v]=svd(A);
    Asqrt=u*sqrt(s);
end
end



