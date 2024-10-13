function main
% Logistic Regression algorithm for binary classification
    clear, close all
    % Load data
    [X,y]=myData;
    
    tic
    [iter1, acu1]=LogisticGD(X,y);     % Logistic Regression by gradient descent
    fprintf('Number of iterations LogisticGD: %d\n',iter1);
    toc

    tic
    [iter2, acu2]=LogisticNR(X,y);     % Logistic Regression by Newton-Raphson
    fprintf('Number of iterations LogisticNR: %d\n',iter2);
    toc

    fprintf('accuracy: %.2f%%, %.2f%%.\n',100*acu1,100*acu2);
end

function [iter, acu, ys]=LogisticGD(X,y)     % Logistic Regression by gradient descent
    [m, n]=size(X);
    y(y<0)=0;             % set all y(i)=-1 to y(i)=0
    x = X;
    X=[ones(1,n); X];     % augmented X with x_0=1 included
    w=ones(m+1,1);        % initial parameter (weight) vector
    tol=10^(-6);          % tolerance
    delta=0.1;            % step size
    %% gradient of log posterior
    g=X*(sgm(w,X)-y)';

    iter=0;
    while norm(g)>tol     % terminate when g is small enough
        iter=iter+1;
        %% update weight by gradient descent
        w=w-delta*g;
        %% update gradient
        g=X*(sgm(w,X)-y)';
    end
    ys=-ones(1,n);        % classification result
    ys(sgm(w,X)>0.5)=1;
    % confusion matrix
    [cm, acu]=ConfusionMatrix(y,ys);
    fprintf("Confusion matrix LogisticGD:\n")
    disp(cm)
    % plot
    myplot(x, y, w)
    title('LogisticGD')
end

function [iter, acu, ys]=LogisticNR(X,y)     % Logistic Regression by Newton-Raphson
    [m, n]=size(X);
    y(y<0)=0;             % set all y(i)=-1 to y(i)=0
    x = X;
    X=[ones(1,n); X];     % augmented X with x_0=1 included
    w=ones(m+1,1);        % initial parameter (weight) vector
    tol=10^(-6);          % tolerence
    delta=0.1;            % step size
    % Note that Newton-Raphson method is different from Newton's method,
    %% so you CAN NOT just copy the code from the textbook without any modification.
    %% gradient of log posterior
    g=X*(sgm(w,X)-y)';
    %% Hesian of log posterior
    H=X*diag(sgm(w,X).*(1-sgm(w,X)))*X';
    iter=0;
    while norm(g)>tol     % terminate when g is small enough
        iter=iter+1;
        %% update weight by Newton-Raphson
        w=w-delta*inv(H)*g;
        %% update gradient
        g=X*(sgm(w,X)-y)';
        %% update Hessian
        H=X*diag(sgm(w,X).*(1-sgm(w,X)))*X';
    end
    ys=-ones(1,n);      % classification result
    ys(sgm(w,X)>0.5)=1;
    % confusion matrix
    [cm, acu]=ConfusionMatrix(y, ys);
    fprintf("Confusion matrix LogisticNR:\n")
    disp(cm)
    % plot
    myplot(x, y, w)
    title('LogisticNR')
end

function s=sgm(w, X)
    % Sigmoid function
    s=1./(1+exp(-w'*X));
end

function [Cm, acu]=ConfusionMatrix(y, ypred)
    % Input: y; ypred-y predicted
    % Output: Cm-2x2 confusion matrix; acu-accuracy
    n=length(y);
    Cm=zeros(2);
    for i=1:n
        if y(i)==1
            k=1;
        else
            k=2;
        end
        if ypred(i)==1
            l=1;
        else
            l=2;
        end
        Cm(k,l)=Cm(k,l)+1;
    end
    acu=trace(Cm)/sum(sum(Cm));
end

function [X, y] = myData
    % load Data
    data_path = '2ClassData.txt';
    data = load(data_path);
    X = data(:,1:2).';
    y = data(:,3).';
end

function myplot(x, y, w)
    % Input: x; y; w-weights
    figure, hold on
    gscatter(x(1,:),x(2,:),y,'br')
    %% plot the straight line such that sgm(w,X)=1/2.
    x1 = min(x(1,:));
    x2 = max(x(1,:));
    y1 = -(w(1)+w(2)*x1)/w(3);      % based on decision boundary
    y2 = -(w(1)+w(2)*x2)/w(3);
    plot([x1 x2],[y1 y2],'k-')
    legend('-1','1','location','northwest')
    box on
    axis equal
    hold off
end
