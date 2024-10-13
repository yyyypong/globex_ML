function main
    clear, close all
    N=200;
  
    x=sym('x',[2 1]);
    a=sym('a',[4 1]);
    f1=@(x,a)a(1)*x(1)+a(2)*x(2)-a(3);
    f2=@(x,a)exp(a(1)*x(1)+a(2)*x(2))-a(3)*x(1).*x(2)+a(4);
    f3=@(x,a)a(1)*x(1).^2-a(2)*x(1).*x(2)+a(3)*x(2).^2+a(4);
    
%     generateData(f1,f2,f3,N);        % generate datasets
      
	figure(1)
    
    data=load('data3.txt');
    X1=data(:,1:2);
    y1=data(:,3);    
	subaxis(3,3,1, 'Spacing', 0.03, 'Padding', 0.01, 'Margin', 0.01);
    h1=scatter3(X1(:,1),X1(:,2),y1,'fill');
    hold on
    h2=subaxis(3,3,2, 'Spacing', 0.03, 'Padding', 0.01, 'Margin', 0.01);
    copyobj(h1,h2);
    hold on
    h3=subaxis(3,3,3, 'Spacing', 0.03, 'Padding', 0.01, 'Margin', 0.01);
    copyobj(h1,h3);
    hold on
    
    data=load('data4.txt');
    X2=data(:,1:2);
    y2=data(:,3);    
    subaxis(3,3,4, 'Spacing', 0.03, 'Padding', 0.01, 'Margin', 0.01);
    h4=scatter3(X2(:,1),X2(:,2),y2,'fill');
    hold on
    h5=subaxis(3,3,5, 'Spacing', 0.03, 'Padding', 0.01, 'Margin', 0.01);
    copyobj(h4,h5);
    hold on
    h6=subaxis(3,3,6, 'Spacing', 0.03, 'Padding', 0.01, 'Margin', 0.01);
    copyobj(h4,h6);
    hold on
    
    data=load('data5.txt');
    X3=data(:,1:2);
    y3=data(:,3);    
    subaxis(3,3,7, 'Spacing', 0.03, 'Padding', 0.01, 'Margin', 0.01);
    h7=scatter3(X3(:,1),X3(:,2),y3,'fill');
    hold on
    h8=subaxis(3,3,8, 'Spacing', 0.03, 'Padding', 0.01, 'Margin', 0.01);
    copyobj(h7,h8);
    hold on
    h9=subaxis(3,3,9, 'Spacing', 0.03, 'Padding', 0.01, 'Margin', 0.01);
    copyobj(h7,h9);
    hold on
    
    [a1 er]=Model1(X1,y1,f1)
    [a2 er]=Model2(X1,y1,f2)
    [a3 er]=Model3(X1,y1,f3)  
    subaxis(3,3,1, 'Spacing', 0.03, 'Padding', 0.01, 'Margin', 0.01);
	myPlot3(X1,a1,f1);
    hold off
    title('model 1 on data 1')
    subaxis(3,3,2, 'Spacing', 0.03, 'Padding', 0.01, 'Margin', 0.01);
	myPlot3(X1,a2,f2);
    hold off
    title('model 2 on data 1')
    subaxis(3,3,3, 'Spacing', 0.03, 'Padding', 0.01, 'Margin', 0.01);
	myPlot3(X1,a3,f3);
    hold off
    title('model 3 on data 1')
    %%%%%%%%
    
    [a1 er]=Model1(X2,y2,f1)
    [a2 er]=Model2(X2,y2,f2)
    [a3 er]=Model3(X2,y2,f3)
    subaxis(3,3,4, 'Spacing', 0.03, 'Padding', 0.01, 'Margin', 0.01)
	myPlot3(X2,a1,f1);
    hold off
    title('model 1 on data 2')
    subaxis(3,3,5, 'Spacing', 0.03, 'Padding', 0.01, 'Margin', 0.01)
	myPlot3(X2,a2,f2);
    hold off
    title('model 2 on data 2')
    subaxis(3,3,6, 'Spacing', 0.03, 'Padding', 0.01, 'Margin', 0.01)
	myPlot3(X2,a3,f3);
    hold off
    title('model 3 on data 2')
    
    [a1 er]=Model1(X3,y3,f1)
    [a2 er]=Model2(X3,y3,f2)
    [a3 er]=Model3(X3,y3,f3)
    subaxis(3,3,7, 'Spacing', 0.03, 'Padding', 0.01, 'Margin', 0.01)
	myPlot3(X3,a1,f1);
    hold off
    title('model 1 on data 3')
    subaxis(3,3,8, 'Spacing', 0.03, 'Padding', 0.01, 'Margin', 0.01)
	myPlot3(X3,a2,f2);
    hold off
    title('model 2 on data 3')
    subaxis(3,3,9, 'Spacing', 0.03, 'Padding', 0.01, 'Margin', 0.01)
	myPlot3(X3,a3,f3);
    hold off
    title('model 3 on data 3')
end


function myPlot3(X,a,f)
    xmin=min(X(:,1));    xmax=max(X(:,1));
    ymin=min(X(:,2));    ymax=max(X(:,2));
    [X, Y]=meshgrid(xmin:0.05:xmax, ymin:0.05:ymax);
    m=size(X,2);
    n=size(Y,1);
    Z=zeros(m,n);
    x=zeros(2,1);
    for i=1:m
        x(1)=X(1,i);
        for j=1:n
            x(2)=Y(j,1);
            Z(i,j)=f(x,a); 
        end
    end
    surf(X,Y,Z'); view(3)
end

function [a1 r2]=Model1(X,y,f)
    N=length(y);
    a=sym('a',[4 1]);
    F=[];
    for n=1:N
        F=[F; f(X(n,:),a)];
    end
    J=jacobian(F,a);
    J=matlabFunction(J);
    F=matlabFunction(F);
    a1=ones(4,1);
    tol=10^(-9);
    er=9;
    k=0;
    while er>tol
        k=k+1;
        a0=a1;        
        Jinv=pinv(J());
        r=F(a0(1),a0(2),a0(3))-y;
        r2=norm(r);
        a1=a0-Jinv*r;
        er=norm(a0-a1);        
        fprintf('%d\tr=%e\t%e\n',k,r2,er)
    end
end
    
    
function [a1 r2]=Model2(X,y,f)
    N=length(y);
    a=sym('a',[4 1]);
    F=[];
    for n=1:N
        F=[F; f(X(n,:),a)];
    end
    J=jacobian(F,a);
    J=matlabFunction(J);
    F=matlabFunction(F);
    a1=ones(4,1);
    tol=10^(-9);
    er=9;
    k=0;
    while er>tol
        k=k+1;
        a0=a1;        
        Jinv=pinv(J(a0(1),a0(2)));
        r=F(a0(1),a0(2),a0(3),a0(4))-y;
        r2=norm(r);
        a1=a0-Jinv*r;
        er=norm(a0-a1);        
        fprintf('%d\tr=%e\t%e\n',k,r2,er)
    end
end

function [a1 r2]=Model3(X,y,f)
    N=length(y);
    a=sym('a',[4 1]);
    F=[];
    for n=1:N
        F=[F; f(X(n,:),a)];
    end
    J=jacobian(F,a);
    J=matlabFunction(J);
    F=matlabFunction(F);
    a1=ones(4,1);
    tol=10^(-9);
    er=9;
    k=0;
    while er>tol
        k=k+1;
        a0=a1;     
        Jinv=pinv(J());
        r=y-F(a0(1),a0(2),a0(3),a0(4));
        a1=a0+Jinv*r;
        r2=norm(r);
        er=norm(a0-a1);   
        fprintf('%d\t%e\t%e\n',k,r2,er)
    end
end


% function generateData(f1,f2,f3,N)
% 	a=[1;2;3;4];        % ground truth parameters
% %    a=[4;3;2;1];        % ground truth parameters
%     rng(2);
%     X=2*rand(N,2)-1;    % N 2-D data points
% %    X(:,2)=X(:,2)*2;
%     c=1;
%     y1=zeros(N,1);
%     y2=zeros(N,1);
%     y3=zeros(N,1);
%     for n=1:N
%         y1(n)=f1(X(n,:),a)+c*(rand-0.5);
%         y2(n)=f2(X(n,:),a)+c*(rand-0.5);
%         y3(n)=f3(X(n,:),a)+c*(rand-0.5);
%     end
%     data3=[X y1];
%     data4=[X y2];
%     data5=[X y3];
%     save('data3.txt','data3','-ASCII')
%     save('data4.txt','data4','-ASCII')
%     save('data5.txt','data5','-ASCII')
%     
% end





