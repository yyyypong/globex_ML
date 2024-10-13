function main
    clear, close all
    sym x;
    a=sym('a',[3 1]);

    f0=@(x,a)a(1)*x+a(2);
    f1=@(x,a)a(1)*x.^2+a(2)*x+a(3);
    f2=@(x,a)a(1)*exp(a(2)*x)+a(3);
    
%     rand('seed',1)
    figure(2)
    N=90;
%     generateData(N);
    e=zeros(3);
    data=zeros(N,2,3);
    data(:,:,1)=load('data0.txt');
    data(:,:,2)=load('data1.txt');
    data(:,:,3)=load('data2.txt');
        
    figure(1)

    x=[min(data(:,1,1)):0.01:max(data(:,1,1))];
    
    for k=1:3             % for 3 sets of data
        [a0 er]=LinearModel(data(:,1,k),data(:,2,k),f0);
        fprintf('Model 1: %9.3f\t%9.3f\t%9.6f\n',a0(1),a0(2),er);
        subplot(3,3,k)
        plot(x,f0(x,a0),'r')
        hold on
        stem(data(:,1,k),data(:,2,k),'fill','r')
        hold on
        stem(data(:,1,k),f0(data(:,1,k),a0),'r')
        hold off
    
        [a1 er]=QuadraticModel(data(:,1,k),data(:,2,k),f1);
        fprintf('Model 2: %9.3f\t%9.3f\t%9.3f\t%9.6f\n',a1(1),a1(2),a1(3),er);
        subplot(3,3,3+k)
        plot(x,f1(x,a1),'g')
        hold on
        stem(data(:,1,k),data(:,2,k),'fill','g')
        hold on
        stem(data(:,1,k),f1(data(:,1,k),a1),'g')
        hold off
        
        [a2 er]=ExponentialModel(data(:,1,k),data(:,2,k),f2);
        fprintf('Model 3: %9.3f\t%9.3f\t%9.3f\t%9.6f\n',a2(1),a2(2),a2(3),er);
        subplot(3,3,6+k)
        plot(x,f2(x,a2),'b');
        hold on
        stem(data(:,1,k),data(:,2,k),'fill','b')
        hold on
        stem(data(:,1,k),f2(data(:,1,k),a2),'b')
        hold off
    end
end

function a=LinearModel0(x,y)
    a=zeros(2,1);
    mx=mean(x);
    my=mean(y);
    cxy=mean(x.*y)-mx*my;
    cxx=mean(x.*x)-mx*mx;
    a(1)=cxy/cxx;
    a(2)=my-a(1)*mx;
end

function a=LinearModel1(x,y)
    a=zeros(2,1);
    S=length(x);
    Sx=0;
    Sy=0;
    Sxy=0;
    Sxx=0;
    for i=1:S
        Sx=Sx+x(i);
        Sy=Sy+y(i);
        Sxy=Sxy+x(i)*y(i);
        Sxx=Sxx+x(i)^2;
    end
    d=S*Sxx-Sx^2;
    a(1)=(S*Sxy-Sx*Sy)/d;
    a(2)=(Sxx*Sy-Sx*Sxy)/d;
end

function [a1 er]=LinearModel(x,y,f)
    eta=0.001;
    tol=0.00001;
    N=length(x);
    a=sym('a',[2 1]);
    F=[];
    for i=1:N
        F=[F; f(x(i),a)];
    end
    J=jacobian(F,a);
    J=matlabFunction(J); 
    F=matlabFunction(F);
    a1=ones(2,1);
    k=0;
    da=1;
    while da>tol
        k=k+1;
        a0=a1;
        r=y-F(a0(1),a0(2));
        J0=J();
         g=-J0.'*r;
        a1=a0-eta*g;   
%        Jinv=pinv(J0);
%        a1=a0+Jinv*r;
        er=norm(r);
        da=norm(a0-a1);
    end
end

function [a1 er]=QuadraticModel(x,y,f)
    eta=0.0001;
    tol=0.00001;
    N=length(x);
    a=sym('a',[3 1]);
    F=[];
    for i=1:N
        F=[F; f(x(i),a)];
    end
    J=jacobian(F,a);
    J=matlabFunction(J); 
    F=matlabFunction(F);
    a1=ones(3,1);
    k=0;
    da=1;
    while da>tol
        k=k+1;
        a0=a1;
        r=y-F(a0(1),a0(2),a0(3));
        J0=J();
        g=-J0.'*r;
        a1=a0-eta*g;      
%        Jinv=pinv(J0);
%        a1=a0+Jinv*r;
        er=norm(r);
        da=norm(a0-a1);
    end
end


function [a1 er]=ExponentialModel(x,y,f)
    eta=0.000001;
    tol=0.00001;
    N=length(x);
    a=sym('a',[3 1]);
    F=[];
    for i=1:N
        F=[F;f(x(i),a)];
    end
    J=jacobian(F,a);
    J=matlabFunction(J); 
    F=matlabFunction(F);
    a1=ones(3,1);
    k=0;
    da=1;
    while da>tol
        k=k+1;
        a0=a1;
        r=y-F(a0(1),a0(2),a0(3));
        J0=J(a0(1),a0(2));
        g=-J0.'*r;
        a1=a0-eta*g;        
%        Jinv=pinv(J0);
%        a1=a0+Jinv*r;
        er=norm(r);
        da=norm(a0-a1);
    end
end


% function generateData(N)
%     x=sym('x',[1 1]);
%     
%     a(1)=4.5;
%     a(2)=7;
%     a(3)=15;
%     b(1)=2;
%     b(2)=0.9;
%     b(3)=30;
%     c(1)=31;
%     c(2)=50;
%     
%     f0=c(1)*x+c(2);
%     f1=a(1)*x^2+a(2)*x+a(3);    
%     f2=b(1)*exp(b(2)*x)+b(3);   
%     f0=matlabFunction(f0);
%     f1=matlabFunction(f1);
%     f2=matlabFunction(f2);
%    
%     x=zeros(N,1);
%     y0=zeros(N,1);
%     y1=zeros(N,1);
%     y2=zeros(N,1);
%     data0=zeros(N,2);
%     data1=zeros(N,2);
%     data2=zeros(N,2);
%     for i=1:N
%         x(i)=rand*8-2;
%         y0(i)=f0(x(i));
%         y1(i)=f1(x(i));
%         y2(i)=f2(x(i));
%         y0(i)=y0(i)+rand*30-15;
%         y1(i)=y1(i)+rand*30-15;
%         y2(i)=y2(i)+rand*30-15;
%         data0(i,1)=x(i);
%         data0(i,2)=y0(i);
%         data1(i,1)=x(i);
%         data1(i,2)=y1(i);
%         data2(i,1)=x(i);
%         data2(i,2)=y2(i);
%     end
%     xmin=min(x);
%     xmax=max(x);
%     t=[xmin:0.01:xmax];
% 	plot(t,f0(t),'r',t,f1(t),'g', t,f2(t),'b')
% 	hold on
%     stem(x,y0,'fill','r');
%     hold on
%     stem(x,y1,'fill','g');
%     hold on
%     stem(x,y2,'fill','b');
%     hold off
%     save('data0.txt','data0','-ASCII')
%     save('data1.txt','data1','-ASCII')
%     save('data2.txt','data2','-ASCII')
% end
