1;
function [pdirection,pcomponent] = pca()
  clear;
  load optdigit.data;
  iris = optdigit;
  m = size(iris)(1);
  n = size(iris)(2);
  X = iris(1:m,1:n-1);
  Y = iris(1:m,n:n);
  X = bsxfun(@minus, X, mean(X));
  
  [U, S, V] = svd (X, 0);
  
  %Part 1a
  pdirection = V(:,1:2);
  pcomponent= U*S(:,1:2);
  
  % Part 1b
  for i = [0,0.5,1]
    temp = diag(S).^i;
    G = U*diag(temp);
    temp = diag(S).^(1-i);
    H = (diag(temp)*V')';
    G = G(:,1:2);
    H = H(:,1:2);
    figure;
    hold on;
    for j = [1:size(G)(1)]
      temp1 = [0,G(j,1)];
      temp2 = [0,G(j,2)];
      plot(temp1, temp2,"-");
    endfor
   
    for j = [1:size(H)(1)]
      temp1 = [0,H(j,1)];
      temp2 = [0,H(j,2)];
      plot(temp1, temp2,"-");
    endfor
    
    if i == 0
      print "1.png";
    elseif i == 0.5
      print "2.png";
    else
      print "3.png";
    endif
  endfor
  
  %Part 1c
  VNEW = zeros(size(V));
  
  for i = [1:size(V)(2)]
    h1= figure;
    hold on;
    VNEW(:,i) = V(:,i); 
    XNEW = U*S*(VNEW');
    for j = [1:size(X)(1)]
      ERROR(j) = norm(X(j,:)-XNEW(j,:));
    endfor
    INDEX = [1:1:size(X)(1)];
    plot(INDEX,ERROR,"-");
    filename =strcat('error_' ,int2str(i));
    saveas(h1, filename,'png');
  endfor
  
  %Part 1d
  VAR = diag(S).^2;;
  INDEX = [1:1:size(VAR)(1)];
  h1 = figure;
  hold on;
  plot(INDEX, VAR',"-");
  saveas(h1, "eigenVar",'png');
endfunction

% KPCA implementation
% We cannot calculate the pdirections

function [pcomponent] = kpca_fun()
  clear;
  
  %uncomment for iris data set
  %load iris.data;
  %data = iris;
  
  %uncomment for optdigit data set
  load optdigit.data;
  data = optdigit;
    
  m = size(data)(1);
  n = size(data)(2);
  X = data(1:m,1:n-1);
  Y = data(1:m,n:n);
  X = bsxfun(@minus, X, mean(X));
  
  %taking p = 2
  for i = [1:size(X)(1)]
    for j = [1:size(X)(1)]
      %K(i,j) = poly_kernel(X(i,:),X(j,:),2); %uncomment for polynomial kernel, taking p = 2;
      K(i,j) =  radial_kernel(X(i,:),X(j,:),2);     %uncomment for radial kernel, taking sigma = 2;
    endfor
  endfor
  
  [EVECTOR EVALUE]=eig(K);
  L = diag(EVALUE);
  
  [sorted index]=sort(L,'descend');
    
  W = zeros(size(EVECTOR));
  for i = [1:size(index)]
    W(:,i) = EVECTOR(:,index(i));
  endfor
  
  Projected_matrix = K*W;
  pcomponent = Projected_matrix;
  x_val = Projected_matrix(:,1);
  y_val = Projected_matrix(:,2);
  figure;
  plot(x_val,y_val,"ro");
  print "kpca_pcomponent.png";
  
  % 2D part 
  INDEX = [1:1:size(sorted)];
  figure;
  plot(INDEX,sorted',"b.-");
  print "eigen_value_kpca.png";
endfunction

%Simple PCA Kernel Function
% xi = data point (row vector) 
% xj = data point (row vector)
function val = pca_kernel(xi,xj)
  val = xi*xj';
end

%Polynomial Kernel Function
% xi = data point (row vector) 
% xj = data point (row vector)
% p is parameter value 
function val = poly_kernel(xi,xj,p)
  val = (xi*xj'+1)^p;
end

%Radial basis function kernel
% xi = data point (row vector) 
% xj = data point (row vector)
% sigma = parameter value cannot be 0 
function val = radial_kernel(xi,xj,sigma)
  if sigma == 0
    disp("Sigma cannot be 0");
    quit(1);
  end
  diff = xi-xj;
  val = exp(-(diff*diff')/(2*sigma*sigma));
end
