function [ Com, output ] = MCCD( A,v0,sm,isBst)
N=length(A);
if nargin<4
  isBst=false;
end
if nargin<3
  sm=1;%Cosine
end
if nargin<2
  v0=ones(N,1);
  isBst=true;
end
%W=PermMat(N);                     % permute the graph node labels
W=eye(N);                     % permute the graph node labels
A=W*A*W';
v0=W*v0;

output = '';

tic
Com=phase1(A,v0,sm);
%toc
%fprintf('phase1\tc=%2d\tNMI=%f Q=%f Qo=%f\n',max(Com),PSNMI(Com,v0),QFModul(Com,A),QFModul(v0,A));

phase1Time = toc;
fprintf('phase1\tc=%2d\tNMI=%f Q=%f Qo=%f\n',max(Com),PSNMI(Com,v0),QFModul(Com,A),QFModul(v0,A));
%phase1Output = sprintf('phase1\tc=%2d\tNMI=%f Q=%f Qo=%f\n',max(Com), PSNMI(Com, v0), QFModul(Com, A), QFModul(v0, A));
%output = strcat(output, phase1Output, sprintf(' Phase1 Time: %f seconds\n', phase1Time));
    

% dessin(A,v0,Com);
% pause
%  [Com]=Merge(A,Com,v0,isBst);%false, true
tic
Com=Merge2(A,Com,max(v0),v0);
%toc
%fprintf('phase2\tc=%2d\tNMI=%f Q=%f Qo=%f\n',max(Com),PSNMI(Com,v0),QFModul(Com,A),QFModul(v0,A));

phase2Time = toc;
fprintf('phase2\tc=%2d\tNMI=%f Q=%f Qo=%f\n',max(Com),PSNMI(Com,v0),QFModul(Com,A),QFModul(v0,A));
phase2Output = sprintf('c=%2d\tNMI=%f Q=%f Qo=%f\n',max(Com), PSNMI(Com, v0), QFModul(Com, A), QFModul(v0, A));
output = strcat(output, phase2Output, sprintf(' Time: %f seconds\n', phase2Time+phase1Time));   

% dessin(A,v0,Com);
Com=Com'*W;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Com=phase1(A,v0,sm)
N=length(A);
Com=1:N;
alpha=2;
betha=1;
%M=initSimMat4LPA(A,sm,alpha,betha);
M = optimized_modified_cosine_similarity(A);
mm=max(M);
for i =1:N
  x=find(abs(M(:,i)-mm(i)) < 0.00001);
  for j =1:length(x)
    y=find(Com==Com(x(j)));
    Com(y)=Com(i);
  end
end
[~,~,Com]=unique(Com,'stable');
end

function M = optimized_modified_cosine_similarity(A)
    % Ensure the matrix is in double precision for accurate computation
    A = double(A);
    
    % Get the number of nodes
    N = size(A, 1);
    
    % Precompute the sums for all rows
    row_sums = sum(A, 2);
    
    % Precompute the pairwise dot products
    dot_products = A * A';
    
    % Initialize the similarity matrix
    M = zeros(N);
    
    % Calculate the modified cosine similarity matrix
    for i = 1:N
        for j = i+1:N
            s = dot_products(i, j);
            t = sqrt(row_sums(i) * row_sums(j));
            M(i, j) = (s + A(i, j) * 2) / (t + 1);
            M(j, i) = M(i, j); % Symmetric matrix
        end
    end
end

function [Com]=Merge2(A,Com,nbC,v0)
mc=max(Com);
M=zeros(mc);
C=cell(1, mc);
E=zeros(mc);
nb=zeros(mc,1);
for i=1:mc
  nb(i) = sum(Com==i);
  C{i} = find(Com==i);
end
for i=1:mc
  for j=i+1:mc
    E(i,j) = sum(sum(A(C{i},C{j})));
    E(j,i)=E(i, j);
    M(i, j)=E(i,j)*(1/(nb(i)*nb(j))+1/min(nb(i),nb(j))^2);
    M(j,i)=M(i, j);
  end
end
while mc>nbC
  [mx,imax]=max(M(:));
  c1=ceil(imax/mc);
  %c2=mod(imax,mc)+1;
  c2=find(abs(M(:,c1)-mx) < 0.001);
  c2=c2(1);
  nb(c1)=nb(c1)+nb(c2);
  aa=C{c1};
  bb=C{c2};
  C{c1}=[aa; bb];
  E(c1,:)=(E(c1,:)+E(c2,:));
  E(:,c1)=E(c1,:);
  E(c1,c1)=0;
  aa=1 ./(nb(c1)*nb) + 1 ./ min(nb(c1),nb).^2;
  M(:, c1)=E(:, c1) .* aa;
  M(c1, c1)=0;
  M(c1 , :)=M(:, c1);
  nb(c2)=[];
  C(c2) = [];
  E(c2,:)=[];
  E(:,c2)=[];
  M(c2,:)=[];
  M(:,c2)=[];
%   for i=1:length(C)
%     Com(C{i})=i;
%   end
%   fprintf('phase2\tc=%2d\tNMI=%f Q=%f Qo=%f\n',max(Com),PSNMI(Com,v0),QFModul(Com,A),QFModul(v0,A));
  mc=mc-1;
end
for i=1:nbC
  Com(C{i})=i;
end
% fprintf('phase2\tc=%2d\tNMI=%f Q=%f Qo=%f\n',max(Com),PSNMI(Com,v0),QFModul(Com,A),QFModul(v0,A));
end


function Q = PSNMI(V,V0)
% function Q=PSNMI(V,V0)
% Normalized Mutual Information (NMI)
%
% An implementation of Normalized Mutual Information (NMI)
% by Erwan Le Martelot. The NMI measure shows 
% the similarity between two partitions. Max similarity is 1
% and min similarity is 0. For details see Danon, Leon, et al. 
% "Comparing community structure identification." Journal of 
% Statistical Mechanics: Theory and Experiment 2005.09 (2005): P09008.
%
% INPUT
% V:        N-by-1 matrix describes 1st partition
% V0:        N-by-1 matrix describes 2nd partition
%
% OUTPUT
% Q:         The Normalized Mututal Information between V and V0
%
% EXAMPLE
% [A,V0]=GGGirvanNewman(32,4,12,4,0);
% VV=GCAFG(A,[0.2:0.5:1.5]);
% Kbst=CNModul(VV,A);
% V=VV(:,Kbst);
% Q=PSNMI(V,V0);
%
if length(V) ~= length(V0)
    error('The two lists have different lengths');
end

% Number of nodes
n = length(V);

% Number of values in each list
nc1 = length(unique(V));
nc2 = length(unique(V0));

% Terms initialisation
term1 = 0;
term2 = 0;
term3 = 0;
    
use_full_matrix = (nc1*nc2 < 1000000);
% Computing terms using full matrix representation
if use_full_matrix
        
    % Build the confusion matrix
    c = zeros(nc1,nc2);
    for i=1:n
        c(V(i),V0(i)) = c(V(i),V0(i)) + 1;
    end
    sumci = sum(c,2);
    sumcj = sum(c,1);

    % Terms computing
    for i=1:nc1
        for j=1:nc2   
            if c(i,j) > 0
                term1 = term1 + ( c(i,j) * log((c(i,j)*n) / (sumci(i)*sumcj(j)) ) );
            end
        end
        term2 = term2 + ( sumci(i) * log(sumci(i)/n) );
    end
    for j=1:nc2
        term3 = term3 + ( sumcj(j) * log(sumcj(j)/n) );
    end
        
% Sparse representation
else
        
    % Build the confusion matrix
    c = sparse(nc1,nc2);
    sumci = zeros(nc1,1);
    sumcj = zeros(nc2,1);
    for i=1:n
        c(V(i),V0(i)) = c(V(i),V0(i)) + 1;
        sumci(V(i)) = sumci(V(i)) + 1;
        sumcj(V0(i)) = sumcj(V0(i)) + 1;
    end

    % Terms computing
    for i=1:nc1
        cols = find(c(i,:));
        for k=1:length(cols)
            j = cols(k);
            term1 = term1 + ( c(i,j) * log( (c(i,j)*n) / (sumci(i)*sumcj(j)) ) );
        end
        term2 = term2 + ( sumci(i) * log(sumci(i)/n) );
    end
    for j=1:nc2
        term3 = term3 + ( sumcj(j) * log(sumcj(j)/n) );
    end
        
end
     
%Result
Q = (-2 * term1) / (term2 + term3);
Q=Q';
end


function Q = QFModul(V,A)
% function Q = QFModul(V,A)
% Modularity  quality function
%
% Computes the classical Newman-Girvan modularity. The code for 
% its evaluation, listed below, was written by E. le Martelot.
% See http://en.wikipedia.org/wiki/Modularity_%28networks%29
% 
% INPUT
% V:      N-by-1 matrix describes a partition
% A:      adjacency matrix of graph
%
% OUTPUT
% Q:      the modularity of V given graph (with adj. matrix) A
% 
% EXAMPLE
% [A,V0]=GGGN(32,4,16,0,0);
% VV=GCAFG(A,[0.2:0.5:1.5]);
% Kbst=CNModul(VV,A);
% V=VV(:,Kbst);
% Q = QFModul(V,A)
%
m = sum(sum(A));
Q = 0;
COMu = unique(V);
for j=1:length(COMu)
    Cj = find(V==COMu(j));
    Ec = sum(sum(A(Cj,Cj)));
    Et = sum(sum(A(Cj,:)));
    if Et>0
        Q = Q + (Ec/m)-(Et/m)^2;
    end
end
end