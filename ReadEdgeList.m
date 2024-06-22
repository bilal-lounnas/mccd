function [A,V0]=ReadEdgeList(EdgeFile,PartitionFile)
E=load(EdgeFile);
V0=load(PartitionFile);
V0=V0(:,2);
M=size(E,1);
N=max(max(E));
A=zeros(N,N);
for m=1:M
u=E(m,1);
v=E(m,2);
A(u,v)=1;
A(v,u)=1;
end
end