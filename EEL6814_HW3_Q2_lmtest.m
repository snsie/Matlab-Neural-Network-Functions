clc;
clear;


Noise=.2;
num_lay=4;
% num_nodes(2,1)=9;
% num_nodes(3,1)=3;
% num_nodes(4,1)=1;
% num_nodes(5,1)=60;
nodes_per_hlay=[9 3];
nstep=2;
usave=.5;
u=usave;
numitr=300;
M=0;
ustep=10;
numruns=10;

act_type='logistic';

wch= @(x,y) x-y;
calcm=@(x,y) M*(x-y);
calcj=@(x) x.*nstep;
splicemat=@(x) x(:);
in1=importdata('C:\Users\Sns08j\Downloads\hmw3_2016\hmw3\Sleepdata1 Input.asc');
out1=importdata('C:\Users\Sns08j\Downloads\hmw3_2016\hmw3\Sleepdata1 Desired.asc');
in2=importdata('C:\Users\Sns08j\Downloads\hmw3_2016\hmw3\Sleepdata2 Input.asc');
out2=importdata('C:\Users\Sns08j\Downloads\hmw3_2016\hmw3\Sleepdata2 Desired.asc');

x=in1.data;
desired=out1.data;
xtest=in2.data;
destest=out2.data;
for i=1:size(xtest,2)
xtest(:,i)=xtest(:,i)-mean(xtest(:,i));
xtest(:,i)=xtest(:,i)/(std(xtest(:,i)));
end
for i=1:size(x,2)
x(:,i)=x(:,i)-mean(x(:,i));
x(:,i)=x(:,i)/(std(x(:,i)));
end
[savet_corr ,savetr_corr ,saveerr ,savefract_corr ,savefract_corr_test ,wremtot] = Run_Grad(act_type, num_lay,nodes_per_hlay,nstep,M,numitr,numruns,x,xtest,desired,destest );
% [savet_corr ,savetr_corr ,saveerr ,savefract_corr ,savefract_corr_test ,wremtot] = Run_LM(act_type, num_lay,nodes_per_hlay,usave,ustep,numitr,numruns,x,xtest,desired,destest );