function [savet_corr ,savetr_corr ,saveerr ,savefract_corr ,savefract_corr_test ,wremtot] = Run_LM(act_type, num_lay,nodes_per_lay,usave,ustep,numitr,numruns,x,xtest,desired,destest )

if strcmpi(act_type,'SoftPlus')==1
actf=@(x) log(1+exp(x));
dactf=@(x) 1./(1+exp(-x));
end
initdw=@(x) x-x;
if strcmpi(act_type,'Logistic')==1
actf = @(x) 1./(1+exp(-x));
dactf= @(x) x.*(1-x);
end
if strcmpi(act_type,'ReLU')==1
actf= @(x) double(ge(x,0).*x);
dactf=@(x) double(ge(x,0));
end
if strcmpi(act_type,'ArcTan')==1
actf = @(x) 2./(1+exp(-2*x))-1;
dactf= @(x) 1-x.^2;
end
num_nodes=zeros(num_lay,1);
num_nodes(2:end-1,1)=nodes_per_lay(:,1);
wch= @(x,y) x-y;
calcm=@(x,y) M*(x-y);
calcj=@(x) x.*nstep;
splicemat=@(x) x(:);
sum2=1;
for iruns=1:numruns
ix = randperm(size(x,1));
x=x(ix',:);
desired=desired(ix',:);
ixt = randperm(size(xtest,1));
xtest=xtest(ixt',:);
destest=destest(ixt',:);

xtest(:,2:end+1)=xtest;
xtest(:,1)=1;
x(:,2:end+1)=x;
x(:,1)=1;

num_out=size(desired,2);
% x=x(:,[1:7 9]);
num_in=size(x,2);
num_nodes(1,1)=num_in-1;
num_nodes(num_lay,1)=num_out;

if actf(.87)==.7014
for i=1:size(desired,1)
    for j=1:size(desired,2)
        if desired(i,j)==0
           desired(i,j)=-1; 
        end
    end
end
for i=1:size(destest,1)
    for j=1:size(destest,2)
        if destest(i,j)==0
           destest(i,j)=-1; 
        end
    end
end
end


dw=cell(1,num_lay-1);
w=cell(1,num_lay-1);
w2=cell(1,num_lay-1);
    for i=1:num_lay-1
        w{i}=zeros(num_nodes(i,1)+1,num_nodes(i+1,1));
        dw{i}=zeros(num_nodes(i,1)+1,num_nodes(i+1,1));
        w2{i}=zeros(num_nodes(i,1)+1,num_nodes(i+1,1));
    end


for i=1:num_lay-1
    net{i}=zeros(1,num_nodes(i+1,1));
    dtot_dout{i}=zeros(1,num_nodes(i+1,1));
    dout_dnet{i}=zeros(1,num_nodes(i+1,1));
end
for i=1:num_lay-1
out{i}=zeros(1,num_nodes(i,1)+1);
end
out{num_lay}=zeros(1,num_nodes(num_lay,1));
err=1;
sum1=1;
err_tot=zeros(numitr*length(x),1);
wsave=cell(3,1);
wcheck=cell(3,1);
 for i=1:num_lay-1
        wsave{1,1}{i}=zeros(num_nodes(i,1)+1,num_nodes(i+1,1));
        wsave{2,1}{i}=zeros(num_nodes(i,1)+1,num_nodes(i+1,1));
        wsave{3,1}{i}=zeros(num_nodes(i,1)+1,num_nodes(i+1,1));
        wcheck{1,1}{i}=zeros(num_nodes(i,1)+1,num_nodes(i+1,1));
        wcheck{2,1}{i}=zeros(num_nodes(i,1)+1,num_nodes(i+1,1));
        wcheck{3,1}{i}=zeros(num_nodes(i,1)+1,num_nodes(i+1,1));
        
 end
sum_fails=-1;
fract_corr_test=0;
numw=0;
temp_numw=cellfun(@(x) numel(x),w,'un',0);
temp_numw{num_lay-1}=temp_numw{num_lay-1};
for i=1:num_lay-1
   numw=temp_numw{i}+numw; 
   temp_numw{i}=numw;
end
    hfract_corr_test=0;
    hfract_corr=0;
    saveiterr=zeros(numitr,1);
zeroline=zeros(numitr,1);
savefact=zeros(3,1);

 for i=1:num_lay-1
         w{i}=(-1 + (1+1)*rand(num_nodes(i,1)+1,num_nodes(i+1,1))).*sqrt(6/(size(w{i},1)-1+size(w{i},2)));
 end

u=usave;
sum1=1;
sum_fails=sum_fails+1;

for j=1:numitr 
    err=0;
   J=zeros(size(x,1)*size(desired,2),numw);
    ix=randperm(size(x,1));
x=x(ix',:);
desired=desired(ix',:);
[w, wsave, sum1 ,err, u] = trainlm(w ,x,desired,num_lay,num_nodes,actf,dactf,wch,wsave,out,sum1,dw,splicemat,numw,J,u,temp_numw,initdw,ustep);

saveiterr(j)=err;
    [ fract_corr ] = cross_validate( x,desired,w,num_lay,out,num_nodes,actf);
 savetrain_corr(j)=fract_corr;

[ fract_corr_test ] = cross_validate( xtest,destest,w,num_lay,out,num_nodes,actf);
save_corr_test(j)=fract_corr_test;
 if fract_corr_test>hfract_corr_test 
   hfract_corr_test=fract_corr_test; 
   wrem=w;
 end
% err
% fract_corr_test
% fract_corr

if fract_corr > hfract_corr
hfract_corr=fract_corr;
end

end
Run_Num=iruns
wremtot{iruns}=wrem;
    savet_corr(iruns,:)=save_corr_test;
    savetr_corr(iruns,:)=savetrain_corr;
    saveerr(iruns,:)=saveiterr;
    savefract_corr(iruns)=hfract_corr;
savefract_corr_test(iruns)=hfract_corr_test;
sum2=sum2+1;
end

end

