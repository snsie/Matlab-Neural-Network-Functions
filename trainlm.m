function [w, wsave, sum1 ,err, u] = trainlm(w ,x,desired,num_lay,num_nodes,actf,dactf,wch,wsave,out,sum1,dw,splicemat,numw,J,u,temp_numw,initdw,ustep)
e=zeros(size(x,1)*size(desired,2),1);
e2=zeros(size(x,1)*size(desired,2),1);
    m=1;
    err=0;
for i=1:size(x,1)
    out{1}=x(i,:);
    for k=1:num_lay-2
        net{k}=out{k}*w{k};
       tempo=1;
        tempo(1,2:num_nodes(k+1,1)+1)=actf(net{k});
        out{k+1}=tempo;
    end
        net{num_lay-1}=out{num_lay-1}*w{num_lay-1};
        out{num_lay}=actf(net{num_lay-1});

err=err+(1/(2*length(x)))*sum((desired(i,:)-out{num_lay}).^2,2);
for j=1:size(out{num_lay},2)
    outtemp=out{num_lay}(j);
    e(i*size(desired,2)+j-size(desired,2),1)=desired(i,j)-outtemp;
% dtot_dout{num_lay-1}=-(desired(i,j)-outtemp);
dout{num_lay}=dactf(outtemp);
dout(1:num_lay-1)=cellfun(dactf,out(1:num_lay-1),'un',0);
dout_dnet{num_lay-1}=dout{num_lay};
% e{num_lay-1)=dtot_dout{num_lay-1}.*dout_dnet{num_lay-1};
dw{num_lay-1}(:,j)=-out{num_lay-1}*dout_dnet{num_lay-1};

tempw=w{num_lay-1}(2:end,j);
for k=num_lay-2:-1:1
%      dtot_dout{num_lay-2}=(dtot_dout{num_lay-1}.*dout_dnet{num_lay-1})*tempw';
    dout_dnet{k}=(dout_dnet{k+1})*tempw';
%      tempw=w{k}(2:end,:);
    dout_dnet{k}=dout{k+1}(1,2:end).*dout_dnet{k};
% dw{k}=-out{k}'*(dtot_dout{k}.*dout_dnet{k});
dw{k}=-out{k}'*(dout_dnet{k});
 tempw=w{k}(2:end,:);
end
J(i*size(desired,2)+j-size(desired,2),:)=cell2mat(cellfun(splicemat,dw,'un',0)');
dw=cellfun(initdw,dw,'un',0);
end
err=err+(1/(2*length(x)))*sum((desired(i,:)-out{num_lay}).^2,2);
end
I=eye(numw);  
E=sum(e.^2,1)/2;
testw=(J'*J+u*I)\J'*e;
wsave{1}{1}=reshape(testw(1:temp_numw{1}),[num_nodes(1)+1, num_nodes(2)]);

for j=2:num_lay-1
    wsave{1}{j}=reshape(testw(temp_numw{j-1}+1:[temp_numw{j}]),[num_nodes(j)+1, num_nodes(j+1)]);
%     sumw=sumw+temp_numw{j};
end

w2=cellfun(wch,w,wsave{1},'un',0);

for i=1:size(x,1);
    out{1}=x(i,:);
    for k=1:num_lay-2
        net{k}=out{k}*w2{k};
       tempo=1;
        tempo(1,2:num_nodes(k+1,1)+1)=actf(net{k});
        out{k+1}=tempo;
    end
        net{num_lay-1}=out{num_lay-1}*w2{num_lay-1};
        out{num_lay}=actf(net{num_lay-1});
         for j=1:size(desired,2)
        e2(i*size(desired,2)+j-size(desired,2),1)=desired(i,j)-out{num_lay}(j);
            end
end
E2=sum(e2.^2,1)/2;
while m<=5 && E2>E
   u=u*ustep;
   testw=(J'*J+u*I)\J'*e;
wsave{1}{1}=reshape(testw(1:temp_numw{1}),[num_nodes(1)+1, num_nodes(2)]);

for j=2:num_lay-1
    wsave{1}{j}=reshape(testw(temp_numw{j-1}+1:[temp_numw{j}]),[num_nodes(j)+1, num_nodes(j+1)]);
%     sumw=sumw+temp_numw{j};
end

w2=cellfun(wch,w,wsave{1},'un',0);

for i=1:size(x,1);
    out{1}=x(i,:);
    for k=1:num_lay-2
        net{k}=out{k}*w2{k};
       tempo=1;
        tempo(1,2:num_nodes(k+1,1)+1)=actf(net{k});
        out{k+1}=tempo;
    end
        net{num_lay-1}=out{num_lay-1}*w2{num_lay-1};
        out{num_lay}=actf(net{num_lay-1});
         for j=1:size(desired,2)
        e2(i*size(desired,2)+j-size(desired,2),1)=desired(i,j)-out{num_lay}(j);
         end
end
 E2=sum(e2.^2,1)/2;
  m=m+1;
end
if m>5 && E2 > E
   w=w2; 
end

if E2 <= E
w=w2;
u=u/ustep;
end
end

