function [w, wsave, sum1 ,err, out] = train(w ,x,desired,num_lay,num_nodes,actf,dactf,calcj,calcm,wch,wsave,out,num_in,num_out,M,sum1,err,w2 )
for i=1:length(x);
    out{1}=x(i,:);
    for k=1:num_lay-2
        net{k}=out{k}*w{k};
       tempo=1;
        tempo(1,2:num_nodes(k+1,1)+1)=actf(net{k});
        out{k+1}=tempo;
    end
        net{num_lay-1}=out{num_lay-1}*w{num_lay-1};
        out{num_lay}=actf(net{num_lay-1});
        
% err_tot(sum1,1)=(1/(2*num_out))*sum((desired(i,:)-out{num_lay}).^2,2);
err=err+(1/(2*length(x)))*sum((desired(i,:)-out{num_lay}).^2,2);


dtot_dout{num_lay-1}=-(desired(i,:)-out{num_lay});
dout=cellfun(dactf,out,'un',0);
dout_dnet{num_lay-1}=dout{num_lay};
dw{num_lay-1}=out{num_lay-1}'*(dtot_dout{num_lay-1}.*dout_dnet{num_lay-1});
for k=num_lay-2:-1:1
%     tempdo=dout{k+1};
    tempw=w{k+1}(2:num_nodes(k+1,1)+1,:);
    dtot_dout{k}=(dtot_dout{k+1}.*dout_dnet{k+1})*tempw';
    dout_dnet{k}=dout{k+1}(1,2:end);
    tempd2=dtot_dout{k}.*dout_dnet{k};
    tempo2=out{k};
    dw{k}=tempo2'*tempd2;
end
% sum1=sum1+1;
% w2=cellfun(@(x,y) x+y,w2,dw,'un',0);
% if rem(sum1,10) == 1
wsave{1}=wsave{2};
wsave{2}=wsave{3};
grad=cellfun(calcj,dw,'un',0);
moment=cellfun(calcm,wsave{2},wsave{1},'un',0);
wsave{3}=cellfun(wch,w,grad,'un',0);
w=wsave{3};
% clear w2
% sum1=1;
%     for k=1:num_lay-1
%         w2{k}=zeros(num_nodes(k,1)+1,num_nodes(k+1,1));
%     end
% end
% w=cellfun(@(x) x+(-1 + (1+1)*rand(size(x,1),size(x,2)))*.001,w,'un',0);
end
end

