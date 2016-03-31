function [ fract_corr ] = cross_validate( x,desired,w,num_lay,out,num_nodes,actf)
 for i=1:size(x,1);
    out{1}=x(i,:);
    for k=1:num_lay-2
        net{k}=out{k}*w{k};
       tempo=1;
        tempo(1,2:num_nodes(k+1,1)+1)=actf(net{k});
        out{k+1}=tempo;
    end
        net{num_lay-1}=out{num_lay-1}*w{num_lay-1};
        out{num_lay}=actf(net{num_lay-1});
        saveo(i,:)=out{num_lay};
 
 end
saveo=double(ge(saveo,.5));

[as ides]=max(desired,[],2);
[as io]=max(saveo,[],2);
% for i=1:size(x,1)
%     if saveo(i,1) >=0
%             saveo(i,1)= 1;
%         end
%         if saveo(i,1) <0
%             saveo(i,1)= -1;
%         end 
% end
% 

sumcorr=0;
suminc=0;
for i=1:size(x,1)
      if  ides(i,1) == io(i,1)
         sumcorr=sumcorr+1;
      end
%       if  saveo(i,1) ==  desired(i,1)
%          sumcorr=sumcorr+1;
%       end
end
fract_corr=sumcorr/size(x,1); 
end

