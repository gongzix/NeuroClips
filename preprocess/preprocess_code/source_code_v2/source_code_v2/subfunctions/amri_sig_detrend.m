%% amri_sig_detrend
% remove polynormial functions from the input time series
%
% Version 0.01

%% History
% 0.01 - 06/03/2014 - ZMLIU - don't call amri_sig_nvr
%                           - noted that detrend may distort the signal
%  

%%
function ots = amri_sig_detrend(its, polyorder)
if nargin<1
    eval('help amri_sig_detrend');
    return
end

if nargin<2
    polyorder=1;
end

polyorder = round(polyorder);
if polyorder<0
    polyorder=0;
end

[nr,nc]=size(its);
its=its(:);
its=its-mean(its);

if polyorder>0
    nt=length(its);
    
    poly=zeros(nt,polyorder+1);
    for i=1:polyorder+1
        poly(:,i)=(1:nt).^(i-1);
        poly(:,i)=poly(:,i)./norm(poly(:,i));
    end
    p=double(poly)\double(its);
    trend=double(poly)*p;
    ots=its-trend;
    
%     poly=zeros(nt,polyorder);
%     for i=1:polyorder
%         poly(:,i)=(1:nt).^i;
%         poly(:,i)=poly(:,i)./norm(poly(:,i));
%     end
%     ots=amri_sig_nvr(its,poly);
 
    ots=reshape(ots,nr,nc);
else
    ots=its;
end


