%% 
% amri_sig_corr
%    returns a p-by-p matrix containing the pairwise linear correlation 
%    coefficient between each pair of columns in the n-by-p matrix A, 
%    or return correlation verctor of coresponding columns of A and B, 
%    or return correlation matrix of the pairwise columns of A and B.
%
% Usage
%   [R,pval] = amri_sig_corr(A);
%   [R,pval] = amri_sig_corr(A,B);
%
% Inputs
%   A: n-by-p data matrix
%   B: n-by-p input matrix
%
% Keywords:
%   mode: 'Auto' or 'Cross'. 'Auto' mode returns the the correlation
%   coefficient of the coresponding columns of matrix A and B. 'Cross' mode
%   returns the cross correlation matrix of A and B.
%
% Output
%   R: p-by-p correlation matrix or a vector of length n
%   pval: p value for Pearson correlation
%
% Version 
%  1.04

%% DISCLAIMER AND CONDITIONS FOR USE:
%     This software is distributed under the terms of the GNU General Public
%     License v3, dated 2007/06/29 (see http://www.gnu.org/licenses/gpl.html).
%     Use of this software is at the user's OWN RISK. Functionality is not
%     guaranteed by creator nor modifier(s), if any. This software may be freely
%     copied and distributed. The original header MUST stay part of the file and
%     modifications MUST be reported in the 'MODIFICATION HISTORY'-section,
%     including the modification date and the name of the modifier.
%

%% MODIFICATION HISTORY
% 1.01 - 07/06/2010 - ZMLIU - compute correlation between two input vectors
%        16/11/2011 - JAdZ  - v1.01 included in amri_eegfmri_toolbox v0.1
% 1.02 - 10/18/2013 - HGWEN - for two input matrix A and B, calculate the 
%                     correlation of the coresponding columns of A and B.
% 1.03 - 04/20/2015 - HGWEN - convect input vectors into column vectors.
% 1.04 - 07/28/2015 - HGWEN - return p-value for Pearson Correlation

function [R, pval] = amri_sig_corr(A,B,varargin)

if nargin<1
    eval('help amri_sig_corr');
    return
end

%% defaults
mode = 'cross';
df = 0;

%%
for i = 1:2:size(varargin,2) 
    Keyword = varargin{i};
    Value   = varargin{i+1};
    if strcmpi(Keyword,'mode')
        mode = Value;
    elseif strcmpi(Keyword,'df')
        df = Value;
    else
        warning(['amri_sig_corr(): unknown keyword ' Keyword]);
    end
end

%%
if (nargin > 1)&&(~isempty(B))
    if isvector(A)
        A = A(:);
    end
    if isvector(B)
        B = B(:);
    end
    if all(size(A)== size(B)) == 0 && strcmpi(mode,'auto') == 1
        error('amri_sig_corr(): A and B must have the same size.');
    end
    p=size(A,2);
    n = size(A,1);
    for i=1:p
        A(:,i)=A(:,i)-mean(A(:,i));
        nn = norm(A(:,i));
        if nn>0
            A(:,i)=A(:,i)/nn;
        end     
    end
    
    for i = 1 : size(B,2)
        B(:,i)=B(:,i)-mean(B(:,i));
        nn = norm(B(:,i));
        if nn> 0
            B(:,i)=B(:,i)/nn;
        end
    end
    
    if (nargin > 1)&&(strcmpi(mode,'auto'))
        R = sum(A.*B,1);
    elseif (nargin > 1)&&(strcmpi(mode,'cross'))
        R = (A')*B;
    end
    
    tval = R.*sqrt(n-2)./sqrt(1-R.^2);
    pval=2.*(1-tcdf(abs(tval),n-2));
    return;
end

p = size(A,2);
n = size(A,1);
for i=1:p
    A(:,i)=A(:,i)-mean(A(:,i));
    nn = norm(A(:,i));
    if nn>0
        A(:,i)=A(:,i)/nn;
    end
end
R=A'*A;

if df > 0
    n = df;
end
tval = R.*sqrt(n-2)./sqrt(1-R.^2);
pval=2.*(1-tcdf(abs(tval),n-2));

end
