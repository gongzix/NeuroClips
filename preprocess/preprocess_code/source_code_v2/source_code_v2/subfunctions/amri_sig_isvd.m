% inputs:
%   X: p-by-n, p is dimension, n is number of samples. X = U*S*V';
%   U0: p-by-k0, k0 is the number of principle components
%   S0: k0-by-k0, standard deviation
%   percp: percentage of variance to keep, in range (0 1];
%
% outputs:
%   U: p-by-k, k is the number of principle components
%   S: k-by-k, standard deviation
%   k: updated number of components by keeping percp*100% variance;

% example:
% X0 = randn(1000,500);
% [U0,S0] = svd(X0,0);
% 
% X = randn(1000,200);
% [U,S] = amri_sig_isvd(X,'var',0.99, 'init',{U0,S0});
% 
% [U1, S1] = amri_sig_isvd([X0 X],'var',0.99);
% 
% junk0 = diag(S);
% junk1 = diag(S1);

% reference:
% Zhao, H., Yuen, P. C., & Kwok, J. T. (2006).
% A novel incremental principal component analysis and its application for face recognition. 
% IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 36(4), 873-886.


%% history
% 0.01 - HGWEN - 09/14/2017 - original file
% 0.02 - HGWEN - 11/16/2017 - change "amri_sig_svd" to "svd" in line 74

%% svd updating
function [U,S, k] = amri_sig_isvd(X, varargin)
    % check inputs
    if nargin<1
        eval('help amri_sig_isvd');
        return
    end

    % defaults
    percp = 0;
    init_flag = 0;

    % Keywords
    for iter = 1:2:size(varargin,2) 
        Keyword = varargin{iter};
        Value   = varargin{iter+1};
        if strcmpi(Keyword,'var')
            percp = Value; % 
        elseif strcmpi(Keyword,'init')
            U0 = Value{1};
            S0 = Value{2};
            init_flag = 1;
        else
            warning(['amri_sig_isvd(): unknown keyword ' Keyword]);
        end
    end

    % updating svd
    if init_flag ==1
        A = X - U0*(U0'*X);
        [Q,R] = qr(A,0);

        k = size(U0,2);
        k0 = k;
        r = size(R,1);
        
        [U, S] = svd([S0, U0'*X; zeros(r,k), R], 0);
        U = [U0 Q]*U;

    else
        [U, S] = svd(X,0);
        k = min(size(X,1),size(X,2));
    end

    if percp>0
        v = diag(S).^2;
        vs = cumsum(v);
        vs = vs/vs(end);
        k = find(vs>percp,1,'first');
        if exist('k0','var')
            k = max(k,k0);
        end
    end

    U = U(:,1:k);
    S = S(1:k,1:k);

end
