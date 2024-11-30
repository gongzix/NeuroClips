%% This code is for predicting the categories from fmri responses
% 
% Data: The video-fMRI dataset are available online: 
% https://engineering.purdue.edu/libi/lab/Resource.html.
% 
% Environment requirement:  
% This code was developed under Red Hat Enterprise Linux environment.
%
% Reference: 
% Wen H, Shi J, Zhang Y, Lu KH, Cao J, and Liu Z. (2017) Neural Encoding
% and Decoding with Deep Learning for Dynamic Natural Vision. Cerebral
% Cortex, In press.
%
% Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). 
% Imagenet classification with deep convolutional neural networks.
% In Advances in neural information processing systems (pp. 1097-1105).
%

% inputs:
%   - Y: Nt-by-Nc matrix, each row is the semantic represenation in the
%       dimension-reduced space. Nt is the number of volumes. 
%       See AlexNet_feature_processing_encoding.m
%   - X: Nt-by-Nv matrix, each row is the cortical representation in the
%       visual cortex.
%   - lambda: a vector, a candidate set of regularization parameters
%   - nfold: a scalar, the number of folds to do cross validation
% outputs:
%   - W: Nc-by-Nv matrix, 
%   - Rmat: Nv*(#lambda)*nfold array, validation accuracy (correlation)
%   - Lambda: optimal regularization parameters
%   - q: optimal number of principal components to keep.

%% History
% v1.0 (original version) --2017/09/17

%% fMRI-based Categorization 

function [Wo, W, Lambda, q] = category_prediction(X, Y, lambda, nfold)
    
    % Load principal components
    dataroot = '/path/to/alexnet_feature_maps/';
    load([dataroot,'AlexNet_feature_maps_pca_layer7.mat'], 'B');
    Yo = Y*B'*sqrt(size(B,1)); % transform Y back to the original semantic space
    
    % validation: nfold cross validation    
    Nt = size(Y,1); % number of total time points
    Nc = size(Y,2); % number of components
    dT = Nt/nfold; % number of time points in a fold
    T = Nt - dT; % number of time points for training

    Rmat = zeros(Nc,length(lambda),nfold,'single');
    
    disp('Validating parameters ..');
    for nf = 1 : nfold
        idx_t = 1:Nt;
        idx_v = (nf-1)*dT+1:nf*dT;
        idx_t(idx_v) = [];
        
        Y_t = Y(idx_t,:);
        Yo_v = Yo(idx_v,:); % Ground truth in the original space
        X_v = X(idx_v,:);
        X_t = X(idx_t,:);
        XXT = X_t*X_t';
        
        % validate the regularization parameter
        for k = 1 : length(lambda)
            disp(['fold: ', num2str(nf),' lambda: ',num2str(k)]);
            lmb = lambda(k);
            W = X_t'*((XXT + lmb*T*eye(T))\Y_t);
            
            Y_vp = X_v * W; % predicted
            % validate the number of componets
            for q =  1 : Nc
                dispstat(['fold: ', num2str(nf),' lambda: ',num2str(k), '; componets: ', num2str(q)]);
                Yo_vp = Y_vp(:,1:q)*B(:,1:q)'*sqrt(size(B,1)); 
                R = amri_sig_corr(Yo_v', Yo_vp', 'mode', 'auto');
                Rmat(q,k,nf) = mean(R);
            end
        end
    end

    R = mean(Rmat,3);
    [~,idx] = max(R(:));
    [q,lmbidx] = ind2sub(size(R),idx);
    Lambda = lambda(lmbidx);
    
    % Train optimal model
    W = X'*((X*X' + lmb*Nt*eye(Nt))\Y); % decoder in dimension-reduced space            
    Wo = W*B(:,1:q)'*sqrt(size(B,1));  % decoder in the original semantic space
    
end
