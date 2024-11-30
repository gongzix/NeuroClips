%% This code is for estimating the feature maps from fmri response
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
%   - Y: Nt-by-Nu matrix, each column is a stardardized time series of a CNN
%       unit (with log-transformation). Nu is the number of units in a CNN kernel
%   - X: Nt-by-Nv matrix, each column is regressor (stardardized). 
%       Nv is the nubmer of selected voxels.
%   - lambda: a vector, a candidate set of regularization parameters
%   - nfold: a scalar, the number of folds to do cross validation
% outputs:
%   - W: Nc-by-Nv matrix, each column is the optimal encoding weights of a voxel
%   - Rmat: Nv*(#lambda)*nfold array, validation accuracy (correlation)
%   - Errmat: validation mean square error.
%   - Lambda: optimal regularization parameters

%% History
% v1.0 (original version) --2017/09/17

%% Estimate the feature maps using mitivariate linear regression
function [W, Rmat, Errmat, Lambda] = visual_reconstruction(Y,  X, lambda, nfold, opts)
    % training setting
    if isempty(opts)
        opts.InitialMomentum = 0.5;     % momentum for first 'InitialMomentumIter' iterations
        opts.FinalMomentum = 0.9;       % momentum for remaining iterations
        opts.InitialMomentumIter = 20;
        opts.MaxIter = 50;
        opts.DropOutRate1 = 0.3;
        opts.DropOutRate2 = 0;
        opts.StepRatio = 5e-4;
        opts.MinStepRatio = 1e-4;
        opts.SparsityCost = 0; 
        opts.lambda = 0.05; 
        opts.BatchSize = 100;
        opts.disp_flag = 1;
        opts.IterNum = 5; % save model every IterNum interations
        opts.filedir = [];
    end
    
    Nt = size(X, 1);
    dT = floor(Nt/nfold);
    Rmat = zeros(length(lambda), nfold, opts.MaxIter);
    Errmat = zeros(length(lambda), nfold, opts.MaxIter);
    for nf = 1 : nfold
        idx_v = (1:dT)+(nf-1)*dT;
        idx_t = 1: Nt;
        idx_t(idx_v) = [];
        
        % define cost weighting matrix
        M = ones(size(Y)); % change M if necessary.

        dimV = size(X,2);
        dimH = size(Y,2);
        sigma = 0.01;
        if isempty(opts.filedir)
            opts.savefilename = [];
        else
            opts.savefilename = [opts.filedir, '/fold', num2str(nf)];
        end
        para = rand_mlreg(dimV, dimH, 'normal', sigma);
        for k =  1 : length(lambda)
            opts.lambda = lambda(k);
            para = amri_sig_mlreg(X(idx_t,:), Y(idx_t,:), M(idx_t,:), X(idx_v,:), Y(idx_v,:), M(idx_v,:), para, opts);
            Rmat(k,nf,:) = para.corrmat(:);
            Errmat(k,nf,:) = para.errmat(:);
        end
    end

    [~, lmbidx] = max(squeeze(mean(Rmat(:,:,end),2)));
    Lambda = lambda(lmbidx);
    
    % train optimal model
    dimV = size(X,2);
    dimH = size(Y,2);
    sigma = 0.01;
    if isempty(opts.filedir)
        opts.savefilename = [];
    else
        opts.savefilename = [opts.filedir, '/optimal'];
    end
    para = rand_mlreg(dimV, dimH, 'normal', sigma);
    opts.lambda = Lambda;
    para = amri_sig_mlreg(X, Y, M, [], [], [], para, opts);
    
    W = para.W;
    
end
