%% This code is for estimating the feature maps from fmri response
% 
% Data: The video-fMRI dataset are available online: 
% https://engineering.purdue.edu/libi/lab/Resource.html.
% 
% Environment requirement:  
% This code was developed under Red Hat Enterprise Linux environment.
%
% inputs:
%   - X: Nt-by-dimV matrix, each column is a regressor.
%   - Y: Nt-by-dimH matrix, each column is the response time series
%   - M: Cost weighting matrix (optional). Has the same size as Y. 
%   - x_v: validation data, each column is a regressor.
%   - y_x: validation data, each column is the response time series
%   - m_v: validation data. Cost weighting matrix.
%   - para: contains intial model.
%   - opts: training settings.

% outputs:
%   - para: contains trained model.


%% History
% v1.0 (original version) --2017/09/13

%% Linear regression model

function para = amri_sig_mlreg(X, Y, M, x_v, y_v, m_v, para, opts )
    % defaults:
    InitialMomentum = 0.5;     % momentum for first #InitialMomentumIter iterations
    FinalMomentum = 0.9;       % momentum for remaining iterations
    lambda = 0.05;       % L1 regularization parameter
    InitialMomentumIter = 20;
    MaxIter = 50;
    DropOutRate1 = 0.3; % input dropout rate
    DropOutRate2 = 0; % output dropout rate
    StepRatio = 5e-4;
    MinStepRatio = 1e-4;
    BatchSize = 0;
    disp_flag = 0;
    IterNum = 10;
    savefilename = [];
    SparsityCost = 0;

    % read parameters
    if(exist('opts','var'))
        if( isfield(opts,'MaxIter') )
            MaxIter = opts.MaxIter;
        end
        if( isfield(opts,'InitialMomentum') )
            InitialMomentum = opts.InitialMomentum;
        end
        if( isfield(opts,'InitialMomentumIter') )
            InitialMomentumIter = opts.InitialMomentumIter;
        end
        if( isfield(opts,'FinalMomentum') )
            FinalMomentum = opts.FinalMomentum;
        end
        if( isfield(opts,'lambda') )
            lambda = opts.lambda;
        end
        if( isfield(opts,'SparsityCost') )
            SparsityCost = opts.SparsityCost;
        end
        if( isfield(opts,'DropOutRate1') )
            DropOutRate1 = opts.DropOutRate1;
        end
        if( isfield(opts,'DropOutRate2') )
            DropOutRate2 = opts.DropOutRate2;
        end
        if( isfield(opts,'StepRatio') )
            StepRatio = opts.StepRatio;
        end
        if( isfield(opts,'MinStepRatio') )
            MinStepRatio = opts.MinStepRatio;
        end
        if( isfield(opts,'BatchSize') )
            BatchSize = opts.BatchSize;
        end
        if( isfield(opts,'disp_flag') )
            disp_flag = opts.disp_flag;
        end
        if( isfield(opts,'IterNum') )
            IterNum = opts.IterNum;
        end
        if( isfield(opts,'savefilename') )
            savefilename = opts.savefilename;
        end

    end

    % initialize some parameters
    [num, dimV] = size(X);
    dimH = size(Y, 2);

    if( BatchSize <= 0 )
        BatchSize = num;
    end

    deltaW = zeros(dimV, dimH);

    %% start training
    unitMomentum = ((FinalMomentum - InitialMomentum)/(num/BatchSize*InitialMomentumIter));
    momentum = InitialMomentum;
    errmat = zeros(1, floor(num/BatchSize)*MaxIter);
    corrmat = zeros(1, floor(num/BatchSize)*MaxIter);
    k = 1;
    for iter=1:MaxIter
        % train one interation
        ind = randperm(num);
        N = floor(num/BatchSize);
        for batch=1:BatchSize:N*BatchSize
            fprintf(1,'epoch %d batch %d\r',iter,ceil(batch/BatchSize)); 

            % set momentum
            if (iter <= InitialMomentumIter)
                momentum = momentum + unitMomentum;
            end
            % select one batch data
            bind = ind(batch : min([batch+BatchSize-1,num]));
            x = double(X(bind,:));
            y = double(Y(bind,:));
            m = double(M(bind,:));
            % perform Dropout on inputs
            if(DropOutRate1 > 0)
                cMat1 = (rand(BatchSize,dimV)>DropOutRate1);
                x = x.*cMat1;
            end
            % perform Dropout on outpouts
            if(DropOutRate2 > 0)
                cMat2 = (rand(BatchSize,dimH)>DropOutRate2);
                y = y.*cMat2;
            end

            % Compute the weights update
            z = x*para.W;
            dW = (2/BatchSize)*x'*((y-z).*m);
            deltaW = momentum * deltaW + (1-momentum)*StepRatio*(dW-(SparsityCost/BatchSize)*x'*(exp(z))...
                -(lambda)*((para.W>0)*2-1));

            % Update the network weights
            para.W = para.W + deltaW;

            if ~isempty(x_v)
                z_v = x_v*para.W;
                err = power((y_v-z_v).*m_v, 2);
                rmse = sqrt(sum(err(:)) / numel(err));
                errmat(k) = rmse; 
                r = amri_sig_corr(y_v(:),z_v(:));
                corrmat(k) = r; 
                k = k + 1;
            end
            if disp_flag && (~isempty(x_v)) && rem(ceil(batch/BatchSize),3) == 1
                figure(101);
                subplot(1,2,1); hist(para.W(:),100);
                subplot(1,2,2); hist(z_v(:),100);
                title('Histogram of W and estimated FMap'); 
                drawnow;
            end
            if disp_flag && (~isempty(x_v))         
                figure(102); plot((1:k-1)/N, errmat(1:k-1));
                title('Estimation error'); drawnow;
            end
            if disp_flag && (~isempty(x_v))         
                figure(103); plot((1:k-1)/N, corrmat(1:k-1));
                title('Estimation accuracy (correlation)'); drawnow;
            end
        end

        if (rem(iter,IterNum)==0) && (~isempty(savefilename))
            para.opts = opts;
            para.errmat = errmat(floor(num/BatchSize):floor(num/BatchSize):end);
            para.corrmat = corrmat(floor(num/BatchSize):floor(num/BatchSize):end);
            save([savefilename,'_iter',num2str(ceil(iter)), '.mat'], 'para');
        end

        if StepRatio > MinStepRatio
            StepRatio = StepRatio*0.96;
        end
    end
    
    para.opts = opts; % training setting
    para.errmat = errmat(floor(num/BatchSize):floor(num/BatchSize):end); % validation root of meam square error
    para.corrmat = corrmat(floor(num/BatchSize):floor(num/BatchSize):end); % validation correlation
    
end
