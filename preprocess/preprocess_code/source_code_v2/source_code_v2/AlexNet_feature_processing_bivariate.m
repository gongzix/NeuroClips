%% This code is for processing the CNN features extracted from videos
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

%% History
% v1.0 (original version) --2017/09/17


%% Process the AlexNet features for bivariate analysis to relate CNN units to brain voxels
% CNN layer labels
layername = {'/conv1';'/conv2';'/conv3';'/conv4';'/conv5';'/fc6';'/fc7';'/fc8'};

% The sampling rate should be equal to the sampling rate of CNN feature
% maps. If the CNN extracts the feature maps from movie frames with 30
% frames/second, then srate = 30. It's better to set srate as even number
% for easy downsampling to match the sampling rate of fmri (2Hz).
srate = 30; 

% Here is an example of using predefined hemodynamic response function
% (HRF) with positive peak at 4s.
p  = [5, 16, 1, 1, 6, 0, 32];
hrf = spm_hrf(1/srate,p);
hrf = hrf(:);
% figure; plot(0:1/srate:p(7),hrf);

dataroot = '/path/to/alexnet_feature_maps/';

% Training movies
for seg = 1 : 18
    secpath = [dataroot,'AlexNet_feature_maps_seg', num2str(seg),'.h5'];
    
    for lay = 1 : length(layername)
        disp(['Seg: ', num2str(seg), '; Layer: ',layername{lay}]);
        % info = h5info(secpath);
        lay_feat = h5read(secpath,[layername{lay},'/data']);
        dim = size(lay_feat);
        Nu = prod(dim(1:end-1)); % number of units
        Nf = dim(end); % number of frames
        lay_feat = reshape(lay_feat,Nu,Nf);% Nu*Nf
        if lay < length(layername)
            lay_feat = log10(lay_feat + 0.01); % log-transformation except the last layer
        end
        ts = conv2(hrf,lay_feat'); % convolude with hrf
        ts = ts(4*srate+1:4*srate+Nf,:);
        ts = ts(srate+1:2*srate:end,:)'; % downsampling
        ts = reshape(ts,[dim(1:end-1),240]);
        
        % check time series
        % figure;plot(squeeze(ts(25,25,56,:)));

        h5create([dataroot,'AlexNet_feature_maps_processed_seg', num2str(seg),'.h5'],[layername{lay},'/data'],...
            [size(ts)],'Datatype','single');
        h5write([dataroot,'AlexNet_feature_maps_processed_seg', num2str(seg),'.h5'], [layername{lay},'/data'], ts);
    end
end

% Testing movies
for test = 1 : 5
    secpath = [dataroot,'AlexNet_feature_maps_test', num2str(test),'.h5'];
    for lay = 1 : length(layername)
        disp(['Test: ', num2str(test), '; Layer: ',layername{lay}]);
        lay_feat = h5read(secpath,[layername{lay},'/data']);
        dim = size(lay_feat);
        Nu = prod(dim(1:end-1)); % number of units
        Nf = dim(end); % number of frames
        lay_feat = reshape(lay_feat,Nu,Nf);% Nu*Nf
        if lay < length(layername)
            lay_feat = log10(lay_feat + 0.01); % log-transformation
        end
        ts = conv2(hrf,lay_feat'); % convolude with hrf
        ts = ts(4*srate+1:4*srate+Nf,:);
        ts = ts(srate+1:2*srate:end,:)'; % downsampling
        ts = reshape(ts,[dim(1:end-1),240]);

        h5create([dataroot,'AlexNet_feature_maps_processed_test', num2str(test),'.h5'],[layername{lay},'/data'],...
            [size(ts)],'Datatype','single');
        h5write([dataroot,'AlexNet_feature_maps_processed_test', num2str(test),'.h5'], [layername{lay},'/data'], ts);
    end
end
