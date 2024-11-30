%% This code is for retinotopic analysis by using CNN
% 
% Data: The raw and preprocessed fMRI data in NIFTI and CIFTI formats are
% available online: https://engineering.purdue.edu/libi/lab/Resource.html.
% This code focuses on the processing of the fMRI data on the cortical
% surface template (CIFTI format).
%
% Reference: 
% Wen H, Shi J, Zhang Y, Lu KH, Cao J, and Liu Z. (2017) Neural Encoding
% and Decoding with Deep Learning for Dynamic Natural Vision. Cerebral
% Cortex, In press.

%% History
% v1.0 (original version) --2017/09/14

%% Concatenate CNN activation time series in the 1st layer across movie segments
% CNN layer labels
layername = {'/conv1';'/conv2';'/conv3';'/conv4';'/conv5';'/fc6';'/fc7';'/fc8'};
dataroot = '/path/to/alexnet_feature_maps/';
lay = 1;

% Training movies
for seg = 1 : 18
    disp(['Seg: ', num2str(seg)]);
    secpath = [dataroot,'AlexNet_feature_maps_processed_seg', num2str(seg),'.h5'];      
    % info = h5info(secpath);
    lay_feat = h5read(secpath,[layername{lay},'/data']);
    dim = size(lay_feat);
    Nf = dim(end); % number of frames
    if seg == 1
       lay_feat_concatenated = zeros([dim(1:end-1),Nf*18],'single'); 
    end
    lay_feat_concatenated(:,:,:,(seg-1)*Nf+1:seg*Nf) = lay_feat;
end

% stadnardize the time series of each unit
dim = size(lay_feat_concatenated);
lay_feat_concatenated = reshape(lay_feat_concatenated, prod(dim(1:end-1)), dim(end));
lay_feat_concatenated_mean = mean(lay_feat_concatenated,2);
lay_feat_concatenated_std = std(lay_feat_concatenated,[],2);
lay_feat_concatenated = bsxfun(@minus, lay_feat_concatenated, lay_feat_concatenated_mean);
lay_feat_concatenated = bsxfun(@rdivide, lay_feat_concatenated, lay_feat_concatenated_std);
lay_feat_concatenated(isnan(lay_feat_concatenated)) = 0;
lay_feat_concatenated = reshape(lay_feat_concatenated, dim);

save([dataroot,'AlexNet_feature_maps_processed_layer1_concatenated.mat'], ...
    'lay_feat_concatenated','lay_feat_concatenated_mean','lay_feat_concatenated_std','-v7.3');

% check time series
% figure;plot(squeeze(lay_feat_concatenated(25,25,56,:)));

% CNN activation time series in the 1st layer for testing movie
% Testing movies
% for test = 1 : 5
%     disp(['Test: ', num2str(test)]);
%     secpath = [dataroot,'AlexNet_feature_maps_processed_test', num2str(test),'.h5'];      
%     % info = h5info(secpath);
%     lay_feat = h5read(secpath,[layername{lay},'/data']);
%     dim = size(lay_feat);
%     lay_feat = reshape(lay_feat, prod(dim(1:end-1)), dim(end));
%     lay_feat = bsxfun(@minus, lay_feat, lay_feat_concatenated_mean);
%     lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_concatenated_std);
%     lay_feat(isnan(lay_feat)) = 0;
%     lay_feat = reshape(lay_feat, dim);
%     save([dataroot,'AlexNet_feature_maps_processed_layer1_test',num2str(test),'.mat'], ...
%     'lay_feat','lay_feat_concatenated_mean','lay_feat_concatenated_std','-v7.3');
% end

%% Load fmri responses
fmripath = '/path/to/fmri/';
load([fmripath,'training_fmri.mat'],'fmri'); % from movie_fmri_processing.m
fmri_avg = (fmri.data1+fmri.data2) / 2; % average across repeats
fmri_avg = reshape(fmri_avg, size(fmri_avg,1), size(fmri_avg,2)*size(fmri_avg,3));


%% Correlate fmri response to the CNN activation time series
% Correlating all the voxels to all the CNN units is time-comsuming.
% Here is the analysis given some example voxels in the visual cortex.
load([dataroot,'AlexNet_feature_maps_processed_layer1_concatenated.mat'], 'lay_feat_concatenated');
dim = size(lay_feat_concatenated);
Nu = prod(dim(1:end-1));% number of units
Nf = dim(end); % number of time points
lay_feat_concatenated = reshape(lay_feat_concatenated, Nu, Nf)';

voxels = [22428, 22797, 51866, 10142, 40075, 22574, 53126, 52478];
Rmat = zeros([dim(1:end-1),length(voxels)]);
k1 = 1;
while k1 <= length(voxels)
   disp(['Voxel: ', num2str(k1)]);
   k2 = min(length(voxels), k1+100);
   R = amri_sig_corr(lay_feat_concatenated, fmri_avg(voxels(k1:k2),:)');
   Rmat(:,:,:,k1:k2) = reshape(R,dim(1),dim(2),dim(3), k2-k1+1);
   k1 = k2+1;
end
save([fmripath, 'cross_corr_fmri_cnn_layer1.mat'], 'Rmat','-v7.3');

% Take the maximum across kernels
R = squeeze(max(Rmat,[],3));

% Show the receptive field of voxels
figure(100);colormap jet
for v = 1: length(voxels)
   map = R(:,:, v)'; 
   imagesc(map);
   axis off image
   title(v); colorbar
   pause;
end


%% Calculate polar angle and eccentricity
% Take the maxima across kernels
R = squeeze(max(Rmat,[],3)); % 55*55*Nv
CR = reshape(R, size(R,1)*size(R,2),size(R,3));

% find the first maximum 20 CNN units
Nu = 20;
Nv = size(CR,2); % number of voxels
Idx = zeros(Nv*2,Nu);
for v = 1 : Nv
    c = CR(:,v)';
    [B, I] = sort(c,'descend');
    idx1 = I(B>0);
    num = min(length(idx1),Nu);
    xy = idx2xy(idx1(1:num), 55);
    Idx((v-1)*2+1:v*2,1:num) = xy;
end

% calculate angle and encentricity
retin = retinotopic(Idx, 55);
retin(:,2) = abs(retin(:,2)); % sin(angle)
% first column is the eccentricity, the second column is the polar angle.

% save retinotopic_mapping.mat retin
