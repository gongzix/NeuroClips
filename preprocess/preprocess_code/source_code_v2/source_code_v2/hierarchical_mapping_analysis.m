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

% Training movies
for lay = 1 : length(layername)
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
        if lay <= 5
            lay_feat_concatenated(:,:,:,(seg-1)*Nf+1:seg*Nf) = lay_feat;
        else
            lay_feat_concatenated(:,(seg-1)*Nf+1:seg*Nf) = lay_feat;
        end
    end

    % check time series
    % figure;plot(squeeze(lay_feat_concatenated(25,25,56,:)));

    save([dataroot,'AlexNet_feature_maps_processed_layer',num2str(lay),'_concatenated.mat'],...
        'lay_feat_concatenated','-v7.3');
end


%% Load fmri responses
fmripath = '/path/to/subject1/fmri/';
load([fmripath,'training_fmri.mat'],'fmri'); % from movie_fmri_processing.m
fmri_avg = (fmri.data1+fmri.data2) / 2; % average across repeats
fmri_avg = reshape(fmri_avg, size(fmri_avg,1), size(fmri_avg,2)*size(fmri_avg,3));


%% Map hierarchical CNN features to brain
% Correlating all the voxels to all the CNN units is time-comsuming.
% Here is the analysis given some example voxels in the visual cortex.
% select voxels
voxels = [21892, 21357, 21885, 51456, 22778, 53919 43797, 54301];

for lay = 1 : length(layername)
    load([dataroot,'AlexNet_feature_maps_processed_layer',num2str(lay),'_concatenated.mat'],'lay_feat_concatenated');
    dim = size(lay_feat_concatenated);
    Nu = prod(dim(1:end-1));% number of units
    Nf = dim(end); % number of time points
    lay_feat_concatenated = reshape(lay_feat_concatenated, Nu, Nf)';

    Rmat = zeros([Nu,length(voxels)]);
    k1 = 1;
    while k1 <= length(voxels)
       disp(['Layer: ',num2str(lay),'; Voxel: ', num2str(k1)]);
       k2 = min(length(voxels), k1+100);
       R = amri_sig_corr(lay_feat_concatenated, fmri_avg(voxels(k1:k2),:)');
       Rmat(:,k1:k2) = R;
       k1 = k2+1;
    end
    save([fmripath, 'cross_corr_fmri_cnn_layer',num2str(lay),'.mat'], 'Rmat', '-v7.3');
end

lay_corr = zeros(length(layername),length(voxels));
for lay = 1 : length(layername)
    disp(['Layer: ',num2str(lay)]);
    load([fmripath, 'cross_corr_fmri_cnn_layer',num2str(lay),'.mat'],'Rmat');
    lay_corr(lay,:) = max(Rmat,[],1);
end

save([fmripath, 'cross_corr_fmri_cnn.mat'], 'lay_corr');

% Assign layer index to each voxel
[~,layidx] = max(lay_corr,[],1);


% Display correlation profile for example voxels
figure(100);
for v = 1 : length(voxels)
    plot(1:length(layername),lay_corr(:,v)','--o');
    title(v);
    pause;
end

