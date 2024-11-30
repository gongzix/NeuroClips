%% This code is for evaluating the reproducibility of the response to the same movie stimuli
%
% Reference: 
% Wen H, Shi J, Zhang Y, Lu KH, Cao J, and Liu Z. (2017) Neural Encoding
% and Decoding with Deep Learning for Dynamic Natural Vision. Cerebral
% Cortex, In press.

%% history
% v1.0 (original version) --2017/09/13

%% Reproducibility analysis
% Load fMRI responses during watching the movie for the first and second time
% Nv: number of voxels, Nv = 59412.
% Nt: number of volumes for one movie segment, Nt = 240.
% Ns: number of movie segments, Ns = 18.
load('D:/Something/研二下/voxel_select/subj03_training_fmri.mat','fmri'); % from movie_fmri_processing.m
 
% Calculate the voxelwise correlation between the responses to the same movie for the
% first time and the second time.
Nv = 59412;
Nt = 240;
Ns = 18;
Rmat = zeros(Nv, Ns); 
for seg = 1 : Ns
   dt1 = fmri.data1(:,:,seg);
   dt2 = fmri.data2(:,:,seg);
   Rmat(:,seg) = amri_sig_corr(dt1',dt2','mode','auto'); 
end

% Fisher's r-to-z-transformation
Zmat = amri_sig_r2z(Rmat);

save('D:/Something/研二下/voxel_select/subj03_training_Zscore.mat','Zmat')
