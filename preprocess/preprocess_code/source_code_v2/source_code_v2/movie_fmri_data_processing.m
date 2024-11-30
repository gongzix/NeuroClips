%% This code is for processing the BOLD fMRI response to natural movies
% 
% Data: The raw and preprocessed fMRI data in NIFTI and CIFTI formats are
% available online: https://engineering.purdue.edu/libi/lab/Resource.html.
% This code focuses on the processing of the fMRI data on the cortical
% surface template (CIFTI format).
% 
% Environment requirement: Install the workbench toolbox published by Human
% Connectome Project (HCP). It is public available on 
% https://www.humanconnectome.org/software/connectome-workbench. 
% This code was developed under Red Hat Enterprise Linux environment.
%
% Reference: 
% Wen H, Shi J, Zhang Y, Lu KH, Cao J, and Liu Z. (2017) Neural Encoding
% and Decoding with Deep Learning for Dynamic Natural Vision. Cerebral
% Cortex, In press.

%% History
%v1.0 (original version) --2017/09/13


% %% Process fMRI data (CIFTI) for an example segment
% fmripath = '/path/to/fmri/'; % path to the cifti files
% filename = 'seg1/cifti/seg1_1_Atlas.dtseries.nii';
% cii = ciftiopen([fmripath, filename],'wb_command');
% 
% % For training data (with prefixed 'seg') and the first testing data
% % (i.e. test1), we disregarded the first volume and the the last 4 volumes, 
% % reducing the number of volumes from 245 to 240 volumes (i.e. 8mins, TR=2s). 
% % For other testing data, we disregarded the first 2 volumes and the last 4
% % volumes, reducing volume# from 246 to 240.
% if size(cii.cdata,2) == 245
%     st = 2;
% else
%     st = 3;
% end
% 
% % Mask out the vertices in subcortical areas 
% Nv = 59412;% number of vertices on the cortical surface
% lbl = zeros(size(cii.cdata,1),1);
% lbl(1:Nv) = 1;
% lbl = (lbl == 1);
% data = double(cii.cdata(lbl,st:end-4));
% 
% % Remove the 4th order polynomial trend for each voxel
% for i = 1 : size(data,1)
%     data(i,:) = amri_sig_detrend(data(i,:),4);
% end
% 
% % Standardization: remove the mean and divide the standard deviation
% data = bsxfun(@minus, data, mean(data,2));
% data = bsxfun(@rdivide, data, std(data,[],2));
% 
% % Check the time series
% figure;
% for i =  1 : 100 : 10000
%     plot(data(i,:));
%     pause;
% end

%% Put all training segments together
Nv = 59412; % number of vertices on the cortical surface
Nt = 240;  % number of volumes
Ns = 18; % number of training movie segments

fmripath = 'C:/Users/44545/fMRI/10_4231_R7X63K3M/10_4231_R7X63K3M/bundle/video_fmri_dataset/subject1/fmri/'; % path to the cifti files
fmri.data1 = zeros(Nv,Nt,Ns,'single');% use single to save memory  
fmri.data2 = zeros(Nv,Nt,Ns,'single');
Rmat = zeros(Nv,Ns); 

for seg = 1 : Ns
    disp(['segment: ',num2str(seg)]);
    filename1 = ['seg',num2str(seg),'/cifti/seg', num2str(seg),'_1_Atlas.dtseries.nii'];
    filename2 = ['seg',num2str(seg),'/cifti/seg', num2str(seg),'_2_Atlas.dtseries.nii'];
    
    cii1 = ciftiopen([fmripath, filename1],'wb_command');
    cii2 = ciftiopen([fmripath, filename2],'wb_command');

    % For training data (with prefixed 'seg') and the first testing data
    % (i.e. test1), we disregarded the first volume and the the last 4 volumes, 
    % reducing the number of volumes from 245 to 240 volumes (i.e. 8mins, TR=2s). 
    % For other testing data, we disregarded the first 2 volumes and the last 4
    % volumes, reducing volume# from 246 to 240.
    if size(cii1.cdata,2) == 245
        st = 2;
    else
        st = 3;
    end

    % Mask out the vertices in subcortical areas 
    lbl = zeros(size(cii1.cdata,1),1);
    lbl(1:Nv) = 1;
    lbl = (lbl == 1);
    data1 = double(cii1.cdata(lbl,st:end-4));
    data2 = double(cii2.cdata(lbl,st:end-4));

    % Remove the 4th order polynomial trend for each voxel
    for i = 1 : size(data1,1)
        data1(i,:) = amri_sig_detrend(data1(i,:),4);
        data2(i,:) = amri_sig_detrend(data2(i,:),4);
    end

    % Standardization: remove the mean and divide the standard deviation
    data1 = bsxfun(@minus, data1, mean(data1,2));
    data1 = bsxfun(@rdivide, data1, std(data1,[],2));
    
    data2 = bsxfun(@minus, data2, mean(data2,2));
    data2 = bsxfun(@rdivide, data2, std(data2,[],2));
    
    fmri.data1(:,:,seg) = data1;
    fmri.data2(:,:,seg) = data2;
    
    % calculate reproducibility to check the data
    R = amri_sig_corr(data1',data2','mode','auto');
    Rmat(:,seg) = R(:);
end

% save fmri
save([fmripath,'training_fmri.mat'], 'fmri', 'Rmat');


%% Put all testing segments together
Nv = 59412; % number of vertices on the cortical surface
Nt = 240;  % number of volumes
fmripath = 'C:/Users/44545/fMRI/10_4231_R7X63K3M/10_4231_R7X63K3M/bundle/video_fmri_dataset/subject1/fmri/'; % path to the cifti files
fmritest.test1 = zeros(Nv,Nt,10,'single');
fmritest.test2 = zeros(Nv,Nt,10,'single');
fmritest.test3 = zeros(Nv,Nt,10,'single');
fmritest.test4 = zeros(Nv,Nt,10,'single');
fmritest.test5 = zeros(Nv,Nt,10,'single');

for seg = 1 : 5
    for rep = 1 : 10
        disp(['segment: ',num2str(seg),'; repeat: ', num2str(rep)]);
        filename = ['test',num2str(seg),'/cifti/test', num2str(seg),'_',num2str(rep),'_Atlas.dtseries.nii'];
        cii = ciftiopen([fmripath, filename],'wb_command');

        % For training data (with prefixed 'seg') and the first testing data
        % (i.e. test1), we disregarded the first volume and the the last 4 volumes, 
        % reducing the number of volumes from 245 to 240 volumes (i.e. 8mins, TR=2s). 
        % For other testing data, we disregarded the first 2 volumes and the last 4
        % volumes, reducing volume# from 246 to 240.
        if size(cii.cdata,2) == 245
            st = 2;
        else
            st = 3;
        end

        % Mask out the vertices in subcortical areas 
        lbl = zeros(size(cii.cdata,1),1);
        lbl(1:Nv) = 1;
        lbl = (lbl == 1);
        data = double(cii.cdata(lbl,st:end-4));

        % Remove the 4th order polynomial trend for each voxel
        for i = 1 : size(data,1)
            data(i,:) = amri_sig_detrend(data(i,:),4);
        end

        % Standardization: remove the mean and divide the standard deviation
        data = bsxfun(@minus, data, mean(data,2));
        data = bsxfun(@rdivide, data, std(data,[],2));
        
        if seg == 1
            fmritest.test1(:,:,rep) = data;
        elseif seg == 2
            fmritest.test2(:,:,rep) = data;    
        elseif seg == 3
            fmritest.test3(:,:,rep) = data;        
        elseif seg == 4
            fmritest.test4(:,:,rep) = data;
        elseif seg == 5
            fmritest.test5(:,:,rep) = data;
        end             
    end
end

% save fmritest
save([fmripath,'testing_fmri.mat'], 'fmritest', '-v7.3');








