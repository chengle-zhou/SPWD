% Superpixel-to-Pixel Weighted Density for Hyperspectral
% Image Noisy Label Detection (TGRS-20190903, code by Chengle Zhou)

close all; clear all; clc
addpath ('.\common')
% rmpath ('.\libsvm-3.20')
% addpath ('.\libsvm-3.20')
%% load original image
load(['.\datasets\KSC.mat']);
load(['.\datasets\KSC_gt2.mat']);

% - Preprocessing -
% The size of image and label
img = KSC;
img2 = KSC;
[rows, cols, bands] = size(img);
no_class = max(GroundT(:,2));
label_distri = KSC_gt;
% The indexes of background data
[bagrX, bagrY] = find(KSC_gt == 0);
% Matrix transformation for original hyperspectral data
img_2d = reshape(img,[rows*cols, bands]);


% Constructing training and testing data
per_class_ture = 24;
GroundT = GroundT';
Value = ones(no_class,1).*per_class_ture; % the number of samples (24) per class
indexes = train_random_select(GroundT(2,:),Value); 
train_SL = GroundT(:,indexes);
test_SL = GroundT;
test_SL(:,indexes) = [];
GroudTest = test_SL(2,:)';
train_samples = img_2d(train_SL(1,:),:);
train_labels = train_SL(2,:);

%% Experiment 1 (original SVM classification result)
[ OA_1,AA_1,kappa_1,CA_1 ] = My_SVM_Classifier(img2,train_samples,train_labels',test_SL,GroudTest);

%% Load the training set that adds the noise labels
load(['.\datasets\Noise_samples_4.mat']);
% load(['.\datasets\Noise_samples_8.mat']);
% load(['.\datasets\Noise_samples_12.mat']);
%% Experiment 2 (SVM classification result with noisy labels)
zt = 0;
for nn = 1:5
    zt = zt+1;
train_noisy_samples = Noise_samples_information{zt};
training_label_1 = train_noisy_samples(:,1);
training_data_1 = img_2d(train_noisy_samples(:,3),:);
[OA_2,AA_2,kappa_2,CA_2] = My_SVM_Classifier(img2,training_data_1,training_label_1,test_SL,GroudTest);
OA_R(nn) = OA_2;
AA_R(nn) = AA_2;
kappa_R(nn) = kappa_2;
end
OA_2 = mean(OA_R);
AA_2 = mean(AA_R);
kappa_2= mean(kappa_R);

%% Experiment 3 (using SPWD's SVM classification results)
zt = 0;
for nn = 1:5
    zt = zt+1;
train_noisy_samples = Noise_samples_information{zt};
train_noisy_index = train_noisy_samples(:,3);
train_noisy = train_noisy_samples;
train_noisy_sample = [train_noisy_index';train_noisy_samples(:,1)'];

% Handling test set again
[~, Ia, Ib] = intersect(train_noisy_index,GroundT(1,:));
test_SL = GroundT;
test_SL(:,Ib) = [];
test_samples = img_2d(test_SL(1,:),:);
GroudTest2 = test_SL(2,:);
% PCA reduce dimension
img_pca = compute_mapping(img_2d,'PCA',3);     
superpixel_data = reshape(img_pca,[rows, cols, 3]);
superpixel_data = mat2gray(superpixel_data);
superpixel_data = im2uint8(superpixel_data);

% Superpixels 
number_superpixels = 9000;  % number of superpixel blocks
lambda_prime = 0.8;  sigma = 10;  conn8 = 1;
SuperLabels = mex_ers(double(superpixel_data),number_superpixels,lambda_prime,sigma,conn8);

% SPWD noisy label detection
index_map = reshape(1:size(img_2d,1),[rows, cols]);

% Superpixel Extraction
[train_se_sample,train_se_data] = SE(GroundT,train_noisy,index_map,label_distri,img_2d,SuperLabels);

% SPWD detection function
K = 6;          % Nearest neighbor
Ct = 0.11;      % Half peak of Gaussian weight function
lambda = 0.1;   % Threshold parameter of decision function
metric = 'SAM';

num = 4;        % Number of noisy labels added per class
% num = 8;
% num = 12;
[train_correct,detec_result,Rho] = SPWD_le(img_2d,train_se_sample,train_noisy_sample,lambda,K,Ct,num,metric);

training_index = train_correct(1,:);
training_label = train_correct(2,:)';
training_data = img_2d(training_index,:);
[OA_3,AA_3,kappa_3,CA_3] = My_SVM_Classifier(img2,training_data,training_label,test_SL,GroudTest2);

OA_Z(nn) = OA_3;
AA_Z(nn) = AA_3;
kappa_Z(nn) = kappa_3;
end
OA_3 = mean(OA_Z);
AA_3 = mean(AA_Z);
kappa_3= mean(kappa_Z);


fprintf('-> SVM trained with clean training set (24(T) per class) %f %f %f\n', OA_1, AA_1, kappa_1);
fprintf('-> SVM trained with noisy training set (24(T) + num per class) %f %f %f\n', OA_2, AA_2, kappa_2);
fprintf('-> SVM trained with Purification training set (sample size varies per class) %f %f %f\n', OA_3, AA_3, kappa_3);

