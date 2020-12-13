function [train_correct,detec_result,Rho] = SWD(img_2d,train_joint_sample,train_noisy_sample,lambda,K,half_peak,num,metric)
% Bing Tu, Chengle Zhou, Danbing He, Siyuan Huang, Antonio Palaza. Hyperspectral
% Classification with Noisy Label Detection via Superpixel-to-Pixel Weighting Distance. IEEE
% Transaction on Geoscience and Remote Sensing. 2020, 58(6): 4116-4131. 

% parametre setting about DP clustering
para.method = 'gaussian';
para.percent = 20.0;

% parameter setting about gaussian weight
height = 1; offset = 0; %half_peak = 1;

sample_index = train_noisy_sample(1,:);
dist_matrix = [];
train_correct = [];
detec_result = [];
Rho = [];
for c = 1 : max(train_noisy_sample(2,:))
    
    per_sample_index = sample_index(find(train_noisy_sample(2,:) == c));
    per_super_block = train_joint_sample(find(train_noisy_sample(2,:) == c));
    
    for j = 1 : length(per_sample_index)
        sample = img_2d(per_sample_index(j),:);
        for z = 1 : length(per_super_block)
            sub_sample = img_2d(per_super_block{z}(1,:),:);
            % 求距离
            %         sample_repmat = (repmat(sample,size(sub_sample,2),1))';
            %         dist = sqrt(sum((sample_repmat - sub_sample).^2));
            dist = pdist_le(sample,sub_sample,metric);
            % KNN - gaussian weight
            k = K;
            len = size(dist,2);
            if len < k
                k = len;
            else
                k = K;
            end
            dist_sort = sort(dist,'ascend');
%             dist_sort = dist_sort(2:end);
            dist_sort_k = dist_sort(1:k);
            dist_sort_k_nor = dist_sort_k./repmat(sqrt(sum(dist_sort_k.*dist_sort_k)),[size(dist_sort_k,1) 1]); %
            coefficient = (dist_sort_k_nor - offset).^2 / (2 * half_peak.^2);
            weight = height * exp(-coefficient);
            %         weight = weight./repmat(sqrt(sum(weight.*weight)),[size(weight,1) 1]); %
            dist_gaussian = sum(weight.*dist_sort_k) / sum(weight);
            dist_matrix(j,z) = dist_gaussian;
        end
    end
    [rho] = cluster_dp_auto(dist_matrix, para);
    Rho = [Rho;rho];
    rho_mean = mean(rho);
    rho_threshold = lambda * rho_mean;
    noisy_index = find(rho <= rho_threshold);
    % Noisy labels检测统计：
    % 正确检测: '1' (N -> N)
    % 错误检测: '-1' (T -> N)
    % 未检测到: '0' (N -> ×)
    % 真实样本: '2' (T)
    Stat_matrix = ones(length(rho),1)*2;
    Si = find(noisy_index <= num);
    Stat_matrix(noisy_index(Si)) = 1;
    Se = find(noisy_index > num);
    Stat_matrix(noisy_index(Se)) = -1;
    N = 1:num;
    [~,Ni,~] = intersect(N,noisy_index);
    N(Ni) = [];
    Stat_matrix(N) = 0;
    detec_result = [detec_result,Stat_matrix];
    % 检测 end
    % detection_index{i} = noisy_index;
    per_sample_index(:,noisy_index) = [];
    lable_correct = ones(1,length(per_sample_index))*c;
    temp_data = [per_sample_index;lable_correct];
    train_correct = [train_correct,temp_data];
end