function [indexes]=train_test_random_new(y,Value)
% function to ramdonly select training samples and testing samples from the
% whole set of ground truth.
% alltrain is the ground truth

K = max(y);

% generate the  training set£¨²úÉúÑµÁ·¼¯£©
indexes = [];
indexes_c=[];

for i=1:K
    index1 = find(y == i);
    per_index1 = randperm(length(index1));   
    Number=per_index1(1:Value(i));
    indexes_c=[indexes_c;index1(Number)'];   
end  
  indexes = indexes_c(:);