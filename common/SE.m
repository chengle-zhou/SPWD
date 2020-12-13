function  [train_joint_sample,train_joint_data] = SE(GroundT,train_noisy,index_map,gt,img_2d,Suerplabels)


index = train_noisy(:,3)';
class = train_noisy(:,1);
for p = 1:size(index,2)
    single_index = index(p);
    super_index = find(Suerplabels == Suerplabels(single_index));
    jonit_index = index_map(super_index);
    [gt_jonit_index,Ai,Bi] = intersect(GroundT(1,:), jonit_index'); 
    train_joint_index = gt_jonit_index;
    train_ture_label = gt(gt_jonit_index);
    train_noisy_label = ones(1,length(gt_jonit_index))*class(p);
    train_joint_sample{p} = [train_joint_index;train_noisy_label;train_ture_label];
    train_joint_data{p} = img_2d(gt_jonit_index,:);


end
% train_joint_sample = [train_joint_index;train_ture_label;train_noisy_label];