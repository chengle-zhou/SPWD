function [ OA,AA,kappa,CA ] = My_SVM_Classifier( img,train_samples,train_labels,test_SL,GroudTest)
[no_lines, no_rows, no_bands] = size(img);
img = reshape(img, no_lines * no_rows, no_bands);
[train_samples,M,m] = scale_func(train_samples);
[img] = scale_func(img,M,m);
[Ccv2 Gcv2 cv cv_t] = cross_validation_svm(train_labels,train_samples);
parameter = sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv2,Gcv2); 
model = svmtrain(train_labels,train_samples,parameter);
SVMresult = svmpredict(ones(no_lines*no_rows,1),img,model); 
SVMResultTest = SVMresult(test_SL(1,:),:);
[OA,AA,kappa,CA] = confusion(GroudTest,SVMResultTest);
end