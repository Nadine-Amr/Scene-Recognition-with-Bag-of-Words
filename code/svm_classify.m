% Starter code prepared by James Hays for CSCI 1430 Computer Vision

function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
% image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation.
% train_labels is an N x 1 cell array, where each entry is a string
%  indicating the ground truth category for each training image.
% test_image_feats is an M x d matrix, where d is the dimensionality of the
%  feature representation. You can assume M = N unless you've modified the
%  starter code.
% predicted_categories is an M x 1 cell array, where each entry is a string
%  indicating the predicted category for each test image.

% This function should train a linear SVM for every category (i.e., one vs all)
% and then use the learned linear classifiers to predict the category of
% every test image. Every test feature will be evaluated with all 15 SVMs
% and the most confident SVM will "win".
% 
% Confidence, or distance from the margin, is W*X + B:
% The learned hyperplane is represented as:
% - W, a row vector
% - B, a scalar bias or offset
% X is a column vector representing the feature, and
% * is the inner or dot product.

%
% A Strategy
% 
% - Use fitclinear() to train a 'one vs. all' linear SVM classifier.
% - Observe the returned model - it defines a hyperplane with weights 'Beta' and 'Bias'.
% - For a test feature point, manually compute the distance form the hyperplane using the model parameters.
% - Store the confidence.
% - Once you have a confidence for every category, assign the most confident category.
% 

% unique() is used to get the category list from the observed training category list. 
% 'categories' will not be in the same order as unique() sorts them. This shouldn't really matter, though.
categories = unique(train_labels);
num_categories = length(categories);
train_size = length(train_labels);
test_size = size(test_image_feats,1);
predicted_categories = cell(test_size,1);
pred_scores = zeros(test_size, num_categories);
% reg_param = 0.2;

%%
for i=1:num_categories
    this_categ = cell(train_size,1);
    this_categ(:) = categories(i);
    this_categ_bin = cellfun(@isequal,this_categ, train_labels);
    SVM_model = fitcsvm(train_image_feats,this_categ_bin);
    
%     [label,score] = predict(SVM_model,test_image_feats, 'BoxConstraint', 1);
    [label,score] = predict(SVM_model,test_image_feats);
    pred_scores(:,i) = score(:,2);
end

for j = 1:test_size
    [the_score, index] = max(pred_scores(j,:));
    predicted_categories(j) = categories(index);
end



