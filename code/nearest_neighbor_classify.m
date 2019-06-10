% Starter code prepared by James Hays for CSCI 1430 Computer Vision

%This function will predict the category for every test image by finding
%the training image with most similar features. Instead of 1 nearest
%neighbor, you can vote based on k nearest neighbors which will increase
%performance (although you need to pick a reasonable value for k).

function predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
% image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation.
% train_labels is an N x 1 cell array, where each entry is a string
%  indicating the ground truth category for each training image.
% test_image_feats is an M x d matrix, where d is the dimensionality of the
%  feature representation. You can assume M = N unless you've modified the
%  starter code.
% predicted_categories is an M x 1 cell array, where each entry is a string
%  indicating the predicted category for each test image.

% Useful functions:
%  matching_indices = strcmp(string, cell_array_of_strings)
%    This can tell you which indices in train_labels match a particular
%    category. Not necessary for simple one nearest neighbor classifier.
 
%   [Y,I] = MIN(X) if you're only doing 1 nearest neighbor, or
%   [Y,I] = SORT(X) if you're going to be reasoning about many nearest
%   neighbors 
% %% Manually Computing Euclidean Distance
% 
% test_size = size(test_image_feats,1);
% predicted_categories = cell(test_size,1);
% categories = unique(train_labels)';
% k = 25;
% 
% for i=1:test_size
%     distances =  bsxfun(@minus,train_image_feats, test_image_feats(i,:));
%     distances = distances.^2;
%     distances = sqrt(sum(distances'));
%     
%     [distances, index] = sort(distances);
%     
%     idx_nearest_k = index(:,1:k);    
%     labels_nearest_k = train_labels(idx_nearest_k);
%     
%     categories_count = zeros(1,15);
% 
%     for j = 1:k
%         categories_count = categories_count + double(strcmp(labels_nearest_k(j), categories));
%     end
%     
%     [max_count, idx_max_count] = max(categories_count);
%     
%     predicted_categories(i) = categories(idx_max_count);
% end
%% KNN-Search

test_size = size(test_image_feats,1);
predicted_categories = cell(test_size,1);
categories = unique(train_labels)';
k = 25;

[idx_nearest_k, dist_nearest_k] = knnsearch(train_image_feats, test_image_feats, 'k',k);

labels_nearest_k = train_labels(idx_nearest_k);
for i = 1:test_size
  
    categories_count = zeros(1,15);
    
    for j = 1:k
        categories_count = categories_count + double(strcmp(labels_nearest_k(i,j), categories));
    end
   
    [max_count, idx_max_count] = max(categories_count);
    predicted_categories(i) = categories(idx_max_count);
end









