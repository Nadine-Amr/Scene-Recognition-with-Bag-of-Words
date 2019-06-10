% Starter code prepared by James Hays for CSCI 1430 Computer Vision

%This feature representation is described in the handout, lecture
%materials, and Szeliski chapter 14.

function image_feats = get_bags_of_words(image_paths)
% image_paths is an N x 1 cell array of strings where each string is an
% image path on the file system.

% This function assumes that 'vocab.mat' exists and contains an N x feature vector length
% matrix 'vocab' where each row is a kmeans centroid or visual word. This
% matrix is saved to disk rather than passed in a parameter to avoid
% recomputing the vocabulary every run.

% image_feats is an N x d matrix, where d is the dimensionality of the
% feature representation. In this case, d will equal the number of clusters
% or equivalently the number of entries in each image's histogram
% ('vocab_size') below.

% You will want to construct feature descriptors here in the same way you
% did in build_vocabulary.m (except for possibly changing the sampling
% rate) and then assign each local feature to its nearest cluster center
% and build a histogram indicating how many times each cluster was used.
% Don't forget to normalize the histogram, or else a larger image with more
% feature descriptors will look very different from a smaller version of the same
% image.

load('vocab.mat');
vocab_size = size(vocab, 1);

N = size(image_paths,1);
step = 1;
image_feats = [];

for i = 1:step:N
    image = im2double(imread(image_paths{i}));
    
    if length(size(image)) == 3
        image = rgb2gray(image);
    end
        
%     [corners] = detectFASTFeatures(image);
    corners = detectHarrisFeatures(image);
     
    if length(corners) > 500
        corners = corners.selectStrongest(500);
    end 
    
%     [im_features, validPoints] = extractFeatures(image, [x y]);%, 'NumBins', 8, 'BlockSize', [4 4]);
%     [im_features, validPoints] = extractFeatures(image, corners);         
%     [im_features] = double(im_features.Features);  
%     [im_features] = extractHOGFeatures(image);
    [im_features, validPoints] = extractFeatures(image, corners, 'Method', 'SURF', 'SURFSize', 128);

    hist = zeros(1,vocab_size);
    for j = 1:size(im_features,1)
        
        distances =  bsxfun(@minus, vocab, im_features(j,:));
        distances = distances.^2;
        distances = sqrt(sum(distances'));
        
        [min_dist, idx_min_dist] = min(distances);
        hist(idx_min_dist) = hist(idx_min_dist) + 1;
    end
    
    hist_mean = mean(hist);
    hist_std = std(hist);
    hist = hist - hist_mean;
    hist = hist/hist_std;
    image_feats = [image_feats; hist];    
end




