% Starter code prepared by James Hays for CSCI 1430 Computer Vision

% This function will extract a set of feature descriptors from the training images,
% cluster them into a visual vocabulary with k-means,
% and then return the cluster centers.

% Notes:
% - To save computation time, we might consider sampling from the set of training images.
% - Per image, we could randomly sample descriptors, or densely sample descriptors,
% or even try extracting descriptors at interest points.
% - For dense sampling, we can set a stride or step side, e.g., extract a feature every 20 pixels.
% - Recommended first feature descriptor to try: HOG.

% Function inputs: 
% - 'image_paths': a N x 1 cell array of image paths.
% - 'vocab_size' the size of the vocabulary.

% Function outputs:
% - 'vocab' should be vocab_size x descriptor length. Each row is a cluster centroid / visual word.

function vocab = build_vocabulary( image_paths, vocab_size )

N = size(image_paths,1);
step = 1;
k = vocab_size;
features = [];

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
%     [im_features,validPoints] = extractHOGFeatures(image,corners);
%     [im_features] = extractHOGFeatures(image);
    [im_features, validPoints] = extractFeatures(image, corners, 'Method', 'SURF', 'SURFSize', 128);
        
     features = [features; im_features];
end
%%
[vocab, idx] = vl_kmeans(features',vocab_size);
vocab = vocab';