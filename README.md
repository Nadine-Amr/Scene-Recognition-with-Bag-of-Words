# Scene-Recognition-with-Bag-of-Words

Based on the assignment developed by James Hays and Sam Birch, scenes are recognized as belonging to one of fifteen categories. This is done in three different ways: (1) using tiny images representation of images and a nearest neighbor classifier, (2) using the bag of words representation and a nearest neighbor classifier and (3) using the bag of words representation and a support vector machine (SVM) classifier. Several versions of the three algorithms were tried out.

# Running the Code

Please download the "data" folder from this link: 
https://drive.google.com/open?id=1_LUG3wNzLz7lH_KvfDpbrWncDc9VeMgm 
and place it in the same folder as the "code" folder before you run.

In order to test the second of the three different algorithms, kindly call the function "projSceneRecBoW()". It will calculate the scene recognition accuracy using the bag of words representation and the nearest neighbor classifier. The "vocab" matrix with 500 clusters is provided so that the vocabulary does not have to be rebuilt.

To test any other combination of algorithms, please change the "FEATURE" and "CLASSIFIER" variables in the function "projSceneRecBoW()" as appropriate.
