# Ubalanced_classes

In solving many practical problems of machine learning methods, researchers are faced with the fact that the present imbalance in the training set classes, ie, classes are uneven (imbalanced dataset). 

Most classification algorithms aimed at minimizing the overall error learning. For example, problems in object recognition in the image goal are to minimize the number of false positives at a sufficient frequency of true detections. The number of examples for the "object" is small compared with the number of example of the "background" is called unbalanced distribution classes. Using standard methods of classification in this situation is often a problem that reducing the overall error qualifier applies fully interested class noise.

Below is a list of the methods currently implemented in the module. This article discusses only some of them. 

Under-sampling

    •	Random majority under-sampling with replacement
    
    •	Extraction of majority-minority Tomek links
    
    •	Under-sampling with Cluster Centroids
    
    •	NearMiss-(1 & 2 & 3)
    
    •	Condensend Nearest Neighbour
    
    •	One-Sided Selection
    
    •	Neighboorhood Cleaning Rule


Over-sampling

    •	Random minority over-sampling with replacement

    •	SMOTE - Synthetic Minority Over-sampling Technique

    •	bSMOTE(1&2) - Borderline SMOTE of types 1 and 2

    •	SVM_SMOTE - Support Vectors SMOTE


Over-sampling followed by under-sampling

    •	SMOTE + Tomek links

    •	SMOTE + ENN


Ensemble sampling

    •	EasyEnsemble

    •	BalanceCascade


We will use an UnbalancedDataset which is a python module offering a number of resampling techniques commonly used in datasets showing strong between-class imbalance.


