# CHAPTER 2: UNSUPERVISED LEARNING

see see jupyter notebook `.ipynb`.

In unsupervised learning, no known output, no teacher to instruct the learning algorithm. The learning algorithm is shown the input data and asked to extract knowledge from this data. 

We'll look at two kinds of unsupervised learning: transforms of the dataset and clustering.

UNSUPERVISED TRANSFORMS: Algorithms that create a new representation of the data which might be easier for algorithms to understand than the original representation.

CLUSTERING ALGORITHMS: Partition data into distinct groups of similar items. 

It is difficult to say whether an algorithm "did well" and often need to inspect results manually. Therefore unsupervised learning is often used in an exploratory setting, where data scientist wants to understand the data better, rather than as part of a larger automatic system. Often they are used to preprocess data for supervised algorithms. 


## PREPROCESSING AND SCALING

It is important to apply the exact same transformation to the training set and the test set for the supervised model to work on the test set. 

For example,

	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train_scaled = scaler.transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	# learning an SVM on the scaled training data
	svm.fit(X_train_scaled, y_train)


## DIMENSIONALITY REDUCTION, FEATURE EXTRACTION, AND MANIFOLD LEARNING

A simple algorithm for visualizing, compressing data, and finding more informative representation of data is *principle component analysis (PCA)*. Two others are *non-negative matrix factorization (NMF)*, commonly used for feature extraction, and *t-SNE*, commonly used for visualization using 2D scatter plots.

### Principle component analysis (PCA)

A method that rotates the dataset in such a way that the rotated features are statistically uncorrelated. 

Breaks data down into principle components, rotates, and removes components as desired. (so can go from 2D representation of data to 1D, for example)

One of most common applications is visualizing high-dimenstional datasets. 

By default `PCA` only rotates and shfits the data, keeping all principle components. To reduce dimensionality of data, we need to specify how many components we want to keep when creating the `PCA` object. 
