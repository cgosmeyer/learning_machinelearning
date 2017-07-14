# CHAPTER 3: UNSUPERVISED LEARNING

see see jupyter notebook `ch3.ipynb`.

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

*Another application of PCA is feature extraction.* 

You can try to do facial matching with `KNearestNeighbors`, etc, but can do better with PCA using the *whitening* option to rescale principle components to the same scale. 

Removing components is like removing terms from a weighted sum. 

### Non-Negative Matrix Factorization (NMF)

Aim is to extract useful features. But unlike PCA, where we wanted components that were orthogonal and explained as much variance of the data as possible, in NMF we want the components and the coefficients to be non-negative. Therefore we can only apply NMF to data where each feature is non-negative.

This method is helpful for data that is crated as the addition of several independent sources, such as an audio track of multiple people speaking or music with many instruments; here NMF can identify original components. 

### Manifold Learning with t-SNE

A class of algorithms for visualization that allow for more complex mappings and often provide better visualizations.

Manifold algorithms are mainly for visualization and are rarely used to generate more than two features. They cannot be applied to test sets; they can only transform the data they were trained for. 


## CLUSTERING

The task of partitioning dataset into groups, called clusters. Goal is to split up the data in such a way that points within a single cluster are very similar.  Similarly to classification algorithms, clustering algorithms assign (or predict) a number to each data point, indicating whihc cluster a particular point belongs to.

### K-Means Clustering

One of the simplest and most commonly used clustering algorithms. Works in alternating two steps:

	1. Assign each data point to the closest cluster center. 

	2. Set each cluster center as the mean of the data points assigned to it.

The algorithm is finished when the assignment of instances to clusters no longer changs.

K-means, however, always assumes all clusters have same "diameter" and can only capture relatively simple shapes. It also assumes all directions are equally important for each cluster; it will fail to identify nonspherical clusters.

VECTOR QUANITIZATION: View of k-means as a decomposition method, where each point is represented using a single component.

With k-means, unlike PCA or NMF, can use many more clusters than input dimensions to encode the data. That is, effectively add more dimensions.

Downside to k-means is that it relies on random intialization. By default, the algorithm is run 10 times with 10 different random initializations and the best result is returned. 

### Agglomerative Clustering

A collection of clustering algorithms that start by declaring each point its own cluster, and then merges the two most similar clusters until some stoping criterion is satisfied. In `scikit-learn` the stopping criterion is the number of clusters. 

`AgglomerativeClustering` cannot make predictions for new data points and therefore has no `predict` method. 

Agglomerative clustering also fails at seperating complex shapes.

### DBSCAN

"Density-based spatial clustering of applications with noise".

Does not require the user to set the number of clusters a priori, it can capture clusters of complex shapes, and it can identify points that are not part of any cluster.

The idea behind DPSCAN is that clusters form dense regions of data, separated bt regions that are relatively empty.  Three kinds of points in the end:

	1. Points within dense regions are called *core samples*.

	2. Points that are within distance `eps` of core points called *boundary points*.

	3. Points that don't belong to any cluster, called *noise*.

Not possible to create more than one big cluster using DBSCAN; agglomerative clustering and k-means are more likely to create clusters of even size. 


## COMPARING AND EVALUATING CLUSTERING ALGORITHMS

How to assess a clustering algorithm relative to a ground truth clustering?

Two metrics are *adjusted rand index (ARI)* and *normalized mutual information (NMI)*, with provide a quantitative measure with an optimum of 1 and a value of 1 for unrelated clusterings.

But usually you don't know the ground truth; if you did, you would use a classifier. Therefore, metrics like ARI and NMI usually only help in developing algorithms, not in assessing success in an application. 

*silhoutte coefficient* is a metric that doesn't require ground truth. However, it doesn't often work well in practice. 

*robustness-based* clustering metrics are slightly better. 
