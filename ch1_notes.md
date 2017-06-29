CHAPTER 1
----------

see jupyter notebook `ch1_example1.ipynb`.

INTRO

Supervised learning: user provides the algorithm with inputs and desired outputs, and the algorithm finds a way to produce the desired output given an input. ("training") Thus it will eventually be able to create an output for an input it has never seen before without the help of a human.

* Use this method if you are able to create a dataset that includes the desired outcome. 

Unsupervised learning: only the input data is known and no known output data is given to the algorithm. 

* Example: If have a large collection of text data, you might want to summarize it and find prevalent themes. You don't known beforehand what these topics are or even how many topics there might be. Therefore, there are no known outputs.

For both types of learning, need to have representation of your input data that a computer can understand. Think of your data as a table. Each row of the table is known as a "sample" or "data point", while the columns (the properties that describe these entities) are called "features".

APPLICATION

You need to split your dataset into "training" and "testing" sets. scikit-learn has a function to do this: `train_test_split`, where 75% of the rows are extracted as the training set. The function first shuffles the rows with a pseudorandom number generator.

In skikit-learn data is denoted with `X`, while labels are denoted with `y`, inspired by f(x)=y in mathematics, where x is the input to the function and y is the output. 

The possible outputs are called "classes" and a single output is called the "label".

There are several classification algorithms in `scikit-learn`. 
"k-nearest neighbors" is one of easier to understand. 
To make a prediction for a new data point, the algorithm finds the point in the training set that is closest to the new point. Then it asssigns the label of this training point to the new data point.  The "k" signifies that instead of using ONLY the closest neighbor to the new data point, we consider any fixed number k of neighbors in the training set (for example, the closest three or five neighbors.) Then we can make a prediction using the majoriting class among these neighbors.

All machine learning models in `skikit-learn` are implemented in their own classes, called `Estimator` classes.  The k-nearest neighbors classification algorithm is implemented in the `KNeighborsClassifier` class in the `neighbors` module. 

	from sklearn.neighbors import KNeighborsClassifier
	knn = KNeighborsClassifier(n_neighbors=1)

The knn object encapsulates the algorithm that will be used to build the model from the training data, as well the algorithm to make predictions on new data points. Will also hold information that the algorithm has extracted from the training data.

Next need build the model on the training set. To do this, call the `fit` method of the knn object. 

	knn.fit(X_train, y_train)

Now can make prodictions using this model on new data. For the 
example of a new Iris,

	X_new = np.array([[5, 2.9, 1, 0.2]])

Make prediction of its species:

	prediction = knn.predict(X_new)

How do we know the prediction is right? We need to TEST the model.

	y_pred = knn.predict(X_test)
	print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

 
CODE SUMMARY

	X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], random_state=0)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
	print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))