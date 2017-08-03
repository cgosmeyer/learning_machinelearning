# CHAPTER 6: ALGORITHM CHAINS AND PIPELINES

see jupyter notebook `ch6.ipynb`.

Most machine learning applications require the chaining together of many different processing steps and machine learning models.  The `Pipeline` class helps to simply this process. Also helps eliminates mistakes you can make in code, since will be writing less code.

To avoid contaminating test data with information from rest of dataset, splitting of the dataset during cross-validation should always be done before doing and preprocessing (ie, scaling).

The `Pipeline` class is commonly used to chain preprocessing steps (like scaling data) together with a supervised model like a classifier. Another primary function is to do grid searches.

## BUILDING PIPELINES


Use Pipeline class to extress workflow for training an SVM after scaling the data with MinMaxScaler. First, build Pipeline object by providing it with a list of steps.

	from sklearn.pipeline import Pipeline
	# two steps: scaler and svm
	pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
	# Now fit the pipeline
	pipe.fit(X_train, y_train)
	# Evaluate
	print("Test score: {:.2f}".format(pipe.score(X_test, y_test)))

With the pipeline, we reduced code needed for preprocessing and classification process. Both `pipe_long` and `pipe_short` do the same things.

There exists a convenience function to make code for creating a pipeline even shorter. 

	from sklearn.pipeline import make_pipeline
	# standard syntax
	pipe_long = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC(C=100))]) 
	# abbreviated syntax
	pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))

Can even do grid searches over multiple different models. 
