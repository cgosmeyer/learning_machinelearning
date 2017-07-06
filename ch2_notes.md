# CHAPTER 2

see see jupyter notebook `ch2_nearest_linear.ipynb`.

Two types of supervised machine learning problems:
1) CLASSIFICATION
	* Goal is to predict a class label from a predfined list of possibilities. 
	* For example, predicting the Iris species, or predicting the language of a website (French or English).

2) REGRESSION
	* Goal is to predict a continuous number ("floating-point" in programming speak or "real" in mathematical peak).
	* For example, predicting a person's annual income from age, education, and location.

A model that can make accurate predictions on unseen data is able to "generalize" from the training set to the test set.  We want a model to be able to generalize as accurately as possible. 

We also want the simplest model with the fewest rules. We don't want to "overfit", that is, fit the model too closely to particulars of the training set. Choosing too simple a model, in converse, is called "underfitting". The model we want will yield the best generalization performance; it will be at the sweet spot between under and overfitting. 

In general, the more data in your training set, the harder to overfit. 

You evaluate regression models with the R^2 model. 
R^2 score (also known as "coefficient of determination") is a measure of goodness of prediction and yields a score between 0 and 1. 
1 = perfect prediction.
0 = constant model that just predicts the mean of the training set responses.

**Sign of underfitting: testing and training score both low and similar**

**Sign of overfitting: training score high, testing score low**


## DRAWBACK OF K-NEAREST NEIGHBORS:

Slow - doesn't work well when many features. In practice, not often used, although it is the easist algorithm to understand.


## LINEAR MODELS

A class of models widely used in practice. They make predictions using a linear function of input features. 

For a dataset with a single feature, you get the equation for a line:

	y = w[0]*x[0] + b

where 
	w[0] is the slope 
	b is the y-axis offset
	y is the prediction the model makes
	x[0] denotes the feature 0

For multiple features (p):

	y = w[0] * x[0] + w[1] * x[1] + ... + w[p] * x[p] + b

You can also think of the predicted response as being the weighted sum of the input features, with weights given by w.

Linear models for regression can be characterized as regression models for which the prediction is a line for single feature, a plane when using two features, or a hyper-plane in higher dimensions.

If you have more features that training data points, any target y can be perfectly modeled (on the training set) as a linear function.

There are many different linear models for regression, for which the difference is in how hte model parameters w and b are learned from the traiing data and how the model complexity can be controlled.

The main parameter of linear models is the regularization parameter, `alpha` in regression models and `C` in classification models.  Another decision is whether to use *L2* or *L1* regularization. 

Linear models are very fast to train and fast to predict. They often perform well when the number of features is large compared to the number of samples.


## LINEAR REGRESSION (ORDINARY LEAST SQUARES)

The simplest and most classic linear method for regression. It finds the parameters w and b that minimize the mean squared error between prediction and tree regression targest, y, on the training set. It has no parameters.

mean squared error = sum of the squared differences between the predictions and the true values, divided by the number of samples

	* "slope" parameters (w), aka weights or coefficients are stored in `coef_` attribute. A np array with one entry per input feature.
	* "intercept" (b) is stored in `intercept_` attribute. A single float number.


## RIDGE REGRESSION

Unlike linear regression, allows us to control complexity (so if overfitting, should turn here). Ridge regression is one of most common alternatives to standard linear regression.

Same formula as used for ordinary least squares BUT the coefficients (w) are chosen not only so that they predict well on the training data, but also so that they fit an additional constraint. 

REGULARIZATION: explicitly restricting a model to avoid overfitting. 

Ridge regression uses *L2* regularization. 

If we are only interested in generalized performance, we should choose `Ridge` model over `LinearRegression` model. You'll have worse performance on training set but better generalization to the test set.

How much you trade-off between simplicity of the model (near-zero coefficients) and its performance on the training set is specificed by the `alpha` parameter.

	alpha = 1.0 is default

	Increasing alpha forces coefficients to move toward zero. This decreases training set performance but might help generalization.

	Decreasing alpha allows the coefficients to be less restricted (i.e., we remove the effect of regularization) and increases chance of overfitting. The more we decrease it, the closer it comes to LinearRegression (`alpha`=0).

LEARNING CURVES: plots that show model performance as function of dataset size.

**Given enough data, ridge and linear regression will have the same performance.**


## LASSO

An alternative to `Ridge` fir regularizing linear regression.

Like ridge, restricts coefficients to be close to zero, but using *L1* regularization.  
The consequence of *L1* is that some coefficients are *exactly zero*, meaning some featuers are entirely ignored by the model. This is a form of automatic feature selection. Making some coefficients exactly zero often makes a model easier to interpret and can reveal the most important features of your model. 

Ridge is usually used as a first choice. 
But if you have a large amount of features and expect only a few to be important, `Lasso` may be the better choice.


## ELASTICNET

Combines penalties of `Lasso` and `Ridge`. But in `ElasticNet` ou need to adjust two parameters: one for L1 regularization and one for L2 regularization.


## LINEAR MODELS FOR CLASSIFICATION

For binary classification:

	y = w[0] * x[0] + w[1] * x[1] + ... + w[p] * x[p] + b > 0

Where the predicted value is thresholded at zero. If function is less than zero, predict class -1. If greater than zero, predict class +1. 

Unlike linear models for regression, where y is a linear output of the features, in linear models for classification the *decision boundary* is a linear function of the output. 

Two common linear classification algorithms are LOGISTIC REGRESSION and LINEAR SUPPORT VECTOR MACHINES (linear SVMs); `linear_model.LogisticRegression` and `svm.LinearSVC`, respectively.

	Parameter "C" determines strength of the regularization.

	Higher C = less regularization - tries to fit training set best as possible

	Lower C = more regularization - model tries harder to find a coefficient vector (w) that is close to zero. 

Note that `LogisticRegression applies an *L2* regularization by default.

Using *L1* instead could be helpful because it limits the model to using only a few features. 


## LINEAR MODELS FOR MULTICLASS CLASSIFICATION

Many linear classification models are only for binary classes (except for logistic regression).  One way to extend binary classification algorithms to a multiclass classification algorithm is the *one-vs.-rest* approach.

ONE-VS-REST: a binary model is learned for each class that tries to seperate that class from all of the other classes, resulting in as many binary models as there are classes.  

To make a prediction, all binary classifiers are run on a test point. The classifier that has the highest score on its single class "wins" and this class label is returned as the prediction.

Therefore, one vector of coefficients (w) and one intercept (b) for each class. 

-------------------------------------------------------------------------------

see jupyter notebook `ch2_decisiontrees.ipynb`

## NAIVE BAYES CLASSIFIERS

Even faster in training than linear models. But often have worse generalization performance. 

Three kinds implemented in `scikit-learn`: `GaussianNB`, `BernoulliNB`, and `MultinomialNB`. 

GAUSSIANNB: Can be applied to any continuous data. Stores average value and standard dev of each feature for each class. 

BERNOULLINB: Assumes binary data. Counts how often every feature of each class is not zero. 

MULTINOMIALNB: Assumes count data. Like Bernoulli, only takes into account the average value of each feature for each class.

Bernoulli and Mulitnomial are often used in text classification. 


## DECISION TREES

Learn a hierarchy of if/else questions, leading to a decision. Used for classification and regression tasks. Work well if you have features on completely different scales or a mix of binary and continuous features. 

You don't need to build the models by hand - you can learn them from data using supervised learning.

Learning a decision tree means learning the sequence of if/else questions that gets us the true answer most quickly. In machine learning, these questions are called *tests*. In binary datasets, the tests are yes/no. In continous data, the tests are in form "Is feature *i* larger than value *a*?"

To build a tree, the algorithm searches over all possible tests and finds the one that is most informative about the target variable. Building a tree like this, continuing until all leaves are pure leads, leads to a highly complex model that is overfit to the training dataset; pure leaves mean that a tree is 100% accurate on the training set. 

To prevent over-fitting:

	1. Stop the creation of the tree early (pre-pruning). Do this by limiting the max depth of tree, max number of leaves, or requiring a minimum number of points in a node to keep splitting it. 

	2. Build the tree but then remove or collapse notes that contain little information (pruning or post-pruning)

Implemented in `skikit-learn` in the `DecisionTreeRegressor` and `DecisionTreeClassifier` classes.  It *only* implements pre-pruning. 

FEATURE IMPORTANCE: a way to summarize a tree. Rates how important each feature is for the decision a tree makes. It is a number between 0 and 1 for each feature, where 0 means "not used at all" and 1 means "perfectly predicts the target". The feature importances always sum to 1.

Regression trees are similar to classification trees. Only difference is that regression models are not able to *extrapolate* or make predictions outside of the range of the training data. 

Advantages of decision trees: 

	* Resulting model can be easily visualized and understood by non-experts.

	* Algorithms are completely invariant to scaling of the data. No pre-processing like normalization or standardization of features is needed. 

Downsides of decision trees:

	* Tend to overfit and provide poort generalization performance. In practice, ensemble methods are used in place of a single decision tree.


## Ensembles of Decision Trees

Ensembles combine multiple machine learning models to create more powerful models. 
Two decision tree ensembles are 

RANDOM FORESTS: Many decision trees, slightly different by randomizing the selected data posted used to build a tree and the selected features in each split test. Final fit will be an average of them all, reducing the overfitting of a single tree. 

GRADIENT BOOSTED DECISION TREES: Builds trees in a serial manner, where each tree tries to correct the mistakes of the previous one. By default, no randomization and instead strong pre-pruning. Often use very shallow trees, depths one to five. 

Benefits of random forests are that they can be easily parallelized; you can make a compact representation of the decision process. If want results to be reproducable, fix the `random_state` or include more trees. Random forests don't perform well on very high dimensional, sparse data, such as text data. 

	Set `n_jobs=-1` to use all cores on your computer

	Random forest rule of thumb: build as many trees as you have time/memory for.

Try random forests before gradient boosted decision trees, unless want faster prediction time. 


-------------------------------------------------------------------------------

## KERNELIZED SUPPORT VECTOR MACHINES (SVMs)

see jupyter notebook `ch2_support_vector_machines.ipynb`

An extension of linear support vector machines that allow for more complex models that are not defined simply by hydroplanes in the input space.

Linear models can be made more flexible by adding interactions or polynomials. Sometimes you cannot seperate classes using just a line! This is where SVMs come in.

Adding nonlinear features to the representation fo the data can make linear models much more powerful. But it could make computation very expensive if we don't know which features to add and if we add many features. A solution to this problem is the KERNEL TRICK, which learns a classifier in a higher-dimensional space without actually computing the new, possibly very large representation.

SUPPORT VECTORS: The subset of training points that are used for defining the decision boundary, lying on the border between the classes. To make a prediction for new point, the distance to each of the support vectors is measured and classification decision is made based on this distance.

	`gamma` parameter determines how far the influence of a single training example reaches.  Low gamma = far reach = large radius for Gaussian kernel (more points considered).  High gamma = limited reach = small radius for Gaussian kernel, leads to more complex model.

	`C` parameter is regularization parameter, limiting the importance of each point.

	By default, C=1 and gamma=1/n_features

SVMs require all features to vary on a similar scale. Often need to pre-process data to bring all features to roughly same scale.


-------------------------------------------------------------------------------

## NEURAL NETWORKS (DEEP LEARNING)

see jupyter notebook `ch2_neural_networks.ipynb`

A family of algorithms, often tailored very carefully to a specific use case. 

One relatively simple deep learning method is Multilayer Perceptrons (MLPs), also known as (vanilla) feed-forward neural networks, or just neural networks.

MLPs can be viewed as generalizations of linear models that perform multiple stages of processing to come to a decision. In an MPL, the process of computing weighted sums is repeated multiple times, first computing HIDDEN UNITS that represent an intermediate processing step. 

What makes this more powerful than a linear model is that a nonlinear function is applied to result of the weighted sum for each hidden unit, allowing the neural network to learn much more complicated functions than a linear model could. 

Two nonlinear functions:

RECTIFYING NONLINEARITY (RELU): Each hidden unit = straight line segment. Add more hidden units to make a smoother boundary.

TANGENS HYPERBOLICUS (TANH): smooth.

	By default, the MLP in `scikit-learn` uses 100 hidden nodes.

	By default, the l2 (`alpha` parameter) is set to a very low value (little regularization)

Weights in neural networks are set randomly before learning is started and this random initiation affects the model that is learned. This can make a difference for smaller networks. 

You must rescale all input features so that they vary in a similar way idally with a mean of 0 and variance of 1.

For larger models, you need to go beyond `scikit-learn` and to the libraries `keras`, `lasagna`, and `tensor-flow`.  These libraries also allow the use of GPUs, while `scikit-learn` does not. 

Neural networks often beat other machine learning algorithms for classification and regression tasks but they often take a long time to train and careful preprocessing of the data. 

-------------------------------------------------------------------------------

## UNCERTAINTY ESTIMATES FROM CLASSIFIERS

see jupyter notebook `ch2_uncertainty.ipynb`

Most classifiers have at least one of two functions for uncertainty estimates: `decision_function` and `predict_proba`. 

The values in the `decision_function` encode how strongly the model believes a data point belongs to the "positive" class.  Positive values indicate preference for the "positive" class.

In binary classification, the "negative" class is always the first entry of the `classes_` attribute and the "positive" class is the second entry. 

The range of the `decision_function` can be arbitrary.

The output of `predict_proba` is a probability for each class. For binary classification, the first entry in each row is the estimated probability of the first class, and the second entry is the estimated probability of the second class. The values are always between 0 and 1 and the sum of the entries for both classes is always 1. 

A model with more complexity (overfitted) makes more "certain" predictions (even if wrong). Less complex models have more uncertainty in its predictions.

CALIBRATED: a model whose reported uncertainty matches how correct it actually is. In a calibrated model, a prediction made with 70% certainty would be correct 70% of the time. 

In multiclass case, `decision_function` has shape (`n_samples`, `n_classes`) and each column provides a "certainty score" for each class. Likewise for `predict_proba`, where again the probabilities for the possible classes for each data point sum to 1.


 





