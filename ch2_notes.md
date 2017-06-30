# CHAPTER 2

see see jupyter notebook `ch2.ipynb`.

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
The consequence of L1 is that some coefficients are *exactly zero*, meaning some featuers are entirely ignored by the model. This is a form of automatic feature selection. Making some coefficients exactly zero often makes a model easier to interpret and can reveal the most important features of your model. 

Ridge is usually used as a first choice. 
But if you have a large amount of features and expect only a few to be important, `Lasso` may be the better choice.


## ELASTICNET

Combines penalties of `Lasso` and `Ridge`. But in `ElasticNet` ou need to adjust two parameters: one for L1 regularization and one for L2 regularization.


## LINEAR MODELS FOR CLASSIFICATION


