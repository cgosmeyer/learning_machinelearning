# CHAPTER 4: REPRESENTING DATA AND ENGINEERING FEATURES

see jupyter notebook `ch4.ipynb`.

CONTINUOUS FEATURES: Numeric. Examples are pixel brightness, size measurements.

CATEGORICAL/DISCREET FEATURES: Usually not numeric. Examples are brand or color of a product or name of the department that it was sold in. No middle ground - either it belongs to the book or clothing department.

FEATURE ENGINEERING: Task of best representing your data for a particular application. 

## CATEGORICAL VARIABLES

How can we represent categorical variables when applying logistic regression?  Applying the formula doesn't make much sense when x[i] are not numbers.

One way to represent categorical variables is using the *one-hot-encoding*, also known as *dummy variables*. Replace categorical variables with one or more new features that can have the values of 0 and 1.  For example, a `workclass` feature with possible values "Gov Employee", "Private Employee", "Self Employed", and "Self Employed Incorporated" can be encoded as follows:

	workclass 	Gov		Private		Self	Self Inc
	Gov			1		0			0		0
	Private		0		1			0		0
	Self		0		0			1		0
	Self Inc 	0		0			0		1

When using this data in a machine learning algorithm, we would drop the original `workclass` feature and keep only the four new 0-1 features.

Pandas offers a way to do this:

	data_dummies = pd.get_dummies(data)

## BINNING, DISCRETIZATION, LINEAR MODELS, AND TREES

Linear models can be made more powerful on continuous data using *binning* (aka *discretization*) of the feature to split it up into multiple features. (More complicated models like trees don't benefit as much.)

For example, imagine a partition of the input range for the feature (numbers -3 to 3) into a fixed number of bins, say, 10.  The data point will then be represented by which bin it falls into. 

## INTERACTIONS AND POLYNOMIALS

Another way to enrich a feature representation, particularly for linear models, is adding *interaction features* and *polynomial features* of the original data. Often used in statistical modeling.

An interaction or product feature can be added in order to indicate which bin a point is in *and* where it lies on the x-axis. It is a product of the bin indicator and the original feature. 

Another option besides binning ot expand a continuous feature is to use polynomials of the original features. For example, for a given feature x we might want to consider x**2, x**3, etc.

Each degree adds a feature. 10 degrees = 10 features.

Using polynomial features together with a linear regression model yields the classical model of polynomial regression. The polynomial yields a very smooth fit but can behave in extreme ways on the boundaries or in regions with little data.

## UNIVARIATE NONLINEAR TRANSFORMATIONS

Some transformations, such as mathematical function `log`, `exp`, or `sin`, are useful for certain features.

Linear models and neural networks are tied to the scale and distribution of each feature and if there is a nonlinear relation between the feature and the target, it becomes hard to model, especially in regression.

The functions `log` and `exp` can adjust the scale in the data so they can be captured better by a linear model or neural network.

*Most models work best when each feature is loosely Gaussian distributed.*  Transformations like `log` and `exp` is a simple way to do this.

## AUTOMATIC FEATURE SELECTION

*Adding more features will make a model more complex and therefore increase the chance of overfitting.*  It could be a good idea to reduce the number of features to only the most useful ones and discard the rest. But how do you know how good a feature is?  Three strategies are *univariate statistics*, *model-based selection*, *iterative selection*. All are supervised methods, so they need the target for fitting the model, meaning we need to split the data into training and test sets and fit the feature selection only on the training part of the data.

### Univariate Statistics

Computes whether there is a statistically significant relationship between each feature and the target.  The features that relate with the highest confidence are selected. Also known as *analysis of variance* (ANOVA). These tests are called *univariate* because they only consider each feature individually. 

### Model-Based Feature Selection

Uses a supervised machine learning model to judge the importance of each feature and keeps only the most important ones. Unlike Univariate Statistics it considers all features at once and so can capture interactions.

### Iterative Feature Selection

Builds a series of models to select features.  A series of models are built with varying numbers of features.  More computationally expensive than previous two methods. 

## EXPERT KNOWLEDGE

Can make use of expert knowledge. You as expert can add features to the data, such as holidays to a dataset of airport prices. 
















