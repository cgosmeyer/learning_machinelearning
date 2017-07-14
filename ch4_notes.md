# CHAPTER 4: REPRESENTING DATA AND ENGINEERING FEATURES

see see jupyter notebook `ch4.ipynb`.

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

Linear models can be made more powerful on continuous data using *binning* (aka *discretization*) of the feature to split it up into multiple features.

For example, imagine a partition of the input range for the feature (numbers -3 to 3) into a fixed number of bins, say, 10.  The data point will then be represented by which bin it falls into. 





