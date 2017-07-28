# CHAPTER 5: MODEL EVALUATION AND IMPROVEMENT

see jupyter notebook `ch5.ipynb`.

Will focus on evaluating models and selecting parameters, primarily for supervised methods.

To evaluate supervised models, we have so far been using the `score` method, which for classification computes the fraction of correctly classified samples. We will explore more robust methods.

## CROSS-VALIDATION

A statistical method of evaluating generalization performance. In cross-validation the data is split repeatedly and multiple models are trained.

*k-fold cross-validation* is more commonly used version of cross-validation. *k* is a user-specified number, usually 5 or 10. For example, in 5-fold cross-validation, the data is first partitioned into five parts called *folds*. Then a sequence of models is trained.  The first model is trained using the first fold as the test set and the remaining folds are used as the training set. The model is built using the data in folds 2-5 and then the accuracy is evaluated on fold 1. Then another model is built, but now using fold 2 as the test set and data in the other folds as training set. The process repeats. For each of five splits of data we compute the accuracy. We summarize by computing the mean.

Cross-validation does not build models; its purpose is to evaluate how well a given algorithm will generalize when trained on a specific dataset.

*stratified k-fold cross-validation* ensures that for each fold, all classes are accurately represented. For example, in a dataset 10% class1 and 90% class2, the fold always contains a 1-to-9 ratio.  Always use this for classifier models.  

## GRID SEARCH

Once know how to evaluate how well a model generalizes, can improve the model's generalization performance by tuning its parameters. 

*grid search* will try all possible combinations of the parameters of interest. 

You need to split the data three ways: training set, validation set (for tuning parameters), and test set.

*Always keep a clearn, seperate test set only used for the final evaluation!*

Because cross-validation is often, in practice, combined with a grid search, *cross-validation* colloquially refers to grid search with cross-validation.

## METRICS FOR BINARY CLASSIFICATION

In statistics,

	false positive: type I error

	false negative: type II error

*Confusion matrices:* one of most comprehensive ways to represent the result of evaluating binary classification.

The output of `confusion_matrix` is a two-by-two array where the rows correspond to the true classes and columns correspond to the predicted classes. Entries in the main diagonal correspond to correct classifications. 

Can summarize the results of a confusion matrix as follows:

	accuracy = (TP+TN) / (TP+TN+FP+FN)

ie, correct predictions divided by number of examples. 

*When classes are imbalanced, accuracy is a poor evaluation measure.*

Precision, on the other hand, measures how many of the samples predicted as positive are actually positive:

	precision = TP / (TP+FP)

This is used as a preformance metric when the goal is to limit the number of false positives.

Recall measures how many of the positive samples are captured by positive predictions:

	recall = TP / (TP+FN)

This is used when we need to identify all positive samples; that is, when it is important to avoid false negatives. 

One way to summarize recall and precision is with *f-score*:

	F = 2 * (precision*recall) / (precision+recall)

There is a trade off between optimizing recall and optimizing precision.

The function `classification_report` provides precision, recall, f1-score, and support (the number of samples in this class according to ground truth).

A *ROC* curve considers all possible thresholds for a given classifier and plots the false positive rate (FPR) against the true positive rate (TPR). 

	FPR = FP / (FP+TN)

The ideal curve is close to the top left; you want a classifier that produces a high recall while keeping a low false positive rate.  ROC curves are often summarized by computing the area under the curve (AUC), where 0 is worst and 1 is best.  Predicting randomly results in an AUC of 0.5. 

## METRICS FOR MULTICLASS CLASSIFICATION

The metrics for multiclass classifcation are derived from binary, but averaged over all classes. 

## REGRESSION METRICS

For most applications, using the default R^2 from the `score` method is enough. 



