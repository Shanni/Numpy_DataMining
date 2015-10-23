by Shan Liu
To test the code type on command line:
	$python3.4 a4main.py iris.txt 30 .95

Brief Summery:

This assignment is on evaluating the performance of the full Bayes classifier and naive Bayes classifier using the paired t-test.

Consider dataset iris.txt. It has 4 numeric dimensions and and a fifth column indicating the class.

Implement Algorithm 22.4, Paired t-Test via Cross-Validation, in Chapter 22, except to use K sounds of bootstrap resampling in place of K-fold cross validation.

In each round, (1) compute the training set Xi using bootstrap resampling (Algorithm 22.3, line 2), (2) learn the parameters of the Bayes classifiers---full and naive (Algorithm 18.1, lines 1-8), (3) compute the testing set X-Xi, (4) test both classifiers on X-Xi (Algorithm 18.1, line 9) and compute the number of errors and error rate for each (Algorithm 22.4, line 5), and (5) tabulate the difference in performance, i.e., error rates, of the full and naive Bayes classifiers (Algorithm 22.4, line 6).

Then, compute the z-score value from the tabulated differences (Algorithm 22.4, lines 7-9).

Finally, test if the two classifiers are different or not (Algorithm 22.4, lines 10-13). To find out the critical value, i.e., bound, for the t-distribution for a given confidence level α you may find the scipy.stats module useful.

Note that the only difference between the full and naive Bayes classifiers is that the former uses the full covariance matrix, whereas the latter uses the diagonal one.

